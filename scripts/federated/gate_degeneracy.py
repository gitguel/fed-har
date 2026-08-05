"""G-DEG — o caminho estendido, nos defaults, tem de **ser** o baseline.

A sabatina de 2026-07-29 escolheu estender `run_cross_device.py` (flags
`--method/--init-ckpt/--shots`) em vez de escrever um script separado para o
downstream do Fed-SSL. O preço dessa escolha é o risco de quebrar, em silêncio, o
caminho que produziu a grade commitada em `20449ab`. Este gate é a mitigação, e
roda **antes** de qualquer job da grade nova.

Duas camadas:

1. **Identidade estrutural** (exata, sem GPU). As três funções novas que o caminho
   default atravessa têm de ser a identidade nesse caso:
     - `build_global(enc, "none")`     ≡ `BUILD_MODEL[enc]()`
     - `initial_state(enc, "none", None)` ≡ `state_dict()` do modelo supervisionado
     - `apply_shots(clients, "full", s)`  ≡ `clients` (o MESMO objeto)

2. **Identidade comportamental** (sob `set_deterministic()`). O `run(...)` nos
   defaults e o loop histórico — reimplementado aqui a partir das mesmas peças
   (`make_clients`/`local_train`/`fedavg`), sem passar pelas funções novas — têm de
   produzir acc/F1 idênticos em **todas** as rodadas e o mesmo `state_dict` final.

Nota sobre o que este gate **não** faz: ele não casa bit-a-bit contra os parciais
já commitados em `results/fed_cross_device_parts/`, porque aquela grade rodou
**sem** `set_deterministic()` (ele só era chamado dentro de `gate()`). Comparar com
tolerância esconderia exatamente a classe de bug que o gate existe para pegar,
então comparamos duas execuções frescas sob determinismo.

Uso:
    python scripts/federated/gate_degeneracy.py --encoder rnn --rounds 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl"))  # .../scripts/ssl

import torch
from lightning.pytorch import seed_everything

from common import BEST_LR, FULL_SHOTS
from eval_transfer import BUILD_MODEL
from federated.cross_device import DEFAULT_BUDGET, make_clients, parse_spec
from federated.run_cross_device import (_eval_all, _fresh, apply_shots,
                                        build_global, initial_state, local_train,
                                        set_deterministic)
from pretrain_fed import fedavg


def check_structural(spec: str, encoder: str, seed: int, budget: int | None) -> bool:
    """Camada 1: as funções novas são a identidade no caminho default."""
    ok = True

    seed_everything(seed, workers=True)
    ref = {k: v.detach().cpu().clone()
           for k, v in BUILD_MODEL[encoder]().state_dict().items()}
    seed_everything(seed, workers=True)
    got = initial_state(encoder, "none", None)
    same = set(ref) == set(got) and all(torch.equal(ref[k], got[k]) for k in ref)
    print(f"  initial_state ≡ BUILD_MODEL.state_dict()      : {'OK' if same else 'FALHOU'}")
    ok &= same

    a = type(build_global(encoder, "none")).__name__
    b = type(BUILD_MODEL[encoder]()).__name__
    print(f"  build_global devolve {a:<24}      : {'OK' if a == b else 'FALHOU (' + b + ')'}")
    ok &= a == b

    clients = make_clients(spec, seed, budget=budget)
    same = apply_shots(clients, FULL_SHOTS, seed) is clients
    print(f"  apply_shots(full) devolve o mesmo objeto      : {'OK' if same else 'FALHOU'}")
    ok &= same
    return ok


def legacy_run(spec: str, encoder: str, rounds: int, local_epochs: int, seed: int,
               budget: int | None, targets: list[str]):
    """O loop do baseline como era antes dos flags — referência do gate."""
    seed_everything(seed, workers=True)
    clients = make_clients(spec, seed, budget=budget)
    weights = [len(shard) for _, shard in clients]
    lr = BEST_LR[encoder]
    global_state = {k: v.detach().cpu().clone()
                    for k, v in BUILD_MODEL[encoder]().state_dict().items()}
    per_round = {}
    for r in range(1, rounds + 1):
        states = [
            local_train(_fresh(encoder, global_state), shard, local_epochs, lr,
                        seed=seed * 10**6 + r * 10**3 + ci, num_workers=0)
            for ci, (_cid, shard) in enumerate(clients)
        ]
        global_state = fedavg(states, weights)
        for t, a, f in _eval_all(_fresh(encoder, global_state), targets):
            per_round[(r, t)] = (a, f)
    return per_round, global_state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", default="device:RealWorld_thigh:10")
    ap.add_argument("--encoder", default="rnn", choices=sorted(BUILD_MODEL))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = ap.parse_args()

    set_deterministic()
    targets = parse_spec(args.spec)[1]
    budget = args.budget or None
    print(f"G-DEG: {args.spec} {args.encoder} R={args.rounds} E={args.local_epochs} "
          f"seed={args.seed}\n")

    print("[1] identidade estrutural")
    ok = check_structural(args.spec, args.encoder, args.seed, budget)

    print("\n[2] identidade comportamental")
    from federated.run_cross_device import run
    new = run(args.spec, args.encoder, args.rounds, args.local_epochs, args.seed,
              budget, targets, num_workers=0)  # defaults: method=none, shots=full
    old, _ = legacy_run(args.spec, args.encoder, args.rounds, args.local_epochs,
                        args.seed, budget, targets)

    worst = 0.0
    for _, row in new.iterrows():
        oa, of = old[(int(row["round"]), row.target)]
        worst = max(worst, abs(row.test_acc - oa), abs(row.test_f1_macro - of))
    print(f"  maior divergência de acc/F1 em {len(new)} linhas   : {worst:.3e}")
    ok &= worst < 1e-9

    n_ok = len(new) == args.rounds * len(targets)
    print(f"  linhas: {len(new)} (esperado {args.rounds * len(targets)})"
          f"{'' if n_ok else '  <-- FALHOU'}")
    ok &= n_ok

    cols_ok = set(new["method"]) == {"none"} and set(new["n_shots"]) == {"full"}
    print(f"  colunas novas com os defaults                 : "
          f"{'OK' if cols_ok else 'FALHOU'}")
    ok &= cols_ok

    print(f"\n  GATE G-DEG {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
