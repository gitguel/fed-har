"""G-EQ1-SSL — com 1 cliente, o FedAvg do pré-treino tem de ser a identidade.

Espelho exato do G-EQ1 do baseline (`federated/run_cross_device.py:gate`), aplicado
ao caminho de pré-treino. Com um único cliente a agregação é a média de um termo,
então qualquer divergência entre "rodar `local_pretrain` direto" e "rodar uma
rodada de FedAvg" aponta bug de estado ou de semente, não de agregação.

Por que este gate existe: o ramo `":" in partition` de `pretrain_fed.py:236` — o
que conecta o pré-treino SSL à partição cross-device compartilhada com o baseline —
foi escrito em 2026-07-17 e **nunca tinha sido executado** até 2026-07-29. Os
únicos checkpoints em `checkpoints/ssl_fed/` eram do eixo `silo`, já aposentado.

Testa também o **G-LOAD**: o `backbone.ckpt` produzido tem de carregar no `Probe`
do downstream com `strict=True` (o `TFC_Backbone` tem layout próprio) e tem de
diferir da inicialização aleatória — um cliente que gere 0 batches devolveria os
pesos globais intactos e o pré-treino "terminaria" sem erro (família F2).

Uso:
    python scripts/ssl/gate_pretrain_fed.py --method tfc --encoder rnn
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))      # .../scripts/ssl

import torch
from lightning.pytorch import seed_everything

from downstream_eval import BUILD_SSL_BACKBONE, Probe
from encoders import ENCODERS
from federated.cross_device import DEFAULT_BUDGET, make_clients
from pretrain_fed import (METHODS, build_global_model, extract_backbone_state,
                          fedavg, local_pretrain, run_fedssl)


def gate_identity(method: str, encoder: str, spec: str, local_epochs: int,
                  seed: int, budget: int | None) -> bool:
    """FedAvg de 1 cliente ≡ treino local direto."""
    clients = make_clients(spec, seed, budget=budget)
    if len(clients) != 1:
        raise ValueError(f"o gate exige 1 cliente; {spec!r} deu {len(clients)}")
    shard = clients[0][1]
    print(f"  {spec} -> 1 cliente, {len(shard)} janelas, E={local_epochs}")

    seed_everything(seed, workers=True)
    template = build_global_model(method, encoder, shard)
    init = {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
    local_seed = seed * 10**6 + 1 * 10**3 + 0

    a = local_pretrain(method, template, init, shard, local_epochs, local_seed, 4)
    b = local_pretrain(method, copy.deepcopy(template), init, shard, local_epochs,
                       local_seed, 4)
    fed = fedavg([a], [len(shard)])

    dev_id = max((fed[k].to(torch.float64) - a[k].to(torch.float64)).abs().max().item()
                 for k in fed)
    dev_rep = max((b[k].to(torch.float64) - a[k].to(torch.float64)).abs().max().item()
                  for k in b)
    print(f"  fedavg([x]) vs x                 : {dev_id:.3e}")
    print(f"  duas execuções da mesma semente   : {dev_rep:.3e}")
    return dev_id < 1e-9 and dev_rep < 1e-6


def gate_load(method: str, encoder: str, spec: str, local_epochs: int,
              seed: int) -> bool:
    """O backbone.ckpt gerado carrega no Probe com strict=True e != init."""
    with tempfile.TemporaryDirectory() as tmp:
        import pretrain_fed
        original = pretrain_fed.FED_CKPT_ROOT
        pretrain_fed.FED_CKPT_ROOT = Path(tmp)
        try:
            ckpt = run_fedssl(method, encoder, spec, "", rounds=1,
                              local_epochs=local_epochs, seed=seed, num_workers=4,
                              force=True)
        finally:
            pretrain_fed.FED_CKPT_ROOT = original

        backbone, enc_dim = BUILD_SSL_BACKBONE[method](encoder)
        before = {k: v.clone() for k, v in backbone.state_dict().items()}
        backbone.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        after = backbone.state_dict()
        changed = any(not torch.equal(before[k], after[k]) for k in before)
        probe = Probe(backbone, enc_dim)
        out = probe(torch.randn(4, 6, 60))
        shape_ok = tuple(out.shape) == (4, 6)
        print(f"  load_state_dict(strict=True)     : OK")
        print(f"  pesos != inicialização aleatória : {'OK' if changed else 'FALHOU'}")
        print(f"  Probe(x).shape == (4, 6)         : {'OK' if shape_ok else out.shape}")
        return changed and shape_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=METHODS, default="tfc")
    ap.add_argument("--encoder", choices=ENCODERS, default="rnn")
    ap.add_argument("--spec", default="single:RealWorld_thigh:10")
    ap.add_argument("--load-spec", default="device:RealWorld_thigh:10")
    ap.add_argument("--local-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = ap.parse_args()

    le = args.local_epochs or (6 if args.method == "lfr" else 1)
    print(f"G-EQ1-SSL / G-LOAD: {args.method}/{args.encoder} seed={args.seed}\n")
    print("[1] identidade do FedAvg com 1 cliente")
    ok = gate_identity(args.method, args.encoder, args.spec, le, args.seed,
                       args.budget or None)
    print("\n[2] G-LOAD: pré-treino -> Probe do downstream")
    ok &= gate_load(args.method, args.encoder, args.load_spec, le, args.seed)

    print(f"\n  GATE {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
