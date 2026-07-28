"""Baseline supervisionado FedAvg no eixo CROSS-DEVICE — sem Flower.

O loop local do `client.py` sempre foi torch puro (Adam + CrossEntropy); o que o
Flower acrescentava era a casca `NumPyClient` e a orquestração Ray. No
cross-device isso não paga: os clientes rodam sequencialmente na mesma GPU, e
tirar o Ray elimina de quebra o gotcha de `PYTHONPATH` que já mordeu o projeto.

Este runner e o `scripts/ssl/pretrain_fed.py` consomem a **mesma**
`cross_device.make_clients`, para que o Δ(Fed-SSL − baseline) não carregue
diferença de partição junto com o efeito do método.

Saída: `results/fed_cross_device.csv`
    encoder, spec, budget, seed, local_epochs, round, target,
    test_acc, test_f1_macro, uplink_mb, downlink_mb

Uso:
    # gate: 1 cliente com as mesmas janelas, R=1, deve reproduzir o centralizado
    python scripts/federated/run_cross_device.py --gate \
        --spec single:RealWorld_thigh:10 --encoder resnetse5 --local-epochs 5 --seed 0

    # um braço
    python scripts/federated/run_cross_device.py \
        --spec device:RealWorld_thigh:10 --encoder resnetse5 --rounds 100 --local-epochs 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Precisa estar definido ANTES de o CUDA inicializar, senão
# `use_deterministic_algorithms` falha nas GEMMs do cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl"))

import pandas as pd
import torch
from lightning.pytorch import seed_everything
from torch.utils.data import ConcatDataset, DataLoader

from common import BATCH_SIZE, BEST_LR, PROJECT_ROOT, SEEDS
from eval_transfer import BUILD_MODEL, DEVICE, evaluate, test_loader
from federated.cross_device import DEFAULT_BUDGET, make_clients, parse_spec
from pretrain_fed import fedavg, state_bytes

CACHE = PROJECT_ROOT / "results" / "fed_cross_device.csv"
COLS = ["encoder", "spec", "budget", "seed", "local_epochs", "round", "target",
        "test_acc", "test_f1_macro", "uplink_mb", "downlink_mb"]


def set_deterministic() -> None:
    """Torna o treino bit-reprodutível na GPU.

    Sem isto, dois treinos idênticos divergem ~1e-2 nos pesos depois de algumas
    épocas (algoritmos não-determinísticos do cuDNN no backward das convoluções)
    — medido em 2026-07-28, e foi o que reprovou a primeira execução do gate.
    Fora do gate não é obrigatório: a grade tem 4 seeds e a variação entra na
    barra de erro.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def local_train(model: torch.nn.Module, dataset, epochs: int, lr: float,
                seed: int, num_workers: int = 0) -> dict:
    """Treino local de um cliente: o mesmo loop de `client.py`, sem a casca Flower.

    Devolve o `state_dict` em CPU. `seed` semeia o shuffle do loader — os dois
    caminhos do gate (federado com 1 cliente e centralizado) chamam esta mesma
    função com a mesma seed, que é o que torna a comparação bit-a-bit legítima.
    """
    seed_everything(seed, workers=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    assert len(loader) > 0, f"cliente com {len(dataset)} amostras gera 0 batches"
    model = model.to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _fresh(encoder: str, state: dict) -> torch.nn.Module:
    model = BUILD_MODEL[encoder]()
    model.load_state_dict(state)
    return model


def _eval_all(model: torch.nn.Module, targets: list[str]) -> list[tuple[str, float, float]]:
    model = model.to(DEVICE).eval()
    return [(t, *evaluate(model, test_loader(t))) for t in targets]


def run(spec: str, encoder: str, rounds: int, local_epochs: int, seed: int,
        budget: int | None, targets: list[str], num_workers: int = 0) -> pd.DataFrame:
    seed_everything(seed, workers=True)
    clients = make_clients(spec, seed, budget=budget)
    weights = [len(shard) for _, shard in clients]
    lr = BEST_LR[encoder]

    global_state = {k: v.detach().cpu().clone()
                    for k, v in BUILD_MODEL[encoder]().state_dict().items()}
    mb = state_bytes(global_state) / 2**20
    print(f"=== [FedAvg-SL {encoder}] {spec} seed{seed}: {len(clients)} clientes "
          f"({sum(weights)} janelas), R={rounds}×E={local_epochs}, "
          f"{mb:.2f} MB/cliente/rodada/sentido ===", flush=True)

    rows = []
    for r in range(1, rounds + 1):
        t0 = time.time()
        states = [
            local_train(_fresh(encoder, global_state), shard, local_epochs, lr,
                        seed=seed * 10**6 + r * 10**3 + ci, num_workers=num_workers)
            for ci, (_cid, shard) in enumerate(clients)
        ]
        global_state = fedavg(states, weights)
        scores = _eval_all(_fresh(encoder, global_state), targets)
        for tgt, acc, f1 in scores:
            rows.append({"encoder": encoder, "spec": spec,
                         "budget": budget if budget is not None else "natural",
                         "seed": seed, "local_epochs": local_epochs, "round": r,
                         "target": tgt, "test_acc": acc, "test_f1_macro": f1,
                         "uplink_mb": mb * len(clients),
                         "downlink_mb": mb * len(clients)})
        print(f"[r{r}/{rounds}] {time.time() - t0:5.1f}s | "
              + ", ".join(f"{t}={a:.4f}" for t, a, _ in scores), flush=True)
    return pd.DataFrame(rows, columns=COLS)


def gate(spec: str, encoder: str, local_epochs: int, seed: int,
         budget: int | None, targets: list[str]) -> bool:
    """G-EQ1: federado com 1 cliente e R=1 tem de reproduzir o centralizado.

    Valida de uma vez o caminho de dados (`make_clients`), o loop local e o
    `fedavg` — com 1 cliente a média é a identidade, então qualquer divergência
    aponta bug de estado/seed, não de agregação.
    """
    set_deterministic()
    clients = make_clients(spec, seed, budget=budget)
    if len(clients) != 1:
        raise ValueError(f"o gate exige 1 cliente; {spec!r} deu {len(clients)}")
    shard = clients[0][1]
    print(f"gate: {spec} -> 1 cliente, {len(shard)} janelas, E={local_epochs}, seed={seed}")

    seed_everything(seed, workers=True)
    init = {k: v.detach().cpu().clone()
            for k, v in BUILD_MODEL[encoder]().state_dict().items()}
    local_seed = seed * 10**6 + 1 * 10**3 + 0

    fed_state = fedavg([local_train(_fresh(encoder, init), shard, local_epochs,
                                    BEST_LR[encoder], seed=local_seed)], [len(shard)])
    cen_state = local_train(_fresh(encoder, init), shard, local_epochs,
                            BEST_LR[encoder], seed=local_seed)

    max_dev = max((fed_state[k].to(torch.float64) - cen_state[k].to(torch.float64))
                  .abs().max().item() for k in fed_state)
    fed = _eval_all(_fresh(encoder, fed_state), targets)
    cen = _eval_all(_fresh(encoder, cen_state), targets)
    print(f"\n  maior divergência de peso federado-vs-centralizado: {max_dev:.3e}")
    print(f"  {'alvo':<18}{'federado':>22}{'centralizado':>22}")
    ok = True
    for (t, fa, ff), (_t, ca, cf) in zip(fed, cen):
        print(f"  {t:<18}{f'{fa:.6f}/{ff:.6f}':>22}{f'{ca:.6f}/{cf:.6f}':>22}")
        ok &= abs(fa - ca) < 1e-9 and abs(ff - cf) < 1e-9
    print(f"\n  GATE {'PASS' if ok and max_dev < 1e-6 else 'FAIL'} "
          f"(acc/F1 idênticos e pesos equivalentes)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, help="<modo>:<datasets>:<contagens>")
    ap.add_argument("--encoder", required=True, choices=sorted(BUILD_MODEL))
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="Janelas por cliente (0 = natural, sem parear).")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="Alvos de avaliação. Default: os datasets do spec.")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="CSV de saída (parcial por-run). Default: cache global.")
    ap.add_argument("--gate", action="store_true",
                    help="Roda o G-EQ1 (1 cliente, R=1 ≡ centralizado) e sai.")
    args = ap.parse_args()

    _mode, names, _counts = parse_spec(args.spec)
    targets = args.targets or names
    budget = args.budget or None

    if args.gate:
        ok = gate(args.spec, args.encoder, args.local_epochs, args.seed[0],
                  budget, targets)
        sys.exit(0 if ok else 1)

    frames = [run(args.spec, args.encoder, args.rounds, args.local_epochs, s,
                  budget, targets, args.num_workers) for s in args.seed]
    new = pd.concat(frames, ignore_index=True)
    cache_path = Path(args.out) if args.out else CACHE
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=COLS)
    out = pd.concat([cache, new], ignore_index=True).drop_duplicates(
        subset=["encoder", "spec", "budget", "seed", "local_epochs", "round", "target"],
        keep="last")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    print(f"\n{len(new)} linhas novas -> {cache_path} ({len(out)} no total)")


if __name__ == "__main__":
    main()
