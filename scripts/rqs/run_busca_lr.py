"""Busca S1 do eixo federado: só a LR do cliente, `E=5` fixo (D6).

4 LRs × 4 encoders × 5 federações × regimes {k=1, Full} × 1 seed = 160 runs,
~11 GPU-h. `E=5` NÃO é buscado — é premissa declarada, com as quatro citações
do `plano_fedssl §2.3`.

A escolha é por federação e encoder, pela acurácia de VALIDAÇÃO (protocolo D4).

Uso:
    python scripts/rqs/run_busca_lr.py --seed 0
    python scripts/rqs/run_busca_lr.py --resumo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

from rqs.config import (BUDGET, E_LOCAL, ENCODERS, FEDERACOES, FULL_SHOTS, LOGS,
                        LR_GRID, RESULTS, R_FT, spec, verifica_clientes)
from gpu_pool import Job, detect_gpus, run_pool  # noqa: E402

RUNNER = Path(__file__).resolve().parents[1] / "federated" / "run_cross_device.py"
PARTS = RESULTS / "busca_lr_parts"
CACHE = RESULTS / "busca_lr.csv"
REGIMES = ["1", FULL_SHOTS]


def parcial(e, d, k, lr, seed) -> Path:
    return PARTS / f"{e}_{d}_k{k}_lr{lr:.0e}_seed{seed}.csv"


def resumo() -> None:
    import pandas as pd
    frames = []
    for p in sorted(PARTS.glob("*.csv")):
        df = pd.read_csv(p)
        df["lr"] = float(p.stem.split("_lr")[1].split("_")[0])
        frames.append(df)
    if not frames:
        print("[BUSCA] nenhum parcial.", flush=True)
        return
    full = pd.concat(frames, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(CACHE, index=False)
    melhor = full.loc[full.groupby(["encoder", "target", "n_shots", "lr"]).val_acc.idxmax()]
    tab = (melhor.groupby(["target", "encoder", "n_shots"])
                 .apply(lambda g: g.loc[g.val_acc.idxmax(), "lr"], include_groups=False)
                 .unstack())
    print(f"[BUSCA] {len(full)} linhas -> {CACHE}\n\nLR vencedora (val_acc):")
    print(tab.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", nargs="+", choices=ENCODERS, default=ENCODERS)
    ap.add_argument("--dataset", nargs="+", choices=sorted(FEDERACOES), default=sorted(FEDERACOES))
    ap.add_argument("--lr", nargs="+", type=float, default=LR_GRID)
    ap.add_argument("--k", nargs="+", default=REGIMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=R_FT)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resumo", action="store_true")
    args = ap.parse_args()

    if args.resumo:
        resumo()
        return
    verifica_clientes()
    PARTS.mkdir(parents=True, exist_ok=True)
    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()] if args.gpus else detect_gpus()) or [0]

    jobs = []
    for e in args.encoder:
        for d in args.dataset:
            for k in args.k:
                for lr in args.lr:
                    p = parcial(e, d, k, lr, args.seed)
                    if not args.force and p.exists():
                        continue
                    label = f"{e}_{d}_k{k}_lr{lr:.0e}"
                    jobs.append(Job(label, [
                        sys.executable, str(RUNNER),
                        "--spec", spec(d), "--encoder", e,
                        "--rounds", str(args.rounds), "--local-epochs", str(E_LOCAL),
                        "--seed", str(args.seed), "--budget", str(BUDGET),
                        "--shots", str(k), "--lr", str(lr), "--out", str(p),
                    ], LOGS / "busca_lr" / f"{label}.log"))

    print(f"[BUSCA] {len(jobs)} jobs | GPUs={gpus}", flush=True)
    falhas = run_pool(jobs, gpus, args.max_parallel or len(gpus))
    if falhas:
        print(f"[BUSCA] {len(falhas)} falharam: {falhas}", flush=True)
    resumo()


if __name__ == "__main__":
    main()
