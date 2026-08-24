"""RQ1, eixo federado: FedAvg supervisionado, partição NATURAL, clientes máximos.

Cada job é um `run_cross_device.py` com `--budget 0` (D3) e `--shots k` (rótulos
por classe POR CLIENTE), R=150, E=5. Parciais em
`results/rqs/rq1_federado_parts/`, consolidado em `results/rqs/rq1_federado.csv`.

Uso:
    python scripts/rqs/run_rq1_federado.py --seed 0
    python scripts/rqs/run_rq1_federado.py --consolidate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

from rqs.config import (BUDGET, CKPT, E_LOCAL, ENCODERS, FEDERACOES, FULL_SHOTS,
                        K_LEVELS, LOGS, RESULTS, R_FT, SEEDS_RQ, parcial_pronto,
                        spec, verifica_clientes)
from rqs.lr_escolhida import decididas, lr_de  # noqa: E402
from gpu_pool import Job, detect_gpus, run_pool  # noqa: E402

RUNNER = Path(__file__).resolve().parents[1] / "federated" / "run_cross_device.py"
CACHE = RESULTS / "rq1_federado.csv"
PARTS = RESULTS / "rq1_federado_parts"
KEY = ["encoder", "spec", "budget", "seed", "local_epochs", "round", "target", "n_shots"]


def parcial(encoder, ds, k, seed) -> Path:
    return PARTS / f"{encoder}_{ds}_k{k}_seed{seed}.csv"




def consolidar() -> None:
    import pandas as pd
    frames = [pd.read_csv(p) for p in sorted(PARTS.glob("*.csv"))]
    if not frames:
        print("[RQ1-F] nenhum parcial.", flush=True)
        return
    full = pd.concat(frames, ignore_index=True)
    subset = [c for c in KEY if c in full.columns]
    full = full.drop_duplicates(subset=subset, keep="last")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(CACHE, index=False)
    print(f"[RQ1-F] {len(full)} linhas de {len(frames)} parciais -> {CACHE}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", nargs="+", choices=ENCODERS, default=ENCODERS)
    ap.add_argument("--dataset", nargs="+", choices=sorted(FEDERACOES), default=sorted(FEDERACOES))
    ap.add_argument("--k", nargs="+", default=[str(k) for k in K_LEVELS])
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS_RQ)
    ap.add_argument("--rounds", type=int, default=R_FT)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--so-lr-decidida", action="store_true",
                    help="Só enfileira células (encoder, federação) cuja LR a busca S1 já fechou.")
    args = ap.parse_args()

    if args.consolidate:
        consolidar()
        return
    verifica_clientes()
    PARTS.mkdir(parents=True, exist_ok=True)
    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()] if args.gpus else detect_gpus()) or [0]

    jobs, prontos, adiadas = [], 0, []
    fechadas = decididas()
    for e in args.encoder:
        for d in args.dataset:
            if args.so_lr_decidida and (e, d) not in fechadas:
                adiadas.append(f"{e}/{d}")
                continue
            for k in args.k:
                # LR POR REGIME (2026-08-24): k in {1,2,4} usa a do k=1, Full a
                # dele. Colapsar num valor so custava 1,08 pp no Full -- do
                # tamanho do efeito que a RQ1 mede. Ver lr_escolhida.py.
                lr = lr_de(e, d, k)
                for s in args.seed:
                    p = parcial(e, d, k, s)
                    if not args.force and parcial_pronto(p, args.rounds, lr):
                        prontos += 1
                        continue
                    label = f"{e}_{d}_k{k}_seed{s}"
                    jobs.append(Job(label, [
                        sys.executable, str(RUNNER),
                        "--spec", spec(d), "--encoder", e,
                        "--rounds", str(args.rounds), "--local-epochs", str(E_LOCAL),
                        "--seed", str(s), "--budget", str(BUDGET), "--shots", str(k),
                        "--lr", str(lr), "--out", str(p),
                        "--ckpt-dir", str(CKPT / "fed" / e / d / f"k{k}"),
                    ], LOGS / "rq1_federado" / f"{label}.log"))

    if adiadas:
        print(f"[RQ1-F] {len(adiadas)} células adiadas (LR ainda não decidida): {adiadas}", flush=True)
    print(f"[RQ1-F] {len(jobs)} jobs ({prontos} prontos) | GPUs={gpus}", flush=True)
    falhas = run_pool(jobs, gpus, args.max_parallel or len(gpus))
    if falhas:
        print(f"[RQ1-F] {len(falhas)} falharam: {falhas}", flush=True)
    consolidar()


if __name__ == "__main__":
    main()
