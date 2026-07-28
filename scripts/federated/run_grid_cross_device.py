"""Driver da grade cross-device (baseline supervisionado FedAvg) em N GPUs.

Grade decidida na sabatina de 2026-07-27/28 (`docs/plano_fedssl.md §2.2/§2.3`):
7 configs × `k ∈ {1, 5}` × seeds × encoders, R=100. Cada job é um subprocesso
`run_cross_device.py` fixado numa GPU (ver `gpu_pool`), gravando um **parcial
próprio** em `results/fed_cross_device_parts/`. Resume: pula o job cujo parcial
já tem as `R × |alvos|` linhas. Ao final consolida em
`results/fed_cross_device.csv`.

Uso:
    python scripts/federated/run_grid_cross_device.py                 # grade completa
    python scripts/federated/run_grid_cross_device.py --encoder resnetse5 --k 1
    python scripts/federated/run_grid_cross_device.py --consolidate   # só junta os parciais
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

import pandas as pd

from common import PROJECT_ROOT, SEEDS
from eval_transfer import BUILD_MODEL
from federated.cross_device import parse_spec
from gpu_pool import Job, detect_gpus, run_pool

RUNNER = Path(__file__).resolve().parent / "run_cross_device.py"
LOG_DIR = PROJECT_ROOT / "logs" / "fed_cross_device"
CACHE = PROJECT_ROOT / "results" / "fed_cross_device.csv"
PART_DIR = PROJECT_ROOT / "results" / "fed_cross_device_parts"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "fed_cross_device"
KEY = ["encoder", "spec", "budget", "seed", "local_epochs", "round", "target"]

# Os 6 braços da grade (o 7º, `single:*`, é o gate — roda à parte, com --gate).
SPECS = [
    "device:RealWorld_thigh:10",                 # in10-RW
    "device:MotionSense:10",                     # in10-MS
    "device:RealWorld_thigh+MotionSense:10+10",  # cross10+10
    "device:RealWorld_thigh+MotionSense:5+5",    # cross5+5
    "iid:RealWorld_thigh:10",                    # controle de feature skew
    "iid:MotionSense:10",                        # idem
]
# k = épocas efetivas de backbone por rodada. No baseline supervisionado não há
# alternância, então local_epochs = k (no LFR seria 6k — ver plano §2.3).
K_LEVELS = [1, 5]


def partial_path(encoder: str, spec: str, k: int, seed: int) -> Path:
    return PART_DIR / f"{encoder}_{spec.replace(':', '-')}_k{k}_seed{seed}.csv"


def done(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    return len(pd.read_csv(path).dropna(subset=["test_f1_macro"])) >= expected_rows


def consolidate() -> None:
    frames = [pd.read_csv(p) for p in sorted(PART_DIR.glob("*.csv"))]
    if not frames:
        print("[CONCAT] nenhum parcial encontrado.", flush=True)
        return
    full = pd.concat(frames, ignore_index=True).drop_duplicates(subset=KEY, keep="last")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(CACHE, index=False)
    print(f"[CONCAT] {len(full)} linhas de {len(frames)} parciais -> {CACHE}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", nargs="+", choices=sorted(BUILD_MODEL),
                    default=["resnetse5"],
                    help="Padrão: só resnetse5 (abrir para os 4 depois do gate).")
    ap.add_argument("--spec", nargs="+", default=SPECS)
    ap.add_argument("--k", nargs="+", type=int, default=K_LEVELS)
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--budget", type=int, default=192)
    ap.add_argument("--gpus", type=str, default=None, help="Ex.: '0,1,2,3'. Padrão: auto.")
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--consolidate", action="store_true", help="Só junta os parciais e sai.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.consolidate:
        consolidate()
        return

    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()]
            if args.gpus else detect_gpus()) or [0]
    PART_DIR.mkdir(parents=True, exist_ok=True)

    jobs, skipped = [], 0
    for encoder in args.encoder:
        for spec in args.spec:
            n_targets = len(parse_spec(spec)[1])
            for k in args.k:
                for seed in args.seed:
                    part = partial_path(encoder, spec, k, seed)
                    if not args.force and done(part, args.rounds * n_targets):
                        skipped += 1
                        continue
                    label = f"{encoder}_{spec.replace(':', '-')}_k{k}_seed{seed}"
                    ckpt = CKPT_DIR / encoder / spec.replace(":", "-") / f"k{k}"
                    jobs.append(Job(label, [
                        sys.executable, str(RUNNER),
                        "--spec", spec, "--encoder", encoder,
                        "--rounds", str(args.rounds), "--local-epochs", str(k),
                        "--seed", str(seed), "--budget", str(args.budget),
                        "--out", str(part), "--ckpt-dir", str(ckpt),
                    ], LOG_DIR / f"{label}.log"))

    print(f"[GRID] {len(jobs)} jobs a rodar ({skipped} já completos) em {len(gpus)} GPUs.",
          flush=True)
    failed = run_pool(jobs, gpus, args.max_parallel or len(gpus))
    if failed:
        print(f"[GRID] {len(failed)} jobs falharam: {failed}", flush=True)
    consolidate()


if __name__ == "__main__":
    main()
