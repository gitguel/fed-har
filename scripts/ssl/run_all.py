"""Driver da grade SSL (LFR) em N GPUs — pré-treino + avaliação downstream.

Duas fases, ambas paralelizadas com um subprocesso por GPU (ver `gpu_pool`):

  Fase 1 — pré-treino: 84 jobs `pretrain_lfr.py` (3 enc × 7 fontes × 4 seeds).
           Resume: pula se `backbone.ckpt` já existe.
  Fase 2 — downstream: 84 jobs `downstream_eval.py --protocol both --shots all`,
           cada um gravando um **parcial próprio** em
           `results/ssl_lfr_parts/<enc>_<source>_<seed>.csv` (48 linhas = 2
           protocolos × 4 regimes × 6 alvos). Resume: pula se o parcial já tem 48
           linhas. Ao final consolida os parciais em `results/ssl_lfr_eval_transfer.csv`.

Uso:
    python scripts/ssl/run_all.py                     # grade completa (8 GPUs auto)
    python scripts/ssl/run_all.py --max-parallel 4
    python scripts/ssl/run_all.py --encoder resnetse5 --source combined
    python scripts/ssl/run_all.py --phase pretrain    # só a fase 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))      # .../scripts/ssl

import pandas as pd

from common import PROJECT_ROOT, SEEDS  # noqa: E402
from gpu_pool import Job, detect_gpus, run_pool  # noqa: E402
from encoders import ENCODERS  # noqa: E402
from pretrain_lfr import SOURCES, ssl_ckpt_dir  # noqa: E402
from downstream_eval import COLS, KEY  # noqa: E402

PRETRAIN_SCRIPT = {
    "lfr": Path(__file__).resolve().parent / "pretrain_lfr.py",
    "tfc": Path(__file__).resolve().parent / "pretrain_tfc.py",
}
DOWNSTREAM = Path(__file__).resolve().parent / "downstream_eval.py"
LOG_DIR = PROJECT_ROOT / "logs" / "ssl_runs"
ROWS_PER_PARTIAL = 2 * 4 * 6  # protocolos × regimes × alvos = 48

# Definidos em main() a partir de --method:
METHOD = "lfr"
CACHE = PROJECT_ROOT / "results" / "ssl_lfr_eval_transfer.csv"
PART_DIR = PROJECT_ROOT / "results" / "ssl_lfr_parts"


def partial_path(encoder: str, source: str, seed: int) -> Path:
    return PART_DIR / f"{encoder}_{source}_{seed}.csv"


def downstream_done(encoder: str, source: str, seed: int) -> bool:
    p = partial_path(encoder, source, seed)
    if not p.exists():
        return False
    try:
        df = pd.read_csv(p)
    except Exception:
        return False
    return len(df.dropna(subset=["test_f1_macro"])) >= ROWS_PER_PARTIAL


def consolidate() -> None:
    frames = []
    if CACHE.exists():
        frames.append(pd.read_csv(CACHE))
    for p in sorted(PART_DIR.glob("*.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:
            print(f"[CONCAT] aviso: falha lendo {p}: {exc}", flush=True)
    if not frames:
        print("[CONCAT] nada para consolidar.", flush=True)
        return
    full = pd.concat(frames, ignore_index=True)
    full["n_shots"] = full["n_shots"].astype(str)
    for c in COLS:
        if c not in full.columns:
            full[c] = pd.NA
    full = (full[COLS].drop_duplicates(subset=KEY, keep="last")
            .sort_values(KEY).reset_index(drop=True))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(CACHE, index=False)
    print(f"[CONCAT] {len(full)} linhas -> {CACHE}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", nargs="+", choices=ENCODERS, default=ENCODERS)
    ap.add_argument("--source", nargs="+", choices=SOURCES, default=SOURCES)
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--method", choices=["lfr", "tfc"], default="lfr",
                    help="Técnica SSL da grade (padrão: lfr).")
    ap.add_argument("--phase", choices=["pretrain", "downstream", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Máx. de épocas do treino downstream (ES paciência 50).")
    ap.add_argument("--gpus", type=str, default=None, help="Ex.: '0,1,2,3'. Padrão: auto.")
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()] if args.gpus else detect_gpus()) or [0]
    max_parallel = args.max_parallel or len(gpus)

    global METHOD, CACHE, PART_DIR
    METHOD = args.method
    CACHE = PROJECT_ROOT / "results" / f"ssl_{METHOD}_eval_transfer.csv"
    PART_DIR = PROJECT_ROOT / "results" / f"ssl_{METHOD}_parts"
    pretrain_script = PRETRAIN_SCRIPT[METHOD]

    combos = [(e, s, seed) for e in args.encoder for s in args.source for seed in args.seed]

    # ── Fase 1: pré-treino ──────────────────────────────────────────────────
    if args.phase in ("pretrain", "both"):
        jobs = []
        for e, s, seed in combos:
            if not args.force and (ssl_ckpt_dir(e, s, seed, METHOD) / "backbone.ckpt").exists():
                continue
            argv = [sys.executable, str(pretrain_script), "--encoder", e, "--source", s,
                    "--seed", str(seed), "--num-workers", str(args.num_workers)]
            if args.force:
                argv.append("--force")
            jobs.append(Job(f"pretrain_{METHOD}_{e}_{s}_seed{seed}", argv,
                            LOG_DIR / f"pretrain_{METHOD}_{e}_{s}_{seed}.log"))
        print(f"[GRID] Fase 1 (pretrain): {len(jobs)} jobs a rodar de {len(combos)} combos.", flush=True)
        run_pool(jobs, gpus, max_parallel)

    # ── Fase 2: downstream ──────────────────────────────────────────────────
    if args.phase in ("downstream", "both"):
        PART_DIR.mkdir(parents=True, exist_ok=True)
        jobs = []
        for e, s, seed in combos:
            if not (ssl_ckpt_dir(e, s, seed, METHOD) / "backbone.ckpt").exists():
                print(f"[GRID] SKIP downstream (sem backbone) {e}/{s}/seed{seed}", flush=True)
                continue
            if not args.force and downstream_done(e, s, seed):
                continue
            out = partial_path(e, s, seed)
            argv = [sys.executable, str(DOWNSTREAM), "--encoder", e, "--source", s,
                    "--seed", str(seed), "--method", METHOD,
                    "--protocol", "both", "--shots", "all",
                    "--epochs", str(args.epochs), "--num-workers", str(args.num_workers),
                    "--out", str(out)]
            if args.force:
                argv.append("--force")
            jobs.append(Job(f"downstream_{METHOD}_{e}_{s}_seed{seed}", argv,
                            LOG_DIR / f"downstream_{METHOD}_{e}_{s}_{seed}.log"))
        print(f"[GRID] Fase 2 (downstream): {len(jobs)} jobs a rodar de {len(combos)} combos.", flush=True)
        run_pool(jobs, gpus, max_parallel)
        consolidate()


if __name__ == "__main__":
    main()
