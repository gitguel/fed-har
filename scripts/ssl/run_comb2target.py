"""Grade do experimento "comb→target" em N GPUs (só downstream, sem pré-treino).

Isola o efeito do corpus de pré-treino na hipótese da inversão
Especialista/Combinado: para cada (encoder, dataset, seed), usa o backbone
pré-treinado no `combined` (já existente da grade v1) e refina **só no train
rotulado do dataset-alvo** — o mesmo protocolo do cenário especialista, mudando
apenas a inicialização. Comparações que o resultado habilita:

    Δ(comb→t − esp)  = efeito do corpus de pré-treino (mesmos dados de finetune)
    Δ(comb − comb→t) = efeito dos dados rotulados combinados no finetune

Grade: 3 encoders × 6 datasets × 4 seeds = 72 jobs, cada um cobrindo
2 protocolos × 4 regimes × 6 alvos = 48 linhas (3456 no total). Parciais em
`results/ssl_lfr_comb2target_parts/`, consolidado em
`results/ssl_lfr_comb2target_eval_transfer.csv` (mesmas colunas do cache SSL;
`source` = dataset do finetune; o backbone é sempre do `combined`).

Uso:
    python scripts/ssl/run_comb2target.py --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))      # .../scripts/ssl

import pandas as pd

from common import DATASETS, PROJECT_ROOT, SEEDS  # noqa: E402
from gpu_pool import Job, detect_gpus, run_pool  # noqa: E402
from encoders import ENCODERS  # noqa: E402
from pretrain_lfr import ssl_ckpt_dir  # noqa: E402
from downstream_eval import COLS, KEY  # noqa: E402

DOWNSTREAM = Path(__file__).resolve().parent / "downstream_eval.py"
CACHE = PROJECT_ROOT / "results" / "ssl_lfr_comb2target_eval_transfer.csv"
PART_DIR = PROJECT_ROOT / "results" / "ssl_lfr_comb2target_parts"
LOG_DIR = PROJECT_ROOT / "logs" / "ssl_runs"
ROWS_PER_PARTIAL = 2 * 4 * 6  # protocolos × regimes × alvos = 48


def partial_path(encoder: str, source: str, seed: int) -> Path:
    return PART_DIR / f"{encoder}_{source}_{seed}.csv"


def partial_done(encoder: str, source: str, seed: int) -> bool:
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
    ap.add_argument("--source", nargs="+", choices=DATASETS, default=DATASETS,
                    help="Dataset(s) do finetune (o backbone é sempre do combined).")
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--gpus", type=str, default=None, help="Ex.: '0,1,2,3'. Padrão: auto.")
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()] if args.gpus else detect_gpus()) or [0]
    max_parallel = args.max_parallel or len(gpus)

    PART_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    combos = [(e, s, seed) for e in args.encoder for s in args.source for seed in args.seed]
    for e, s, seed in combos:
        if not (ssl_ckpt_dir(e, "combined", seed) / "backbone.ckpt").exists():
            print(f"[GRID] SKIP (sem backbone combined) {e}/seed{seed}", flush=True)
            continue
        if not args.force and partial_done(e, s, seed):
            continue
        argv = [sys.executable, str(DOWNSTREAM), "--encoder", e, "--source", s,
                "--seed", str(seed), "--protocol", "both", "--shots", "all",
                "--pretrain-source", "combined",
                "--epochs", str(args.epochs), "--num-workers", str(args.num_workers),
                "--out", str(partial_path(e, s, seed))]
        if args.force:
            argv.append("--force")
        jobs.append(Job(f"c2t_{e}_{s}_seed{seed}", argv,
                        LOG_DIR / f"c2t_{e}_{s}_{seed}.log"))
    print(f"[GRID] comb→target: {len(jobs)} jobs a rodar de {len(combos)} combos.", flush=True)
    run_pool(jobs, gpus, max_parallel)
    consolidate()


if __name__ == "__main__":
    main()
