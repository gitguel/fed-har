"""Driver da grade supervisionada nos regimes few-shot, em N GPUs.

Paraleliza os treinos por combinação (encoder × fonte × seed × n_shots), um
subprocesso por GPU (ver `gpu_pool`). Cada treino grava seus próprios checkpoints
(`checkpoints/supervised/<enc>/<source>/seed<N>/shots<K>/`), então não há corrida
de escrita. Ao final, roda `eval_transfer.py` uma vez (serial, só inferência) para
consolidar `results/supervised_eval_transfer.csv` com a coluna `n_shots`.

Por padrão roda apenas os regimes **novos** (1/10/100); o regime `full` já foi
treinado antes (84 checkpoints). Resume: pula combos cujo `best.ckpt` já existe.

Uso:
    python scripts/supervised/run_all_shots.py                 # 3×7×4×3 = 252 treinos, 8 GPUs
    python scripts/supervised/run_all_shots.py --shots 1 10 100 full   # inclui full (re-treina)
    python scripts/supervised/run_all_shots.py --encoder resnetse5 --no-eval
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

from common import (  # noqa: E402
    COMBINED_DATASET_NAME,
    DATASETS,
    PROJECT_ROOT,
    SEEDS,
    normalize_shots,
    supervised_ckpt_dir,
)
from gpu_pool import Job, detect_gpus, run_pool  # noqa: E402

SUP_DIR = Path(__file__).resolve().parent
ENCODERS = ["resnetse5", "cnnpff", "rnn", "tstcc"]
SOURCES = [COMBINED_DATASET_NAME] + DATASETS
LOG_DIR = PROJECT_ROOT / "logs" / "sup_shots_runs"
EVAL_SCRIPT = SUP_DIR.parent / "eval_transfer.py"


def train_argv(encoder: str, source: str, seed: int, n_shots, max_epochs: int):
    """Comando de treino de uma combinação (combined usa train_combined.py)."""
    shots = str(n_shots)
    if source == COMBINED_DATASET_NAME:
        script = SUP_DIR / "train_combined.py"
        return [sys.executable, str(script), "--encoder", encoder, "--seed", str(seed),
                "--shots", shots, "--max-epochs", str(max_epochs)]
    script = SUP_DIR / f"train_{encoder}.py"
    return [sys.executable, str(script), "--dataset", source, "--seed", str(seed),
            "--shots", shots, "--max-epochs", str(max_epochs)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", nargs="+", choices=ENCODERS, default=ENCODERS)
    ap.add_argument("--source", nargs="+", choices=SOURCES, default=SOURCES)
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--shots", nargs="+", default=["1", "10", "100"],
                    help="Regimes a treinar. Padrão: 1 10 100 (full já existe).")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--gpus", type=str, default=None, help="Ex.: '0,1,2,3'. Padrão: auto.")
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Re-treina mesmo com best.ckpt presente.")
    ap.add_argument("--no-eval", action="store_true", help="Não roda eval_transfer.py ao final.")
    args = ap.parse_args()

    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()] if args.gpus else detect_gpus()) or [0]
    max_parallel = args.max_parallel or len(gpus)
    shot_regimes = normalize_shots(args.shots)

    jobs = []
    total = 0
    for e in args.encoder:
        for s in args.source:
            for seed in args.seed:
                for n_shots in shot_regimes:
                    total += 1
                    if not args.force and (supervised_ckpt_dir(e, s, seed, n_shots) / "best.ckpt").exists():
                        continue
                    label = f"{e}_{s}_seed{seed}_shots{n_shots}"
                    jobs.append(Job(label, train_argv(e, s, seed, n_shots, args.max_epochs),
                                    LOG_DIR / f"{label}.log"))

    print(f"[GRID] treinos: {len(jobs)} a rodar de {total} combos "
          f"(encoders={args.encoder} shots={shot_regimes}).", flush=True)
    failures = run_pool(jobs, gpus, max_parallel)

    if args.no_eval:
        print("[GRID] --no-eval: pulando eval_transfer.py.", flush=True)
        return
    print("[GRID] treinos concluídos; rodando eval_transfer.py (serial)...", flush=True)
    subprocess.run([sys.executable, str(EVAL_SCRIPT)], check=False)
    if failures:
        print(f"[GRID] atenção: {len(failures)} treino(s) falharam: {failures}", flush=True)


if __name__ == "__main__":
    main()
