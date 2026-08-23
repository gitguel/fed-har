"""RQ2: pré-treino SSL federado + fine-tuning federado, na mesma grade da RQ1.

Duas fases, sequenciais (é o protocolo dos quatro papers — `plano_fedssl §2.3`):

  A. pré-treino  `pretrain_fed.py`, partição natural, R=100, batch 64 (D5).
                 TF-C `--local-epochs 5`; LFR `--local-epochs 30` (6k, alternância 1:5).
  B. fine-tuning `run_cross_device.py --method M --init-ckpt <backbone>`, R=150.

O braço SEM pré-treino é a própria RQ1 (`run_rq1_federado.py`) — não se repete aqui.

Uso:
    python scripts/rqs/run_rq2.py --fase pretrain --seed 0
    python scripts/rqs/run_rq2.py --fase finetune --seed 0
    python scripts/rqs/run_rq2.py --consolidate
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

from rqs.config import (BUDGET, CKPT, E_LOCAL, ENCODERS, FEDERACOES, K_LEVELS,
                        LOCAL_EPOCHS_PRE, LOGS, METHODS, RESULTS, R_FT, R_PRE,
                        SEEDS_RQ, spec, verifica_clientes)
from gpu_pool import Job, detect_gpus, run_pool  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1]
PRE_RUNNER = SCRIPTS / "ssl" / "pretrain_fed.py"
FT_RUNNER = SCRIPTS / "federated" / "run_cross_device.py"
CACHE = RESULTS / "rq2_finetuning.csv"
PARTS = RESULTS / "rq2_finetuning_parts"
KEY = ["encoder", "spec", "budget", "seed", "local_epochs", "round", "target",
       "method", "n_shots", "pretrain_rounds"]


def backbone(method: str, encoder: str, ds: str, seed: int) -> Path:
    """Espelha `pretrain_fed.fed_ckpt_dir` (combo vazio colapsa no Path)."""
    root = Path(os.environ["FEDHAR_FEDSSL_CKPT_ROOT"])
    return root / method / encoder / spec(ds).replace(":", "-") / f"seed{seed}" / "backbone.ckpt"


def parcial(method, encoder, ds, k, seed) -> Path:
    return PARTS / f"{method}_{encoder}_{ds}_k{k}_seed{seed}.csv"


def pronto(path: Path, linhas: int) -> bool:
    import pandas as pd
    return path.exists() and len(pd.read_csv(path).dropna(subset=["test_f1_macro"])) >= linhas


def consolidar() -> None:
    import pandas as pd
    frames = [pd.read_csv(p) for p in sorted(PARTS.glob("*.csv"))]
    if not frames:
        print("[RQ2] nenhum parcial.", flush=True)
        return
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=[c for c in KEY if c in full.columns], keep="last")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(CACHE, index=False)
    print(f"[RQ2] {len(full)} linhas de {len(frames)} parciais -> {CACHE}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fase", choices=["pretrain", "finetune"], default="pretrain")
    ap.add_argument("--method", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--encoder", nargs="+", choices=ENCODERS, default=ENCODERS)
    ap.add_argument("--dataset", nargs="+", choices=sorted(FEDERACOES), default=sorted(FEDERACOES))
    ap.add_argument("--k", nargs="+", default=[str(k) for k in K_LEVELS])
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS_RQ)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--pre-rounds", type=int, default=R_PRE)
    ap.add_argument("--ft-rounds", type=int, default=R_FT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--consolidate", action="store_true")
    args = ap.parse_args()

    if args.consolidate:
        consolidar()
        return
    verifica_clientes()
    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()] if args.gpus else detect_gpus()) or [0]
    jobs, prontos = [], 0

    if args.fase == "pretrain":
        for m in args.method:
            for e in args.encoder:
                for d in args.dataset:
                    for s in args.seed:
                        if not args.force and backbone(m, e, d, s).exists():
                            prontos += 1
                            continue
                        label = f"pre_{m}_{e}_{d}_seed{s}"
                        jobs.append(Job(label, [
                            sys.executable, str(PRE_RUNNER),
                            "--method", m, "--encoder", e,
                            "--partition", spec(d), "--budget", str(BUDGET),
                            "--rounds", str(args.pre_rounds),
                            "--local-epochs", str(LOCAL_EPOCHS_PRE[m]),
                            "--seed", str(s),
                        ], LOGS / "rq2_pretrain" / f"{label}.log"))
    else:
        PARTS.mkdir(parents=True, exist_ok=True)
        faltando = []
        for m in args.method:
            for e in args.encoder:
                for d in args.dataset:
                    for s in args.seed:
                        bb = backbone(m, e, d, s)
                        if not bb.exists():
                            faltando.append(f"{m}/{e}/{d}/seed{s}")
                            continue
                        for k in args.k:
                            p = parcial(m, e, d, k, s)
                            if not args.force and pronto(p, args.ft_rounds):
                                prontos += 1
                                continue
                            label = f"ft_{m}_{e}_{d}_k{k}_seed{s}"
                            jobs.append(Job(label, [
                                sys.executable, str(FT_RUNNER),
                                "--spec", spec(d), "--encoder", e,
                                "--rounds", str(args.ft_rounds), "--local-epochs", str(E_LOCAL),
                                "--seed", str(s), "--budget", str(BUDGET),
                                "--shots", str(k), "--method", m,
                                "--init-ckpt", str(bb),
                                "--pretrain-rounds", str(args.pre_rounds),
                                "--out", str(p),
                                "--ckpt-dir", str(CKPT / "rq2" / m / e / d / f"k{k}"),
                            ], LOGS / "rq2_finetune" / f"{label}.log"))
        if faltando:
            print(f"[RQ2] ATENÇÃO: {len(faltando)} backbones ausentes — rode a fase "
                  f"pretrain antes. Ex.: {faltando[:3]}", flush=True)

    print(f"[RQ2/{args.fase}] {len(jobs)} jobs ({prontos} prontos) | GPUs={gpus}", flush=True)
    falhas = run_pool(jobs, gpus, args.max_parallel or len(gpus))
    if falhas:
        print(f"[RQ2/{args.fase}] {len(falhas)} falharam: {falhas}", flush=True)
    if args.fase == "finetune":
        consolidar()


if __name__ == "__main__":
    main()
