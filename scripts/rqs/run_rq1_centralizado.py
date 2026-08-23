"""RQ1, eixo centralizado: treina e avalia in-domain nos regimes pareados.

Regime = `k × n_clientes` samples/class (docs/desenho_experimental.md §2.3), então
o `n_shots` **muda por dataset**: `k=1` é 10 no RealWorld e 36 no WISDM.

Grade: 4 encoders × 5 datasets × 4 regimes × seeds. Escreve em
`results/rqs/rq1_centralizado.csv` e em `checkpoints/rqs/supervised/`.

Uso:
    python scripts/rqs/run_rq1_centralizado.py --seed 0        # uma seed por vez
    python scripts/rqs/run_rq1_centralizado.py --consolidate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

from rqs.config import (CKPT, ENCODERS, FEDERACOES, FULL_SHOTS, K_LEVELS, LOGS,
                        MAX_EPOCHS_CENTR, RESULTS, SEEDS_RQ, shots_centralizado,
                        verifica_clientes)
from common import supervised_ckpt_dir  # noqa: E402
from gpu_pool import Job, detect_gpus, run_pool  # noqa: E402

SUP_DIR = Path(__file__).resolve().parents[1] / "supervised"
CACHE = RESULTS / "rq1_centralizado.csv"
KEY = ["encoder", "dataset", "seed", "k", "n_shots", "target"]


def avaliar(encoder, dataset, seed, k, n_shots):
    """Avalia in-domain o best.ckpt daquele treino. Devolve None se não existe."""
    import torch
    from eval_transfer import evaluate, load_checkpoint, test_loader
    ckpt = supervised_ckpt_dir(encoder, dataset, seed, n_shots) / "best.ckpt"
    if not ckpt.exists():
        return None
    model = load_checkpoint(encoder, ckpt)
    acc, f1 = evaluate(model, test_loader(dataset))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(encoder=encoder, dataset=dataset, seed=seed, k=str(k),
                n_shots=str(n_shots), target=dataset, test_acc=acc, test_f1_macro=f1)


def consolidar(rows) -> None:
    import pandas as pd
    novo = pd.DataFrame(rows)
    if CACHE.exists():
        novo = pd.concat([pd.read_csv(CACHE), novo], ignore_index=True)
    novo = novo.drop_duplicates(subset=KEY, keep="last").sort_values(KEY)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    novo.to_csv(CACHE, index=False)
    print(f"[CACHE] {len(novo)} linhas -> {CACHE}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", nargs="+", choices=ENCODERS, default=ENCODERS)
    ap.add_argument("--dataset", nargs="+", choices=sorted(FEDERACOES), default=sorted(FEDERACOES))
    ap.add_argument("--k", nargs="+", default=[str(k) for k in K_LEVELS])
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS_RQ)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--consolidate", action="store_true", help="Só avalia o que já treinou.")
    args = ap.parse_args()
    verifica_clientes()

    ks = [FULL_SHOTS if str(x) == FULL_SHOTS else int(x) for x in args.k]
    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()] if args.gpus else detect_gpus()) or [0]

    combos = [(e, d, s, k, shots_centralizado(d, k))
              for e in args.encoder for d in args.dataset for s in args.seed for k in ks]

    if not args.consolidate:
        jobs = []
        for e, d, s, k, n in combos:
            if not args.force and (supervised_ckpt_dir(e, d, s, n) / "best.ckpt").exists():
                continue
            label = f"{e}_{d}_seed{s}_k{k}_shots{n}"
            jobs.append(Job(label, [
                sys.executable, str(SUP_DIR / f"train_{e}.py"),
                "--dataset", d, "--seed", str(s), "--shots", str(n),
                "--max-epochs", str(MAX_EPOCHS_CENTR),
            ], LOGS / "rq1_centralizado" / f"{label}.log"))
        print(f"[RQ1-C] {len(jobs)} treinos de {len(combos)} combos | GPUs={gpus}", flush=True)
        falhas = run_pool(jobs, gpus, args.max_parallel or len(gpus))
        if falhas:
            print(f"[RQ1-C] {len(falhas)} falharam: {falhas}", flush=True)

    print("[RQ1-C] avaliando in-domain (serial)...", flush=True)
    rows = [r for r in (avaliar(e, d, s, k, n) for e, d, s, k, n in combos) if r]
    if rows:
        consolidar(rows)
    else:
        print("[RQ1-C] nenhum checkpoint para avaliar.", flush=True)


if __name__ == "__main__":
    main()
