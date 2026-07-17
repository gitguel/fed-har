"""Particiona o train.csv de cada dataset DAGHAR por usuário (cross-device).

Gera, para todos os 6 datasets:

    datasets/DAGHAR/user_partitions/<dataset>/user_<id>.csv   (todas as colunas)
    datasets/DAGHAR/user_partitions/manifest.csv

O manifest tem uma linha por (dataset, user) com nº de janelas e contagem por
classe (`n_class_0..5`). Os splits validation/test do DAGHAR são disjuntos por
usuário em todas as bases (verificado em 2026-07-17), então os clientes usam
apenas o train.csv e a avaliação downstream continua usando os splits padrão.

Uso:
    poetry run python scripts/federated/partition_users.py [--force]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts
from common import DAGHAR_ROOT, PROJECT_ROOT  # noqa: E402

OUT_ROOT = DAGHAR_ROOT.parent / "user_partitions"
LABEL_COL = "standard activity code"
N_CLASSES = 6


def norm_user_id(raw) -> str:
    """Normaliza o id (int/float/str) para um token estável de nome de arquivo."""
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    return str(raw).replace(" ", "_")


def partition_dataset(dataset_dir: Path, out_dir: Path) -> list[dict]:
    df = pd.read_csv(dataset_dir / "train.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw_id, group in df.groupby("user", sort=True):
        uid = norm_user_id(raw_id)
        path = out_dir / f"user_{uid}.csv"
        group.to_csv(path, index=False)
        counts = group[LABEL_COL].value_counts()
        row = {
            "dataset": dataset_dir.name,
            "user": uid,
            "n_windows": len(group),
            "path": str(path.relative_to(PROJECT_ROOT)),
        }
        for c in range(N_CLASSES):
            row[f"n_class_{c}"] = int(counts.get(c, 0))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="regera mesmo se user_partitions/ já existir")
    args = parser.parse_args()

    if OUT_ROOT.exists() and not args.force:
        sys.exit(f"{OUT_ROOT} já existe; use --force para regerar.")

    all_rows = []
    for dataset_dir in sorted(DAGHAR_ROOT.iterdir()):
        if not (dataset_dir / "train.csv").exists():
            continue
        rows = partition_dataset(dataset_dir, OUT_ROOT / dataset_dir.name)
        all_rows.extend(rows)
        n_win = sum(r["n_windows"] for r in rows)
        print(f"{dataset_dir.name:18s} {len(rows):3d} usuários, {n_win} janelas")

    manifest = pd.DataFrame(all_rows)
    manifest.to_csv(OUT_ROOT / "manifest.csv", index=False)
    print(f"\nManifest: {OUT_ROOT / 'manifest.csv'} ({len(manifest)} clientes)")


if __name__ == "__main__":
    main()
