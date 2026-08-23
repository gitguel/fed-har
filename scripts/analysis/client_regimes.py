"""Fatos por CLIENTE (usuário) do DAGHAR: o que limita os regimes de rótulo.

`dataset_facts.py` conta janelas por usuário; este conta **janelas por (usuário,
classe)** — a granularidade que decide (a) quantos clientes cada federação pode
ter, (b) qual `n_shots` por cliente é viável, (c) que batch cabe no cliente e
(d) qual regime centralizado é o par de cada regime federado.

Rodar: poetry run python scripts/analysis/client_regimes.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

ROOT = Path(__file__).resolve().parents[2]
VIEW = ROOT / "datasets" / "DAGHAR" / "standardized_view"
LABEL = "standard activity code"
KS = (1, 2, 4, 8, 16, 32, 64)
BATCHES = (64, 32, 16, 8, 4)


def datasets():
    return sorted(d for d in os.listdir(VIEW) if (VIEW / d).is_dir())


def load(name):
    return pd.read_csv(VIEW / name / "train.csv", usecols=["user", LABEL])


def main() -> None:
    if not VIEW.exists():
        sys.exit(f"dataset não encontrado em {VIEW} (ver CLAUDE.md)")

    comp, cell_rows, k_rows, batch_rows = [], [], [], []

    for name in datasets():
        df = load(name)
        users = sorted(df.user.unique())
        classes = sorted(df[LABEL].unique())
        # matriz clientes × classes: janelas de cada (usuário, classe)
        M = (df.groupby(["user", LABEL]).size()
               .unstack(fill_value=0)
               .reindex(index=users, columns=classes, fill_value=0)
               .to_numpy())
        per_user = M.sum(1)
        present = M[M > 0]

        comp.append({
            "dataset": name, "clientes": len(users), "classes": len(classes),
            "janelas": int(M.sum()),
            "jan/cli min": int(per_user.min()),
            "jan/cli med": int(np.median(per_user)),
            "jan/cli max": int(per_user.max()),
            "cli c/ todas classes": int((M > 0).all(1).sum()),
            "células (u,c) preench.": f"{int((M > 0).sum())}/{M.size}",
        })

        cell_rows.append({
            "dataset": name,
            "min>0": int(present.min()), "p10": int(np.percentile(present, 10)),
            "mediana": int(np.median(present)), "p90": int(np.percentile(present, 90)),
            "max": int(present.max()),
            "min incl. classe ausente": int(M.min()),
        })

        for k in KS:
            take = np.minimum(M, k)                 # o que cada célula entrega
            por_classe = take.sum(0)                # total no sistema, por classe
            local = take.sum(1)                     # janelas locais por cliente
            completos = int((M >= k).all(1).sum())  # clientes com k em TODAS as classes
            k_rows.append({
                "dataset": name, "k (fed, /classe/cliente)": k,
                "clientes completos": f"{completos}/{len(users)}",
                "rótulos totais": int(take.sum()),
                "centr. equiv. /classe min": int(por_classe.min()),
                "centr. equiv. /classe max": int(por_classe.max()),
                "jan. locais min": int(local.min()),
                "jan. locais med": int(np.median(local)),
            })

        row = {"dataset": name}
        for b in BATCHES:
            row[f"cli >= {b}"] = f"{int((per_user >= b).sum())}/{len(users)}"
        batch_rows.append(row)

    def show(title, rows, index="dataset"):
        print(f"\n== {title} ==")
        print(pd.DataFrame(rows).set_index(index).to_string())

    show("A. composição por cliente (split train)", comp)
    show("B. janelas por célula (usuário, classe)", cell_rows)
    print("\n== C. batch viável: clientes com >= B janelas no total ==")
    print(pd.DataFrame(batch_rows).set_index("dataset").to_string())
    print("\n== D. regime federado k -> equivalente centralizado ==")
    kd = pd.DataFrame(k_rows).set_index(["dataset", "k (fed, /classe/cliente)"])
    print(kd.to_string())


if __name__ == "__main__":
    main()
