"""Regenera as tabelas de fatos de `docs/dados_daghar.md` a partir dos CSVs.

Fonte única dos números sobre os dados (classes, usuários, janelas, skews). Se
uma tabela do doc divergir da saída daqui, o doc está errado — nunca o contrário.

Rodar: poetry run python scripts/analysis/dataset_facts.py
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

ROOT = Path(__file__).resolve().parents[2]
VIEW = ROOT / "datasets" / "DAGHAR" / "standardized_view"
LABEL = "standard activity code"
FEATURE_RE = re.compile(r"(accel|gyro)-[xyz]-\d+$")
BATCH = 64  # piso de viabilidade do cliente em SSL (ver plano_fedssl.md §3)


def datasets():
    return sorted(d for d in os.listdir(VIEW) if (VIEW / d).is_dir())


def label_skew(df) -> float:
    """TV média entre a distribuição de classes de cada usuário e a global."""
    p_global = df[LABEL].value_counts(normalize=True)
    per_user = []
    for _, g in df.groupby("user"):
        p_user = g[LABEL].value_counts(normalize=True).reindex(p_global.index).fillna(0)
        per_user.append(0.5 * np.abs(p_user - p_global).sum())
    return float(np.mean(per_user))


def feature_skew(df, feats) -> float:
    """η² entre-usuários das features, CONTROLADO por classe.

    Por classe: SS_entre-usuários / SS_total, média sobre features; depois média
    sobre classes. O controle por classe é essencial — sem ele a variância
    "andar × sentar" dominaria e o número mediria dificuldade da tarefa, não
    heterogeneidade entre pessoas.
    """
    per_class = []
    for _, gc in df.groupby(LABEL):
        x = gc[feats].to_numpy(float)
        grand = x.mean(0)
        ss_total = ((x - grand) ** 2).sum(0)
        ss_between = np.zeros(len(feats))
        for _, gu in gc.groupby("user"):
            mean_u = gu[feats].to_numpy(float).mean(0)
            ss_between += len(gu) * (mean_u - grand) ** 2
        ok = ss_total > 0
        per_class.append((ss_between[ok] / ss_total[ok]).mean())
    return float(np.mean(per_class))


def main() -> None:
    if not VIEW.exists():
        sys.exit(f"dataset não encontrado em {VIEW} (ver CLAUDE.md para baixar)")

    rows, skews = [], []
    for name in datasets():
        splits = {}
        for split in ("train", "validation", "test"):
            splits[split] = pd.read_csv(VIEW / name / f"{split}.csv")
        train = splits["train"]
        feats = [c for c in train.columns if FEATURE_RE.match(c)]
        per_user = train.user.value_counts()
        classes = sorted(set().union(*(set(d[LABEL].unique()) for d in splits.values())))
        rows.append({
            "dataset": name,
            "classes": len(classes),
            "codigos": ",".join(str(c) for c in classes),
            "users_train": splits["train"].user.nunique(),
            "users_val": splits["validation"].user.nunique(),
            "users_test": splits["test"].user.nunique(),
            "jan_min": per_user.min(),
            "jan_med": int(per_user.median()),
            "jan_max": per_user.max(),
            f"users_lt_{BATCH}": int((per_user < BATCH).sum()),
            "janelas_train": len(train),
        })
        skews.append({"dataset": name,
                      "TV_rotulo": label_skew(train),
                      "eta2_feature": feature_skew(train, feats)})

    facts = pd.DataFrame(rows).set_index("dataset")
    print("== §1/§3 composição, classes e partição por usuário ==")
    print(facts.to_string())
    print(f"\ntotal de usuários no train = {facts.users_train.sum()}"
          f" | janelas no train (= corpus `combined`) = {facts.janelas_train.sum()}")
    print("\n== §4 heterogeneidade ==")
    print(pd.DataFrame(skews).set_index("dataset").round(3).to_string())


if __name__ == "__main__":
    main()
