"""Cobertura das grades experimentais, lida dos caches de `results/`.

Existe para que "o que já rodou" nunca precise ser escrito à mão num doc — a
versão manual dessa contagem já esteve errada em 6 pontos ao mesmo tempo.

Para cada cache: nº de linhas, encoders presentes, e se a grade está completa
(produto dos valores únicos das colunas-chave == nº de linhas).

Rodar: poetry run python scripts/analysis/cache_status.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Colunas que definem a célula de cada grade (as demais são métricas).
KEY_COLS = ["encoder", "source", "seed", "protocol", "n_shots", "target",
            "scenario", "round"]


def describe(path: Path) -> dict:
    df = pd.read_csv(path)
    keys = [c for c in KEY_COLS if c in df.columns]
    expected = 1
    for c in keys:
        expected *= df[c].nunique()
    encoders = sorted(df.encoder.unique()) if "encoder" in df.columns else []
    metrics = [c for c in ("test_acc", "test_f1_macro") if c in df.columns]
    return {
        "cache": path.name,
        "linhas": len(df),
        "grade": expected,
        "completa": "sim" if len(df) == expected else "NÃO",
        "encoders": len(encoders),
        "quais": ",".join(encoders),
        "NaN": int(df[metrics].isna().sum().sum()) if metrics else 0,
    }


def main() -> None:
    caches = sorted(RESULTS.glob("*.csv"))
    if not caches:
        sys.exit(f"nenhum cache em {RESULTS}")
    rows = [describe(p) for p in caches]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n'grade' = produto dos valores únicos das colunas-chave presentes; "
          "divergência indica célula faltando ou duplicada.")


if __name__ == "__main__":
    main()
