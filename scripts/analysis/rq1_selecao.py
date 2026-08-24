"""RQ1: quanto do Δ federado−centralizado é otimismo de seleção?

Só LÊ caches (`results/rqs/`), não treina nada.

## O problema

Os dois braços da RQ1 escolhem o modelo reportado de maneiras diferentes:

| braço | como escolhe |
|---|---|
| centralizado | early stopping, **paciência 50** em `val_loss`, ≤100 épocas — protocolo do benchmark (da Luz et al., §V, verificado no PDF em 2026-08-24) |
| federado | **argmax de `val_acc` sobre as 150 rodadas** — regra nossa, sem contrapartida no benchmark |

São duas diferenças: a **métrica** (`val_loss` × `val_acc`; selecionar na mesma
família do que se reporta é mais otimista) e o **lookahead** (paciência limitada ×
varredura completa). O próprio benchmark fixa hiperparâmetros entre condições
justamente para que ganhos não sejam "an artifact of better-tuned hyperparameters"
(Apêndice A) — o mesmo princípio se aplica à seleção.

## O que este script faz

Emula no cache do federado a **disciplina** do early stopping: percorre as rodadas,
guarda a melhor `val_acc` e para quando passam `paciência` rodadas sem melhora,
reportando o `test_acc` da melhor rodada *até ali*. Varre várias paciências para o
resultado não ficar refém de um valor arbitrário.

**O que ele NÃO corrige:** a métrica. `val_loss` não existe no cache federado
(`run_cross_device._eval_val` devolve só acc/F1). A partir de 2026-08-24 o runner
grava `val_loss`; runs anteriores não têm. Então o número aqui é um **limite
superior do custo de federar corrigido** — ainda sobra o otimismo de métrica.

Uso:
    poetry run python scripts/analysis/rq1_selecao.py
    poetry run python scripts/analysis/rq1_selecao.py --paciencias 10 25 50 75
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

from rqs.config import K_LEVELS, RESULTS, R_FT  # noqa: E402

PARTS = RESULTS / "rq1_federado_parts"
CHAVE = ["encoder", "target", "n_shots", "seed"]


def carrega_federado():
    """Todas as rodadas das células federadas COMPLETAS."""
    import pandas as pd
    fs = sorted(glob.glob(str(PARTS / "*.csv")))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df.dropna(subset=["test_acc", "val_acc"])
    cont = df.groupby(CHAVE)["round"].count()
    completos = cont[cont >= R_FT].index
    return df.set_index(CHAVE).loc[completos].reset_index().sort_values(CHAVE + ["round"])


def escolhe_com_paciencia(g, paciencia: int):
    """Emula EarlyStopping(patience) + ModelCheckpoint(best) sobre as rodadas.

    Devolve o `test_acc` da melhor rodada vista até a parada. `paciencia` maior ou
    igual ao nº de rodadas equivale ao argmax global (a regra atual, D4).
    """
    melhor_val, melhor_test, melhor_r = -1.0, None, -1
    for r, v, t in zip(g["round"].values, g.val_acc.values, g.test_acc.values):
        if v > melhor_val:
            melhor_val, melhor_test, melhor_r = v, t, r
        elif r - melhor_r > paciencia:
            break
    return melhor_test, melhor_r


def main() -> None:
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paciencias", nargs="+", type=int, default=[10, 25, 50, 75, R_FT])
    args = ap.parse_args()

    fed = carrega_federado()
    if fed.empty:
        print("[RQ1-SEL] sem parciais federados completos.", flush=True)
        return
    cen = pd.read_csv(RESULTS / "rq1_centralizado.csv")[
        ["encoder", "dataset", "k", "seed", "test_acc"]]
    cen["k"] = cen["k"].astype(str)
    KS = [str(k) for k in K_LEVELS]

    linhas, rodadas = {}, {}
    for pac in args.paciencias:
        esc = []
        for chave, g in fed.groupby(CHAVE, sort=False):
            t, r = escolhe_com_paciencia(g, pac)
            esc.append((*chave, t, r))
        e = pd.DataFrame(esc, columns=CHAVE + ["test_acc", "round"])
        e = e.rename(columns={"target": "dataset", "n_shots": "k"})
        e["k"] = e["k"].astype(str)
        m = cen.merge(e, on=["encoder", "dataset", "k", "seed"], suffixes=("_c", "_f"))
        m["d"] = (m.test_acc_f - m.test_acc_c) * 100
        linhas[f"pac={pac}"] = m.groupby("k").d.mean().reindex(KS)
        rodadas[f"pac={pac}"] = m.groupby("k")["round"].mean().reindex(KS)
        n = m.groupby("k").size().reindex(KS)

    tab = pd.DataFrame(linhas)
    tab.insert(0, "n", n)
    print("\nΔ = federado − centralizado (pp), por paciência do early stopping emulado")
    print("(`pac=%d` = argmax global = a regra atual, D4)\n" % R_FT)
    print(tab.round(2).to_string())
    print("\nRodada escolhida (média) sob cada paciência:")
    print(pd.DataFrame(rodadas).round(0).to_string())
    ref = f"pac={R_FT}"
    if ref in tab and "pac=50" in tab:
        print(f"\nOtimismo de lookahead (D4 − paciência 50), por regime:")
        print((tab[ref] - tab["pac=50"]).round(2).to_string())
    print("\nATENÇÃO: isto corrige o lookahead, NÃO a métrica (val_acc × val_loss).")
    print("O Δ real do custo de federar é MAIS negativo do que a coluna pac=50.")


if __name__ == "__main__":
    main()
