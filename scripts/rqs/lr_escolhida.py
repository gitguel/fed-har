"""Decide e serve a LR do cliente por (encoder, federação, REGIME), a partir da busca S1.

A busca (`run_busca_lr.py`) varre a `LR_GRID` em dois regimes por célula
(encoder, federação): `k=1` e `Full`. Este módulo decide qual LR usar.

## Por que POR REGIME (mudança de 2026-08-24)

O desenho original (`docs/desenho_experimental.md` D6) pedia **uma LR por
federação e encoder**, combinando os dois regimes por posto médio. Os dados da
busca derrubaram essa premissa: dos 20 pares (encoder, federação), só **4
concordam** sobre a melhor LR entre `k=1` e `Full`; **9 discordam por 2+ passos
de grade** (LR 10× diferente), e o viés é sistemático — o `Full` prefere LR maior
em 12/20 células. A amplitude entre a melhor e a pior LR tem mediana de 4,3 pp
(`k=1`) e 6,6 pp (`Full`), então não é ruído de segunda ordem.

Custo do colapso, medido contra o ótimo de cada regime (grade de 4 pontos):

| regra                        | regret `k=1`  | regret `Full` |
|------------------------------|---------------|---------------|
| posto médio (D6 original)    | 0,46 pp       | 1,08 pp       |
| vencedora do `Full` p/ tudo  | 2,76 pp       | 0,00 pp       |
| **por regime (esta)**        | **0,00 pp**   | **0,00 pp**   |

O 1,08 pp do posto médio tem o tamanho do próprio efeito que a RQ1 quer medir
(custo da federação, ~1 pp pelo Exp.2) — ruído de hiperparâmetro do tamanho do
sinal. Por regime custa **zero GPU a mais**: os dois regimes já foram buscados.

## Como `k=2` e `k=4` são servidos

Não são buscados (decisão de escopo de 2026-08-24: a busca para em `k=1` e
`Full`). Eles **herdam a LR do `k=1`** — vizinho mais próximo em escassez de
rótulo. É uma premissa declarada, não uma medição; o regret dela é desconhecido,
limitado por cima pelo regret de aplicar a LR do `k=1` no `Full` (2,76 pp), que é
o caso extremo.

Uma célula só é decidida quando **os dois regimes** têm a `LR_GRID` inteira — é o
que permite disparar a grade federada por federação, à medida que a busca anda.

Uso:
    python scripts/rqs/lr_escolhida.py            # decide o que dá e imprime a tabela
    python scripts/rqs/lr_escolhida.py --pendentes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

from rqs.config import ENCODERS, FEDERACOES, FULL_SHOTS, LR_GRID, RESULTS  # noqa: E402
from common import BEST_LR  # noqa: E402

PARTS = RESULTS / "busca_lr_parts"
TABELA = RESULTS / "lr_escolhida.csv"
REGIMES = ["1", str(FULL_SHOTS)]


def regime_de(k) -> str:
    """Regime cuja LR serve o nível `k`. `k` in {1,2,4} herda a do `k=1`."""
    return str(FULL_SHOTS) if str(k) == str(FULL_SHOTS) else "1"


def _parcial(e: str, d: str, k: str, lr: float, seed: int) -> Path:
    return PARTS / f"{e}_{d}_k{k}_lr{lr:.0e}_seed{seed}.csv"


def _val_acc(p: Path) -> float | None:
    """val_acc da rodada escolhida pela validação (D4). None se o parcial é inútil."""
    import pandas as pd
    try:
        df = pd.read_csv(p).dropna(subset=["val_acc"])
    except Exception:
        return None
    return float(df.val_acc.max()) if len(df) else None


def decidir(seed: int = 0) -> "list[dict]":
    """Decide a LR de cada (encoder, federação, regime) com a LR_GRID completa.

    A célula (encoder, federação) só entra se os DOIS regimes estiverem completos:
    servir um regime e não o outro deixaria a grade federada meio-tunada.
    """
    linhas = []
    for e in ENCODERS:
        for d in sorted(FEDERACOES):
            scores = {}
            for k in REGIMES:
                por_lr = {}
                for lr in LR_GRID:
                    v = _val_acc(_parcial(e, d, k, lr, seed))
                    if v is None:
                        break
                    por_lr[lr] = v
                if len(por_lr) < len(LR_GRID):
                    break
                scores[k] = por_lr
            if len(scores) < len(REGIMES):
                continue

            for k in REGIMES:
                melhor = max(LR_GRID, key=lambda lr: scores[k][lr])
                pior = min(scores[k].values())
                linhas.append(dict(
                    encoder=e, dataset=d, regime=k, lr=melhor, seed=seed,
                    val_acc=round(scores[k][melhor], 4),
                    amplitude_pp=round((scores[k][melhor] - pior) * 100, 2),
                ))
    return linhas


def gravar(linhas: "list[dict]") -> None:
    import pandas as pd
    if not linhas:
        return
    df = pd.DataFrame(linhas).sort_values(["encoder", "dataset", "regime"])
    TABELA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABELA, index=False)
    n_cel = len(df) // len(REGIMES)
    print(f"[LR] {n_cel}/{len(ENCODERS) * len(FEDERACOES)} células decididas "
          f"({len(df)} linhas, uma por regime) -> {TABELA}", flush=True)
    print(df.pivot(index=["dataset", "regime"], columns="encoder",
                   values="lr").to_string(), flush=True)
    iguais = (df.pivot(index=["encoder", "dataset"], columns="regime", values="lr")
                .pipe(lambda t: (t[REGIMES[0]] == t[REGIMES[1]]).sum()))
    print(f"[LR] regimes concordam sobre a LR em {iguais}/{n_cel} células.", flush=True)


def _cache() -> dict:
    import pandas as pd
    if not TABELA.exists():
        return {}
    df = pd.read_csv(TABELA)
    if "regime" not in df.columns:      # tabela do desenho antigo (uma LR por célula)
        return {}
    return {(r.encoder, r.dataset, str(r.regime)): float(r.lr) for r in df.itertuples()}


def lr_de(encoder: str, dataset: str, k) -> float:
    """LR do cliente para aquela célula e nível de rótulo.

    `k` in {1,2,4} usa a LR do regime `k=1`; `Full` usa a dele. Cai no
    `BEST_LR[encoder]` enquanto a busca não decidiu a célula.
    """
    return _cache().get((encoder, dataset, regime_de(k)), BEST_LR[encoder])


def decididas() -> set:
    """Células (encoder, dataset) com os DOIS regimes fechados pela busca."""
    c = _cache()
    return {(e, d) for (e, d, r) in c
            if all((e, d, reg) in c for reg in REGIMES)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pendentes", action="store_true", help="Lista as células que faltam.")
    args = ap.parse_args()

    linhas = decidir(args.seed)
    gravar(linhas)
    if args.pendentes:
        prontas = {(l["encoder"], l["dataset"]) for l in linhas}
        falta = [f"{e}/{d}" for e in ENCODERS for d in sorted(FEDERACOES)
                 if (e, d) not in prontas]
        print(f"\n[LR] {len(falta)} células pendentes: {falta}", flush=True)


if __name__ == "__main__":
    main()
