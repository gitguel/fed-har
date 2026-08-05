"""Ladder de rótulos do Fed-SSL cross-device e transferência do pré-treino.

Responde às duas perguntas que ficaram na fila depois que a grade fechou
(2026-08-04), lendo **só** caches de `results/`:

  A. **A vantagem do SSL sobrevive à supervisão plena?** A grade do Fed-SSL só
     roda `method=none` em shots [1,2,5,10]; o degrau `full` sem pré-treino vive
     em `fed_cross_device.csv`. Este script faz o join e completa a ladder.

  B. **O pré-treino precisa ser no domínio do alvo?** `fedssl_crossspec.csv` tem
     o braço com `pretrain_spec != spec` (backbone pré-treinado nos usuários de
     um dataset, finetuning federado nos usuários do outro). Contra o pré-treino
     in-domain e contra o `none`, isola quanto da representação é transferível.

**O join do `full` só é honesto porque os protocolos coincidem** — mesmo
`budget=192`, `local_epochs=5`, R=150, seeds 0–3, mesmos encoders e specs.
`fed_cross_device.csv` também guarda um braço `local_epochs=1` (R=100) de
calibração: ele é **descartado** aqui, senão o degrau `full` compararia contra
outro protocolo. Os specs `iid:*` também saem (são o controle de feature skew,
não têm par no braço SSL).

**Três protocolos de seleção de rodada, sempre.** Um Δ que só existe sob um
deles é artefato de seleção — já aconteceu com o `resnetse5`:
  - `val`    — argmax da média de `val_acc` entre os alvos do run (é a regra do
               `best.ckpt` em `run_cross_device.py`; o teste nunca decide);
  - `last`   — última rodada;
  - `tail20` — média do teste nas 20 últimas rodadas.

Saída: tabelas no stdout e o tidy derivado
`results/derived/fedssl_selected.csv` (uma linha por run × alvo × protocolo),
que é o que os notebooks devem ler em vez de reimplementar a seleção.

Rodar: poetry run python scripts/analysis/fedssl_ladder.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DERIVED = RESULTS / "derived"

# Uma "célula" é um run; ele cobre 1 ou 2 alvos (specs mistos têm 2).
RUN_KEY = ["method", "encoder", "spec", "pretrain_spec", "n_shots", "seed"]
CELL_KEY = RUN_KEY + ["target"]
METRICS = ["test_acc", "test_f1_macro"]

SHOT_ORDER = ["1", "2", "5", "10", "full"]
NO_PRETRAIN = "-"          # marcador de "sem pré-treino" na coluna pretrain_spec
IN_DOMAIN = "in-domain"    # marcador de "pré-treino no mesmo spec do finetuning"


# --------------------------------------------------------------------------- #
# carga                                                                        #
# --------------------------------------------------------------------------- #
def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # `fed_cross_device.csv` é anterior à ladder e não tem a coluna: aquele
    # cache É o degrau de supervisão plena.
    df["n_shots"] = df.get("n_shots", pd.Series("full", index=df.index)).astype(str)
    df["seed"] = df["seed"].astype(int)
    df["round"] = df["round"].astype(int)
    return df


def load_unified() -> pd.DataFrame:
    """Junta as três fontes num único frame com o mesmo esquema."""
    base = _norm(pd.read_csv(RESULTS / "fedssl_cross_device.csv", low_memory=False))
    # Na grade base o pré-treino é sempre no mesmo spec do finetuning; o `none`
    # não tem pré-treino nenhum. A coluna vem NaN nos dois casos, então o
    # significado tem de ser reconstruído a partir do método.
    base["pretrain_spec"] = base["method"].map(
        lambda m: NO_PRETRAIN if m == "none" else IN_DOMAIN)

    cross = _norm(pd.read_csv(RESULTS / "fedssl_crossspec.csv", low_memory=False))

    # Degrau `full` sem pré-treino: só o braço que casa com o protocolo do SSL.
    full = _norm(pd.read_csv(RESULTS / "fed_cross_device.csv", low_memory=False))
    full = full[(full.local_epochs == 5) & (~full.spec.str.startswith("iid:"))].copy()
    full["method"] = "none"
    full["n_shots"] = "full"
    full["pretrain_rounds"] = 0
    full["pretrain_spec"] = NO_PRETRAIN

    df = pd.concat([base, cross, full], ignore_index=True)

    dup = df.duplicated(subset=CELL_KEY + ["round"]).sum()
    if dup:
        raise SystemExit(f"[ERRO] {dup} linhas duplicadas na chave da célula — "
                         "o join colapsou runs distintos.")
    return df


# --------------------------------------------------------------------------- #
# seleção de rodada                                                            #
# --------------------------------------------------------------------------- #
def select_rounds(df: pd.DataFrame, tail: int = 20) -> pd.DataFrame:
    """Reduz cada run a uma linha por alvo, sob os três protocolos."""
    out = []

    # `val`: a rodada é escolhida no run inteiro (média de val_acc entre alvos),
    # não por alvo — é o que o `best.ckpt` faz. Escolher por alvo vazaria a
    # estrutura do teste na decisão em specs de 2 alvos.
    per_round = df.groupby(RUN_KEY + ["round"], dropna=False)["val_acc"].mean()
    best = per_round.groupby(level=list(range(len(RUN_KEY)))).idxmax()
    best_rounds = pd.DataFrame(
        [i for i in best.values], columns=RUN_KEY + ["round"])
    sel = df.merge(best_rounds, on=RUN_KEY + ["round"], how="inner").copy()
    sel["protocol"] = "val"
    out.append(sel)

    last_rounds = (df.groupby(RUN_KEY, dropna=False)["round"].max()
                     .rename("round").reset_index())
    sel = df.merge(last_rounds, on=RUN_KEY + ["round"], how="inner").copy()
    sel["protocol"] = "last"
    out.append(sel)

    rmax = df.groupby(RUN_KEY, dropna=False)["round"].transform("max")
    tailed = df[df["round"] > rmax - tail]
    sel = (tailed.groupby(CELL_KEY, dropna=False)[METRICS].mean()
                 .reset_index())
    sel["protocol"] = f"tail{tail}"
    out.append(sel)

    keep = CELL_KEY + METRICS + ["protocol"]
    return pd.concat([o[keep] for o in out], ignore_index=True)


# --------------------------------------------------------------------------- #
# A. ladder de rótulos, agora com o degrau `full`                              #
# --------------------------------------------------------------------------- #
def ladder(sel: pd.DataFrame, metric: str = "test_f1_macro") -> pd.DataFrame:
    """Δ(SSL − none) por encoder × n_shots, pareado dentro de spec/alvo/seed."""
    # Pareamento explícito: o Δ só é legítimo contra o MESMO spec, alvo e seed.
    pair_key = ["protocol", "encoder", "spec", "target", "n_shots", "seed"]
    base = sel[sel.method == "none"].set_index(pair_key)[metric]
    ssl = sel[sel.method.isin(["tfc", "lfr"]) & (sel.pretrain_spec == IN_DOMAIN)]

    d = ssl.join(base.rename("none"), on=pair_key)
    missing = d["none"].isna().sum()
    if missing:
        print(f"[aviso] {missing} células SSL sem par `none` — excluídas.")
    d = d.dropna(subset=["none"])
    d["delta"] = d[metric] - d["none"]

    # dp ENTRE SEEDS (cada seed reduzida à sua média), convenção do repo.
    per_seed = d.groupby(["protocol", "method", "encoder", "n_shots", "seed"])["delta"].mean()
    g = per_seed.groupby(level=["protocol", "method", "encoder", "n_shots"])
    return pd.DataFrame({"delta_pp": g.mean() * 100, "dp_seeds_pp": g.std() * 100,
                         "n_seeds": g.size()}).reset_index()


# --------------------------------------------------------------------------- #
# B. o pré-treino precisa ser in-domain?                                       #
# --------------------------------------------------------------------------- #
def crossspec(sel: pd.DataFrame, metric: str = "test_f1_macro") -> pd.DataFrame:
    """Compara pré-treino in-domain vs cross-domain vs nenhum, no mesmo alvo."""
    specs = sorted(sel[sel.pretrain_spec.str.startswith("device:")].spec.unique())
    s = sel[sel.spec.isin(specs)].copy()
    s["arm"] = s["pretrain_spec"].map(
        lambda p: {NO_PRETRAIN: "none", IN_DOMAIN: "in"}.get(p, "cross"))
    # `none` não tem método SSL; replica-se como piso dos dois braços.
    floor = s[s.arm == "none"].drop(columns="method")
    floor = pd.concat([floor.assign(method=m) for m in ("tfc", "lfr")],
                      ignore_index=True)
    s = pd.concat([s[s.arm != "none"], floor], ignore_index=True)

    pair = ["protocol", "method", "encoder", "spec", "target", "n_shots", "seed"]
    w = s.pivot_table(index=pair, columns="arm", values=metric)
    w = w.dropna(subset=["in", "cross", "none"])
    w["ganho_in"] = w["in"] - w["none"]
    w["ganho_cross"] = w["cross"] - w["none"]
    w["custo_transf"] = w["cross"] - w["in"]

    cols = ["none", "in", "cross", "ganho_in", "ganho_cross", "custo_transf"]
    per_seed = w.groupby(["protocol", "method", "encoder", "spec", "seed"])[cols].mean()
    g = per_seed.groupby(level=["protocol", "method", "encoder", "spec"])
    out = (g.mean() * 100).round(2)
    out["dp_custo_pp"] = (g["custo_transf"].std() * 100).round(2)
    return out.reset_index()


def retention(cs: pd.DataFrame, min_gain_pp: float = 2.0) -> pd.DataFrame:
    """Fração do ganho in-domain que sobrevive ao pré-treino fora do domínio.

    A razão só é interpretável onde há ganho a reter: com `ganho_in` perto de
    zero ela explode ou troca de sinal sem significar nada. Por isso o corte em
    `min_gain_pp` — as células cortadas não são "sem retenção", são **sem
    efeito para medir**, e aparecem na contagem `n_celulas_sem_ganho`.
    """
    keep = cs[cs.ganho_in >= min_gain_pp].copy()
    keep["retencao_%"] = (keep.ganho_cross / keep.ganho_in * 100).round(0)
    cols = ["protocol", "method", "encoder", "spec", "ganho_in",
            "ganho_cross", "retencao_%"]
    return keep[cols].sort_values(["protocol", "method", "encoder", "spec"])


def _show(title: str, df: pd.DataFrame, index: bool = False) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(df.to_string(index=index))


def main() -> None:
    pd.set_option("display.width", 200)
    df = load_unified()
    print(f"[carga] {len(df)} linhas | "
          f"{df.groupby(CELL_KEY, dropna=False).ngroups} células × alvo")

    sel = select_rounds(df)
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = DERIVED / "fedssl_selected.csv"
    sel.to_csv(out, index=False)
    print(f"[tidy] {len(sel)} linhas -> {out}")

    lad = ladder(sel)
    lad["n_shots"] = pd.Categorical(lad.n_shots, SHOT_ORDER, ordered=True)
    for proto in ("val", "last", "tail20"):
        p = lad[lad.protocol == proto].pivot_table(
            index=["method", "encoder"], columns="n_shots",
            values="delta_pp", observed=True)
        _show(f"A. Δ F1-macro (SSL − none), pp — seleção `{proto}`",
              p.round(1), index=True)

    _show("A-dp. dispersão entre seeds do Δ (pp), seleção `val`",
          lad[lad.protocol == "val"].pivot_table(
              index=["method", "encoder"], columns="n_shots",
              values="dp_seeds_pp", observed=True).round(1), index=True)

    cs = crossspec(sel)
    for proto in ("val", "last", "tail20"):
        _show(f"B. pré-treino in-domain vs cross-domain (F1-macro %, "
              f"média sobre shots/seeds) — seleção `{proto}`",
              cs[cs.protocol == proto].drop(columns="protocol"))

    ret = retention(cs)
    _show("B-resumo. retenção do ganho quando o pré-treino é fora do domínio "
          "(só células com ganho_in ≥ 2 pp)", ret)
    print(f"\ncélulas com ganho_in < 2 pp (razão não interpretável): "
          f"{len(cs) - len(ret)} de {len(cs)}")
    med = ret.groupby("protocol")["retencao_%"].median()
    print("mediana da retenção por protocolo (%):",
          ", ".join(f"{k}={v:.0f}" for k, v in med.items()))


if __name__ == "__main__":
    main()
