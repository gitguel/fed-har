#!/usr/bin/env python3
"""Wilcoxon pareado por par (técnica SSL × encoder) — a rota que não custa seed.

Motivação (auditoria de variância de 2026-08-05, §12 do notebook Fed-SSL): com 4
seeds e `dp(Δ) ≈ 2,5 pp`, nenhuma célula isolada sustenta um claim. O benchmark do
da Luz tem **3** seeds e mesmo assim afirma coisas — porque o `n` dos testes dele
não é o número de seeds, é o número de **configurações pareadas**. Este script
aplica a mesma rota aos nossos caches.

Quatro blocos, todos só de LEITURA:

  A. **Réplica vs Tabela 10 do paper** (p. 18). 48 células (4 encoders × 3 métodos
     × 4 regimes), média sobre os 6 datasets in-domain. É o que nos dá — ou não —
     o direito de invocar o benchmark como referência.

  B. **Wilcoxon centralizado**: SSL − supervisionado, por par, pareado sobre
     configurações (dataset × regime). Bonferroni sobre os 8 pares.

  C. **Wilcoxon federado**: SSL − FedAvg supervisionado, mesmos 8 pares, pareado
     sobre (spec × alvo × regime). Responde se a conclusão centralizada sobrevive
     à federação.

  D. **Recorte por spec**, incluindo `5+5` e `10+10` (as duas federações mistas).

**Duas unidades de pareamento, sempre.** A `seed` como observação (n grande) infla
o `n` com pseudo-réplica: 4 seeds da mesma configuração não são 4 observações
independentes. A unidade **honesta** é a configuração com as seeds já mediadas —
é ela que o script chama de primária. A outra fica ao lado como diagnóstico; se as
duas discordam, quem manda é a primária.

Uso:
    poetry run python scripts/analysis/wilcoxon_pares.py [--outdir <pasta>]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"

DS6 = ["KuHar", "MotionSense", "RealWorld_thigh", "RealWorld_waist", "UCI", "WISDM"]
ENCODERS = ["resnetse5", "cnnpff", "rnn", "tstcc"]
METODOS = ["lfr", "tfc"]
LAB = {"lfr": "LFR", "tfc": "TF-C", "sup": "supervisionado", "none": "FedAvg sup."}
ALFA = 0.05

# Tabela 10 do paper (p. 18): média sobre os 6 datasets × 3 seeds, full fine-tuning.
# Só as linhas dos 4 encoders que também são nossos. (média, dp)
BENCH_T10 = {
    ("sup", "resnetse5"): {"1": 50.3, "10": 69.3, "100": 75.5, "full": 79.4},
    ("sup", "tstcc"):     {"1": 45.2, "10": 60.0, "100": 73.3, "full": 76.1},
    ("sup", "cnnpff"):    {"1": 34.2, "10": 55.4, "100": 74.6, "full": 77.7},
    ("sup", "rnn"):       {"1": 31.7, "10": 40.6, "100": 64.5, "full": 69.1},
    ("tfc", "resnetse5"): {"1": 39.6, "10": 69.7, "100": 81.6, "full": 83.6},
    ("tfc", "tstcc"):     {"1": 50.6, "10": 68.5, "100": 76.4, "full": 79.7},
    ("tfc", "cnnpff"):    {"1": 45.3, "10": 77.0, "100": 85.1, "full": 86.1},
    ("tfc", "rnn"):       {"1": 44.9, "10": 69.5, "100": 78.8, "full": 83.7},
    ("lfr", "resnetse5"): {"1": 52.4, "10": 70.8, "100": 75.8, "full": 79.8},
    ("lfr", "tstcc"):     {"1": 41.2, "10": 65.0, "100": 75.9, "full": 81.3},
    ("lfr", "cnnpff"):    {"1": 37.9, "10": 55.5, "100": 71.5, "full": 74.0},
    ("lfr", "rnn"):       {"1": 41.6, "10": 58.7, "100": 69.2, "full": 71.3},
}


# --------------------------------------------------------------------------- #
# carga                                                                        #
# --------------------------------------------------------------------------- #
def carrega_centralizado() -> pd.DataFrame:
    """In-domain, full fine-tuning: a fatia que casa com a Tabela 10."""
    fr = []
    for m, f in [("sup", "supervised_eval_transfer"),
                 ("lfr", "ssl_lfr_eval_transfer"), ("tfc", "ssl_tfc_eval_transfer")]:
        d = pd.read_csv(RESULTS / f"{f}.csv")
        if "protocol" in d.columns:          # o SL não tem linear readout
            d = d[d.protocol == "finetune"]
        d = d[(d.source == d.target) & d.source.isin(DS6)].copy()
        d["method"] = m
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["n_shots"] = d.n_shots.astype(str)
    d["acc"] = d.test_acc * 100
    d["config"] = d.target + "|" + d.n_shots           # unidade de pareamento
    return d[["method", "encoder", "config", "target", "n_shots", "seed", "acc"]]


def carrega_federado() -> pd.DataFrame:
    """`fedssl_selected.csv`: uma linha por run × alvo × protocolo de seleção.

    Fica só o `protocol == "val"` (a rodada escolhida pela validação, que é a regra
    do `best.ckpt`) e o pré-treino `in-domain` — o braço cruzado é outra pergunta.
    """
    d = pd.read_csv(RESULTS / "derived" / "fedssl_selected.csv")
    d = d[(d.protocol == "val")
          & (d.pretrain_spec.isin(["in-domain", "-"]))].copy()
    d["n_shots"] = d.n_shots.astype(str)
    d["acc"] = d.test_acc * 100
    d["config"] = d.spec + "|" + d.target + "|" + d.n_shots
    d["method"] = d.method.replace({"none": "sup"})
    return d[["method", "encoder", "spec", "config", "target", "n_shots", "seed", "acc"]]


# --------------------------------------------------------------------------- #
# o teste                                                                      #
# --------------------------------------------------------------------------- #
def _wilcoxon(dif: np.ndarray) -> dict:
    """Wilcoxon dos postos sinalizados, bilateral, + tamanho de efeito.

    `rbc` é a correlação bisserial de postos: (W+ − W−)/(W+ + W−) ∈ [−1, 1]. É o
    tamanho de efeito próprio do teste — o `p` sozinho só diz "não é zero".
    """
    dif = np.asarray(dif, float)
    dif = dif[~np.isnan(dif)]
    n = len(dif)
    if n < 6 or np.all(dif == 0):
        return dict(n=n, mediana=np.median(dif) if n else np.nan,
                    p=np.nan, rbc=np.nan, pos=int((dif > 0).sum()))
    stat, p = wilcoxon(dif, alternative="two-sided", zero_method="wilcox")
    r = np.argsort(np.argsort(np.abs(dif))) + 1.0          # postos dos |dif|
    wpos, wneg = r[dif > 0].sum(), r[dif < 0].sum()
    return dict(n=n, mediana=float(np.median(dif)), p=float(p),
                rbc=float((wpos - wneg) / (wpos + wneg)), pos=int((dif > 0).sum()))


def testa_pares(d: pd.DataFrame, ref: str = "sup", chave=("config",),
                por_seed: bool = False) -> pd.DataFrame:
    """Para cada (método SSL, encoder): Wilcoxon do Δ contra `ref`, pareado.

    `por_seed=False` (primário) media as seeds dentro da configuração antes de
    parear; `True` mantém cada seed como observação (diagnóstico de pseudo-réplica).
    """
    idx = list(chave) + (["seed"] if por_seed else [])
    linhas = []
    for m in METODOS:
        for e in ENCODERS:
            a = d[(d.method == m) & (d.encoder == e)].groupby(idx).acc.mean()
            b = d[(d.method == ref) & (d.encoder == e)].groupby(idx).acc.mean()
            j = pd.concat([a.rename("ssl"), b.rename("ref")], axis=1).dropna()
            r = _wilcoxon((j.ssl - j.ref).values)
            linhas.append(dict(metodo=LAB[m], encoder=e, **r))
    t = pd.DataFrame(linhas)
    t["p_bonf"] = np.minimum(t.p * len(t), 1.0)            # família = os 8 pares
    t["sig"] = np.where(t.p_bonf.isna(), "—",
                        np.where(t.p_bonf < ALFA,
                                 np.where(t.mediana > 0, "SSL vence", "SSL perde"),
                                 "n.s."))
    return t


def p_minimo(n: int) -> float:
    """Menor `p` bilateral alcançável com `n` pares — o piso de poder do teste.

    O Wilcoxon é de postos: o melhor caso possível é *todas* as diferenças com o
    mesmo sinal, e aí `p = 2/2**n`. Com n = 5 esse piso é 0,0625 — **acima de
    0,05**. Ou seja: com 5 configurações, nenhum resultado, por mais limpo que
    seja, pode ser declarado significante. Não é falta de efeito, é falta de teste.
    """
    return min(2.0 / 2 ** n, 1.0) if n > 0 else 1.0


def testa_agregado(d: pd.DataFrame, ref: str = "sup", chave=("config",)) -> pd.DataFrame:
    """O teste do benchmark: um por método, empilhando os encoders.

    É a rota que multiplica o `n` sem custar seed — e é literalmente a que sustenta
    o "SSL significantly outperform supervised baselines" da literatura, que agrega
    sobre dataset × regime × encoder.
    """
    idx = list(chave) + ["encoder"]
    linhas = []
    for m in METODOS:
        a = d[(d.method == m)].groupby(idx).acc.mean()
        b = d[(d.method == ref)].groupby(idx).acc.mean()
        j = pd.concat([a.rename("ssl"), b.rename("ref")], axis=1).dropna()
        linhas.append(dict(metodo=LAB[m], encoder="TODOS", **_wilcoxon((j.ssl - j.ref).values)))
    t = pd.DataFrame(linhas)
    t["p_bonf"] = np.minimum(t.p * len(t), 1.0)
    t["sig"] = np.where(t.p_bonf < ALFA,
                        np.where(t.mediana > 0, "SSL vence", "SSL perde"), "n.s.")
    return t


def _fmt(t: pd.DataFrame) -> str:
    v = t.copy()
    for c in ["mediana", "rbc"]:
        if c in v:
            v[c] = v[c].map(lambda x: f"{x:+.2f}")
    for c in ["p", "p_bonf"]:
        if c in v:
            v[c] = v[c].map(lambda x: "—" if pd.isna(x) else
                            (f"{x:.1e}" if x < 1e-3 else f"{x:.4f}"))
    v = v.rename(columns={"mediana": "Δ mediano (pp)", "pos": "Δ>0", "n": "pares"})
    return v.to_string(index=False)


# --------------------------------------------------------------------------- #
# blocos                                                                       #
# --------------------------------------------------------------------------- #
def bloco_a(dc: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("A. RÉPLICA CENTRALIZADA vs TABELA 10 DO BENCHMARK (p. 18)")
    print("=" * 78)
    print("Média sobre os 6 datasets in-domain × 4 seeds (deles: × 3 seeds).\n")
    linhas = []
    for (m, e), reg in BENCH_T10.items():
        for k, bench in reg.items():
            c = dc[(dc.method == m) & (dc.encoder == e) & (dc.n_shots == k)]
            linhas.append(dict(metodo=LAB[m], encoder=e, shots=k, n=len(c),
                               nosso=c.acc.mean(), bench=bench,
                               dif=c.acc.mean() - bench))
    t = pd.DataFrame(linhas)
    piv = t.pivot_table(index=["metodo", "encoder"], columns="shots", values="dif")
    print(piv[["1", "10", "100", "full"]].round(1).to_string())
    print(f"\nMAE global {t.dif.abs().mean():.2f} pp · viés {t.dif.mean():+.2f} pp · "
          f"maior desvio {t.dif.abs().max():.1f} pp · {len(t)} células")
    print("Por método:  " + " · ".join(
        f"{LAB[m]} MAE {t[t.metodo == LAB[m]].dif.abs().mean():.2f}" for m in
        ["sup", "lfr", "tfc"]))
    return t


def bloco_a2(dc: pd.DataFrame, tb: pd.DataFrame) -> None:
    """O Δ (SSL − SL) por par: o deles, calculado da Tabela 10, contra o nosso.

    A réplica bater em NÍVEL (bloco A) não é o mesmo que bater em Δ — o Δ é a
    diferença que o claim usa, e ele pode divergir mesmo com os níveis próximos.
    Aqui o eixo é o que interessa: quem ganha do supervisionado, e em que ordem.
    """
    print("\n" + "=" * 78)
    print("A2. O Δ (SSL − supervisionado) POR PAR: benchmark vs nosso")
    print("=" * 78)
    linhas = []
    for m in METODOS:
        for e in ENCODERS:
            dif = [BENCH_T10[(m, e)][k] - BENCH_T10[("sup", e)][k]
                   for k in ["1", "10", "100", "full"]]
            nosso = tb[(tb.metodo == LAB[m]) & (tb.encoder == e)].mediana.iloc[0]
            linhas.append(dict(metodo=LAB[m], encoder=e,
                               bench_mediana=float(np.median(dif)),
                               bench_min=min(dif), bench_max=max(dif),
                               nosso_mediana=nosso))
    t = pd.DataFrame(linhas).sort_values("bench_mediana", ascending=False)
    t["sinal"] = np.where(np.sign(t.bench_mediana) == np.sign(t.nosso_mediana),
                          "concorda", "DIVERGE")
    v = t.copy()
    for c in ["bench_mediana", "bench_min", "bench_max", "nosso_mediana"]:
        v[c] = v[c].map(lambda x: f"{x:+.2f}")
    print("Δ do benchmark = mediana sobre os 4 regimes da Tab. 10 (6 datasets já "
          "mediados);\nΔ nosso = mediana sobre as 24 configurações (dataset × regime).\n")
    print(v.to_string(index=False))
    rho = t[["bench_mediana", "nosso_mediana"]].corr(method="spearman").iloc[0, 1]
    print(f"\nConcordância de sinal: {(t.sinal == 'concorda').sum()}/8 · "
          f"Spearman entre as ordens: ρ = {rho:+.2f}")


def bloco_b(dc: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("B. WILCOXON CENTRALIZADO — SSL vs supervisionado, por par")
    print("=" * 78)
    t = testa_pares(dc)
    print(f"\nPrimário: pareado sobre configuração (dataset × regime), seeds mediadas.")
    print(f"n = 24 pares por par · piso de p = {p_minimo(24):.1e} (folgado)")
    print(_fmt(t))
    print("\nAgregado por método (empilha os 4 encoders) — o teste do benchmark:")
    print(_fmt(testa_agregado(dc)))
    d2 = testa_pares(dc, por_seed=True)
    print(f"\nDiagnóstico: cada seed como observação (pseudo-réplica; n = 4×).")
    print(_fmt(d2[["metodo", "encoder", "n", "mediana", "p_bonf", "sig"]]))
    return t


def bloco_c(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("C. WILCOXON FEDERADO — SSL vs FedAvg supervisionado, por par")
    print("=" * 78)
    t = testa_pares(df)
    print(f"\nPrimário: pareado sobre configuração (spec × alvo × regime), "
          f"seeds mediadas.")
    print(f"n = 30 pares por par · piso de p = {p_minimo(30):.1e} (folgado)")
    print(_fmt(t))
    print("\nAgregado por método (empilha os 4 encoders):")
    print(_fmt(testa_agregado(df)))
    d2 = testa_pares(df, por_seed=True)
    print(f"\nDiagnóstico: cada seed como observação.")
    print(_fmt(d2[["metodo", "encoder", "n", "mediana", "p_bonf", "sig"]]))
    return t


def bloco_d(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("D. POR FEDERAÇÃO — o recorte das duas populações mistas")
    print("=" * 78)
    print("Bonferroni recalculado DENTRO de cada spec (8 pares por federação).")
    saida = []
    for spec in sorted(df.spec.unique()):
        s = df[df.spec == spec]
        t = testa_pares(s)
        t.insert(0, "spec", spec)
        n = int(t.n.max())
        piso, piso_b = p_minimo(n), min(p_minimo(n) * 8, 1.0)
        print(f"\n--- {spec} ---")
        print(f"n = {n} configurações · piso de p = {piso:.4f} · "
              f"após Bonferroni = {piso_b:.4f} "
              f"→ {'sobrevive' if piso_b < ALFA else 'NENHUM resultado pode ser significante'}")
        print(_fmt(t.drop(columns="spec")))
        ag = testa_agregado(s)
        print("  agregado por método (n = 4×):")
        print("  " + _fmt(ag).replace("\n", "\n  "))
        saida.append(t)
    return pd.concat(saida, ignore_index=True)


def bloco_e(tb: pd.DataFrame, tc: pd.DataFrame) -> None:
    """O ranking de pares sobrevive à federação?"""
    print("\n" + "=" * 78)
    print("E. CENTRALIZADO vs FEDERADO — o mesmo par ganha nos dois?")
    print("=" * 78)
    j = tb.merge(tc, on=["metodo", "encoder"], suffixes=("_cent", "_fed"))
    j["concorda"] = np.where(np.sign(j.mediana_cent) == np.sign(j.mediana_fed),
                             "sim", "NÃO")
    v = j[["metodo", "encoder", "mediana_cent", "sig_cent",
           "mediana_fed", "sig_fed", "concorda"]].copy()
    for c in ["mediana_cent", "mediana_fed"]:
        v[c] = v[c].map(lambda x: f"{x:+.2f}")
    print(v.to_string(index=False))
    rho = j[["mediana_cent", "mediana_fed"]].corr(method="spearman").iloc[0, 1]
    print(f"\nSpearman entre os Δ medianos dos 8 pares: ρ = {rho:+.2f}")
    print(f"Concordância de sinal: {(j.concorda == 'sim').sum()}/8")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", help="se dado, grava os CSVs das tabelas")
    a = ap.parse_args()

    dc, df = carrega_centralizado(), carrega_federado()
    print(f"centralizado: {len(dc)} células · federado: {len(df)} células")

    ta = bloco_a(dc)
    tb = bloco_b(dc)
    bloco_a2(dc, tb)
    tc = bloco_c(df)
    td = bloco_d(df)
    bloco_e(tb, tc)

    if a.outdir:
        out = Path(a.outdir)
        out.mkdir(parents=True, exist_ok=True)
        for nome, t in [("replica_vs_tabela10", ta), ("wilcoxon_centralizado", tb),
                        ("wilcoxon_federado", tc), ("wilcoxon_por_spec", td)]:
            t.to_csv(out / f"{nome}.csv", index=False)
        print(f"\n[OK] 4 CSVs em {out}")


if __name__ == "__main__":
    main()
