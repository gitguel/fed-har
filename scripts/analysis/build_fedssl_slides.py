#!/usr/bin/env python3
"""Assets da seção de resultados do Fed-SSL cross-device, em PDF vetorial.

Irmão do `build_presentation_assets.py` (aquele cobre o SSL **centralizado**;
este, a grade **federada cross-device**). Como ele: lê apenas os caches de
`results/`, nunca treina, e escreve só no diretório passado em `--outdir` — que
é obrigatório justamente para não sobrescrever registro de apresentação já
entregue.

As cinco figuras respondem, nesta ordem, às quatro hipóteses da seção:

  1. `fig1_custo_domain_shift`  — misturar domínio prejudica o SL federado?
     **A resposta depende da comparação**, e a figura mostra as duas:
       · SUBSTITUIR metade dos clientes por estrangeiros (cross5+5 vs in-domain,
         10 clientes nos dois lados) — custa, e o custo CRESCE com o rótulo;
       · ACRESCENTAR um segundo domínio (cross10+10 vs in-domain, 20 contra 10) —
         é praticamente de graça, e no few-shot AJUDA.
     Só a primeira sustenta "cross-domain prejudica". Ver `docs/mapa_experimentos.md`
     §10.4: o Δ do cross10+10 é "acrescentar um domínio", não "dobrar o dado".

  2. `fig2_ssl_no_cross`        — dentro das federações mistas, quais pares batem o SL.
  3. `fig3_placar_por_par`      — o placar completo: quem ajuda, quem atrapalha, quem empata.
  4. `fig4_replica_benchmark`   — a réplica do Δ do benchmark (direito de invocá-lo).
  5. `fig5_piso_de_poder`       — por que "não deu significante" às vezes é sobre o desenho.

Uso:
    poetry run python scripts/analysis/build_fedssl_slides.py \\
        --outdir docs/apresentacao_11_08
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
RESULTS = PROJECT_ROOT / "results"
W = RESULTS / "derived" / "wilcoxon"

from analysis.wilcoxon_pares import BENCH_T10, LAB  # noqa: E402

# ── paleta ────────────────────────────────────────────────────────────────────
# Cor segue a ENTIDADE, nunca o veredito nem a posição. `tfc` é laranja e `lfr` é
# verde em toda figura do projeto (mesma paleta dos notebooks). Os dois ALVOS
# (fig. 1) ganham roxo/vermelho — hues que não colidem com nenhum método, para o
# leitor não ler "TF-C" onde está escrito "RealWorld".
# Validadas com os seis checks do dataviz, `pairs=all`, superfície #fcfcfb:
#   tfc/lfr    → PASS (normal ΔE 27.6 · pior CVD 9.2 deutan); o verde fica em
#                2.74:1 de contraste, abaixo de 3:1 → OBRIGA rótulo direto + tabela.
#   RW/MS      → PASS sem nenhum WARN (normal ΔE 33.6 · pior CVD 22.7 protan;
#                contrastes 8.33:1 e 3.85:1).
CM = {"TF-C": "#eb6834", "LFR": "#1baf7a"}
CT = {"RealWorld_thigh": "#4a3aa7", "MotionSense": "#e34948"}
INK, INK2, SURF, GRID = "#1a1a19", "#5c5b55", "#fcfcfb", "#c9c8c2"

SHOTS = ["1", "2", "5", "10", "full"]
SHOTS_LAB = ["1", "2", "5", "10", "full\n(32)"]
ALFA, FAMILIA = 0.05, 8
ALVO_LAB = {"RealWorld_thigh": "alvo: RealWorld_thigh", "MotionSense": "alvo: MotionSense"}

OUT: Path = None


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 9,
        "axes.edgecolor": "#d8d7d2", "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
    })


def salva(fig, nome):
    p = OUT / f"{nome}.pdf"
    fig.savefig(p, bbox_inches="tight")           # PDF vetorial: escala no projetor
    fig.savefig(OUT / f"{nome}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p.relative_to(PROJECT_ROOT)}")


def p_minimo(n):
    return min(2.0 / 2 ** n, 1.0) if n > 0 else 1.0


# ── carga ─────────────────────────────────────────────────────────────────────
def carrega_wilcoxon():
    d = {n: pd.read_csv(W / f"wilcoxon_{n}.csv") for n in ("centralizado", "federado")}
    d["spec"] = pd.read_csv(W / "wilcoxon_por_spec.csv")
    for v in d.values():
        v["par"] = v.metodo + " + " + v.encoder
    return d


def carrega_sl_shift():
    """Δ do supervisionado federado ao misturar domínio, pareado na mesma célula.

    Duas comparações distintas, e confundi-las é o erro fácil desta seção:
      `sub` = cross5+5 − in-domain  → SUBSTITUIÇÃO a 10 clientes fixos;
      `add` = cross10+10 − in-domain → ADIÇÃO de um segundo domínio (20 clientes).
    """
    d = pd.read_csv(RESULTS / "derived" / "fedssl_selected.csv")
    d = d[(d.protocol == "val") & (d.method == "none")].copy()
    d["n_shots"] = d.n_shots.astype(str)
    d["acc"] = d.test_acc * 100
    d["arm"] = d.spec.map({
        "device:RealWorld_thigh:10": "in", "device:MotionSense:10": "in",
        "device:RealWorld_thigh+MotionSense:5+5": "sub",
        "device:RealWorld_thigh+MotionSense:10+10": "add"})
    p = d.pivot_table(index=["target", "n_shots", "encoder", "seed"],
                      columns="arm", values="acc").reset_index()
    p["sub"] = p["sub"] - p["in"]
    p["add"] = p["add"] - p["in"]
    return p


# ── figura 1 ──────────────────────────────────────────────────────────────────
def fig1_custo_domain_shift(p):
    painel = [("sub", "SUBSTITUIR metade dos clientes por estrangeiros",
               "cross5+5  vs  in-domain   ·   10 clientes nos dois lados"),
              ("add", "ACRESCENTAR um segundo domínio",
               "cross10+10  vs  in-domain   ·   20 clientes contra 10")]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharey=True)
    for ax, (col, tit, sub) in zip(axes, painel):
        ax.axhline(0, color=INK2, lw=1.2, zorder=2)
        for alvo, c in CT.items():
            q = p[p.target == alvo]
            # pontos translucidos = media por encoder (mostra que o sinal nao vem
            # de um encoder so); linha = media sobre encoders e seeds.
            porenc = q.groupby(["n_shots", "encoder"])[col].mean()
            for (s, _), v in porenc.items():
                ax.plot(SHOTS.index(s), v, "o", ms=4, color=c, alpha=.3, zorder=3)
            m = q.groupby("n_shots")[col].mean().reindex(SHOTS)
            ax.plot(range(len(SHOTS)), m.values, "-o", lw=2.2, ms=8, color=c,
                    mec=SURF, mew=1.5, zorder=4, label=ALVO_LAB[alvo])
            # Rotulo direto em todo ponto, deslocado por SERIE (roxo sempre acima,
            # vermelho sempre abaixo): as duas linhas chegam a 0,5 pp uma da outra
            # e o deslocamento por sinal do valor as faria colidir.
            dy = 11 if alvo == "RealWorld_thigh" else -18
            for i, v in enumerate(m.values):
                ax.annotate(f"{v:+.1f}", (i, v), xytext=(0, dy),
                            textcoords="offset points", ha="center", fontsize=7.5,
                            color=c, fontweight="bold")
        x = p.groupby(["target", "n_shots", "encoder"])[col].mean().dropna()
        st, pv = wilcoxon(x)
        ax.set_title(f"{tit}\n{sub}", loc="left", fontsize=10, color=INK, pad=8)
        ax.annotate(f"mediana {x.median():+.2f} pp  ·  n = {len(x)}  ·  "
                    f"p = {pv:.1e}" + ("" if pv < ALFA else "  (n.s.)"),
                    (.5, .965), xycoords="axes fraction", ha="center", va="top",
                    fontsize=8.5, color=INK if pv < ALFA else INK2,
                    fontweight="bold" if pv < ALFA else "normal")
        ax.set_xticks(range(len(SHOTS)))
        ax.set_xticklabels(SHOTS_LAB)
        ax.set_xlabel("rótulos por classe, por cliente")
        ax.set_xlim(-.4, len(SHOTS) - .6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Δ acurácia vs federação in-domain (pp)")
    axes[0].set_ylim(-13.5, 6.2)          # folga para a anotação no topo
    fig.legend(handles=[Line2D([], [], marker="o", ls="-", lw=2.2, ms=8, color=c,
                               mec=SURF, mew=1.5, label=ALVO_LAB[a])
                        for a, c in CT.items()],
               loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(.5, -.06))
    fig.suptitle("O que o domain shift custa ao supervisionado federado — e por que a resposta "
                 "depende da pergunta", x=.008, ha="left", fontsize=12, y=1.02)
    fig.tight_layout()
    salva(fig, "fig1_custo_domain_shift")


# ── lollipop compartilhado (figs 2 e 3) ───────────────────────────────────────
def lollipop(ax, d, ordem, titulo, sig_label=True, ms=9, fs=8):
    ax.axvline(0, color=GRID, lw=1, zorder=1)
    for _, r in d.iterrows():
        y = ordem.index(r.par)
        c, vence = CM[r.metodo], r.sig == "SSL vence"
        ax.plot([0, r.mediana], [y, y], color=c, lw=2, alpha=.55, zorder=2,
                solid_capstyle="round")
        ax.plot(r.mediana, y, "o", ms=ms, zorder=3, color=c if vence else SURF,
                mec=c, mew=2)
        # Rotulo direto sempre a DIREITA: o verde do LFR nao alcanca 3:1 contra o
        # fundo, e a esquerda o texto colide com o rotulo do eixo y.
        txt = f"{r.mediana:+.1f}  {int(r.pos)}/{int(r.n)}" if sig_label else f"{r.mediana:+.1f}"
        ax.annotate(txt, (r.mediana, y), xytext=(0.8, 0), textcoords="offset fontsize",
                    va="center", ha="left", fontsize=fs,
                    color=INK if vence else INK2,
                    fontweight="bold" if vence else "normal")
    ax.set_title(titulo, loc="left", fontsize=10, color=INK, pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.set_ylim(-.7, len(ordem) - .3)


def legenda():
    return [Line2D([], [], marker="o", ls="", ms=9, color=CM["TF-C"], label="TF-C"),
            Line2D([], [], marker="o", ls="", ms=9, color=CM["LFR"], label="LFR"),
            Line2D([], [], marker="o", ls="", ms=9, color=SURF, mec=INK2, mew=2,
                   label="n.s. — indistinguível do supervisionado")]


# ── figura 2 ──────────────────────────────────────────────────────────────────
def fig2_ssl_no_cross(ds, ordem):
    specs = [("device:RealWorld_thigh+MotionSense:5+5", "cross5+5"),
             ("device:RealWorld_thigh+MotionSense:10+10", "cross10+10")]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), sharex=True, sharey=True)
    for ax, (spec, rot) in zip(axes, specs):
        d = ds[ds.spec == spec]
        nn = int(d.n.max())
        # A nota sobre o piso vale para os dois paineis e vai no nivel da figura:
        # no titulo do painel ela transborda para o vizinho.
        lollipop(ax, d, ordem, f"{rot}\nn = {nn} configurações · piso de p = "
                 f"{p_minimo(nn):.4f}", sig_label=True, fs=7.5)
        ax.set_xlabel("Δ mediano — SSL − FedAvg supervisionado (pp)")
    axes[0].set_yticks(range(len(ordem)))
    axes[0].set_yticklabels(ordem)
    axes[0].set_xlim(-3, 33)
    fig.legend(handles=legenda(), loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(.5, -.05))
    fig.suptitle("Dentro das federações mistas, o SSL recupera parte do que o shift custou\n"
                 "com n = 10, Bonferroni ×8 deixa o piso em 0,0156 — só um placar perfeito "
                 "de 10/10 sobrevive; não há meio-termo nessa escala",
                 x=.008, ha="left", fontsize=12, y=1.06)
    fig.tight_layout()
    salva(fig, "fig2_ssl_no_cross")


# ── figura 3 ──────────────────────────────────────────────────────────────────
def fig3_placar_por_par(dc, df, ordem):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharex=True, sharey=True)
    lollipop(axes[0], dc, ordem, "Centralizado  ·  n = 24 (dataset × regime)")
    lollipop(axes[1], df, ordem, "Federado  ·  n = 30 (spec × alvo × regime)")
    for ax in axes:
        ax.set_xlabel("Δ mediano — SSL − supervisionado (pp)")
    axes[0].set_yticks(range(len(ordem)))
    axes[0].set_yticklabels(ordem)
    axes[0].set_xlim(-3, 30)
    fig.legend(handles=legenda(), loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(.5, -.03))
    fig.suptitle("O placar por par — preenchido vence após Bonferroni ×8; o número ao lado "
                 "é quantas configurações ficaram positivas",
                 x=.008, ha="left", fontsize=12, y=1.01)
    fig.tight_layout()
    salva(fig, "fig3_placar_por_par")


# ── figura 4 ──────────────────────────────────────────────────────────────────
def fig4_replica_benchmark(dc, ordem):
    regimes = ["1", "10", "100", "full"]
    bench = {(LAB[m], e): float(np.median([BENCH_T10[(m, e)][k] - BENCH_T10[("sup", e)][k]
                                           for k in regimes]))
             for m in ["lfr", "tfc"] for e in ["resnetse5", "cnnpff", "rnn", "tstcc"]}
    gray = "#9a9992"
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.axvline(0, color=GRID, lw=1, zorder=1)
    for _, r in dc.iterrows():
        y = ordem.index(r.par)
        b, c = bench[(r.metodo, r.encoder)], CM[r.metodo]
        ax.plot([b, r.mediana], [y, y], color=c, lw=1.6, alpha=.45, zorder=2,
                solid_capstyle="round")
        ax.plot(b, y, "D", ms=7, color=SURF, mec=gray, mew=1.8, zorder=3)
        ax.plot(r.mediana, y, "o", ms=9, color=c, mec=SURF, mew=1.5, zorder=4)
        ax.annotate(f"{b:+.1f} → {r.mediana:+.1f}", (max(b, r.mediana), y),
                    xytext=(0.8, 0), textcoords="offset fontsize", va="center",
                    ha="left", fontsize=8, color=INK2)
    b = np.array([bench[(r.metodo, r.encoder)] for _, r in dc.iterrows()])
    rho = pd.Series(b).corr(dc.mediana.reset_index(drop=True), method="spearman")
    concorda = int((np.sign(b) == np.sign(dc.mediana)).sum())
    ax.set_yticks(range(len(ordem)))
    ax.set_yticklabels(ordem)
    ax.set_xlim(-3.5, 21)
    ax.set_ylim(-.7, len(ordem) - .3)
    ax.set_xlabel("Δ  —  SSL − supervisionado (pp)")
    ax.set_title("A réplica bate na ESTRUTURA do efeito, não só no nível\n"
                 f"concordância de sinal {concorda}/8  ·  Spearman entre as ordens ρ = {rho:+.2f}",
                 loc="left", fontsize=11, color=INK, pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(handles=[
        Line2D([], [], marker="D", ls="", ms=7, color=SURF, mec=gray, mew=1.8,
               label="benchmark (da Luz, Tab. 10)"),
        Line2D([], [], marker="o", ls="", ms=9, color=CM["TF-C"], mec=SURF, label="nosso — TF-C"),
        Line2D([], [], marker="o", ls="", ms=9, color=CM["LFR"], mec=SURF, label="nosso — LFR")],
        frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    salva(fig, "fig4_replica_benchmark")


# ── figura 5 ──────────────────────────────────────────────────────────────────
def fig5_piso_de_poder():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    n = np.arange(3, 33)
    piso = 2.0 / 2 ** n
    ax.semilogy(n, piso, color=INK2, lw=2, zorder=3)
    ax.axhline(ALFA, color=GRID, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.axhline(ALFA / FAMILIA, color=GRID, lw=1.4, ls=(0, (1, 2)), zorder=2)
    ax.annotate("α = 0,05", (32, ALFA), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK2, va="bottom", ha="right")
    ax.annotate("α/8 (Bonferroni)", (32, ALFA / FAMILIA), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=INK2, va="bottom", ha="right")
    # Tinta neutra de proposito: aqui a distincao e de STATUS (testavel ou nao) e
    # reusar o laranja/verde faria ler "metodo" onde esta escrito "poder".
    ax.fill_between(n, piso, 1, where=(piso > ALFA / FAMILIA), color=GRID, alpha=.28,
                    zorder=1)
    for nn, rot in [(5, "federação única"), (10, "cross5+5 e 10+10"),
                    (24, "centralizado"), (30, "federado")]:
        p = p_minimo(nn)
        ok = p < ALFA / FAMILIA
        ax.plot(nn, p, "o", ms=9, color=INK if ok else SURF, mec=INK, mew=2, zorder=4)
        ax.annotate(f"n={nn}\n{rot}", (nn, p), xytext=(0, 12 if not ok else -28),
                    textcoords="offset points", ha="center", fontsize=8, color=INK,
                    fontweight="bold")
    ax.set_xlabel("n = número de configurações pareadas")
    ax.set_ylabel("menor p bilateral alcançável")
    ax.set_title("O piso de poder do Wilcoxon: p mínimo = 2/2ⁿ\n"
                 "na faixa sombreada, nenhum resultado pode ser significante — "
                 "por mais limpo que seja",
                 loc="left", fontsize=11, color=INK)
    ax.set_ylim(1e-10, 3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    salva(fig, "fig5_piso_de_poder")


# ── figura 6 (Exp. 2) ─────────────────────────────────────────────────────────
PART_DIR = RESULTS / "fedssl_cross_device_parts"
XPART_DIR = RESULTS / "fedssl_crossspec_parts"
KEY_E2 = ["method", "encoder", "spec", "seed", "n_shots"]
SIMPLE = ["device:RealWorld_thigh:10", "device:MotionSense:10"]
PLATEAU = 20


def _le_parts(paths):
    d = pd.concat([pd.read_csv(f) for p in paths for f in sorted(p.glob("*.csv"))],
                  ignore_index=True)
    d["n_shots"] = d.n_shots.astype(str)
    return d


def _plato(d):
    """Média das PLATEAU últimas rodadas — a convenção do repo para estas grades.

    Não é `argmax(val)`: o §3 do notebook cross-device mostrou que o argmax infla de
    forma desigual entre braços.
    """
    r = d["round"].max()
    return d[d["round"] > r - PLATEAU].groupby(KEY_E2 + ["target"])["test_acc"].mean() * 100


def carrega_exp2():
    """Δ = acc(pré-treino em 10 clientes) − acc(pré-treino em 1 cliente).

    ⚠️ O controle é `pre-single-*-10` (mesmas 1.920 janelas, dos mesmos 10 usuários,
    num cliente só) — **não** `pre-single-*-10-full`, que é outro controle (cliente
    único com o dado TODO, sem o orçamento de 192). O glob `pre-single-*` pega os
    dois e, como `pretrain_spec` não entra na chave de agrupamento, mediá-los
    silenciosamente funde duas perguntas diferentes.
    """
    raw = _le_parts([PART_DIR])
    raw = raw[raw.method.isin(["lfr", "tfc"]) & raw.spec.isin(SIMPLE)]
    cen = _le_parts(list(XPART_DIR.glob("pre-single-*-10")))
    e = pd.concat([_plato(raw).rename("fed"), _plato(cen).rename("cen")], axis=1).dropna()
    e["d"] = e.fed - e.cen
    cel = e.reset_index()
    # Unidade de pareamento do teste: a configuração (spec × alvo × degrau), seeds
    # mediadas — a mesma do `wilcoxon_pares.py`.
    cfg = (cel.groupby(["method", "encoder", "spec", "target", "n_shots"])["d"]
              .mean().reset_index())
    return cel, cfg


def fig6_custo_de_federar_pretreino(cel, cfg):
    """O eixo é o REGIME DE RÓTULOS, não um agregado por método.

    Motivo: os dois métodos pagam em regimes **opostos** — o TF-C paga quase tudo no
    1-shot e zera no `full`; o LFR não paga nada no few-shot e paga ~1,5 pp do
    5-shot em diante. Um número agregado por método media uma inversão, e é por isso
    que o agregado troca de sinal relativo conforme a métrica e a regra de rodada.
    O padrão por regime, esse, é estável nas quatro combinações testadas
    (acurácia/F1 × tail20/argmax-val).
    """
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.axhline(0, color=INK2, lw=1.2, zorder=2)
    med = {m: cel[cel.method == m].groupby("n_shots")["d"].mean().reindex(SHOTS)
           for m in ["lfr", "tfc"]}
    for m in ["lfr", "tfc"]:
        c = CM[LAB[m]]
        g = cel[cel.method == m]
        for (s, _), v in g.groupby(["n_shots", "encoder"])["d"].mean().items():
            ax.plot(SHOTS.index(s), v, "o", ms=4, color=c, alpha=.28, zorder=3)
        ax.plot(range(len(SHOTS)), med[m].values, "-o", lw=2.4, ms=8, color=c,
                mec=SURF, mew=1.5, zorder=4, label=LAB[m])
        # As duas linhas se CRUZAM, entao deslocamento fixo por serie colide. A regra
        # e "para fora": em cada x, quem esta por cima rotula acima, quem esta por
        # baixo rotula abaixo.
        outro = "tfc" if m == "lfr" else "lfr"
        for i, v in enumerate(med[m].values):
            acima = v >= med[outro].values[i]
            ax.annotate(f"{v:+.1f}", (i, v), xytext=(0, 12 if acima else -19),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=c, fontweight="bold")
    ax.set_xticks(range(len(SHOTS)))
    ax.set_xticklabels(SHOTS_LAB)
    ax.set_xlim(-.4, len(SHOTS) - .6)
    ax.set_xlabel("rótulos por classe, por cliente")
    ax.set_ylabel("Δ acurácia: 10 clientes − 1 cliente (pp)")
    ax.annotate("federar custa ↓", (.015, .06), xycoords="axes fraction", ha="left",
                fontsize=9, color=INK2, style="italic")

    todo = cfg.d.values
    # Abaixo do xlabel, nao em cima dele.
    fig.text(.5, -.055, f"custo total, sobre as 80 configurações:  {np.median(todo):+.2f} pp  "
             f"(p = {wilcoxon(todo)[1]:.1e})", ha="center", fontsize=9, color=INK)
    fig.text(.5, -.115, "os dois métodos pagam em regimes OPOSTOS — por isso um agregado único "
             "por método não é estável; o padrão por regime é",
             ha="center", fontsize=8, color=INK2, style="italic")
    ax.set_title("Federar o pré-treino custa ~1 pp — mas cada método paga em um regime diferente\n"
                 "controle exato: as mesmas 1.920 janelas, dos mesmos 10 usuários, num cliente "
                 "só; o fine-tuning é idêntico nos dois braços",
                 loc="left", fontsize=11, color=INK, pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.tight_layout()
    salva(fig, "fig6_custo_de_federar_pretreino")

    linhas = []
    for (m, enc), g in cfg.groupby(["method", "encoder"]):
        x = g.d.values
        _, p = wilcoxon(x)
        linhas.append(dict(par=f"{LAB[m]} + {enc}", n=len(x), mediana=float(np.median(x)),
                           neg=int((x < 0).sum()), p_bonf=min(p * FAMILIA, 1.0)))
    return pd.DataFrame(linhas)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True,
                    help="pasta de destino (ex.: docs/apresentacao_11_08). "
                         "Obrigatório: nunca sobrescreve registro de apresentação entregue.")
    args = ap.parse_args()
    OUT = Path(args.outdir)
    if not OUT.is_absolute():
        OUT = PROJECT_ROOT / OUT
    OUT.mkdir(parents=True, exist_ok=True)
    _style()

    w = carrega_wilcoxon()
    dc, df, ds = w["centralizado"], w["federado"], w["spec"]
    ordem = dc.sort_values("mediana").par.tolist()   # uma ordem de linhas para todas

    print(f"Escrevendo em {OUT.relative_to(PROJECT_ROOT)}/")
    fig1_custo_domain_shift(carrega_sl_shift())
    fig2_ssl_no_cross(ds, ordem)
    fig3_placar_por_par(dc, df, ordem)
    fig4_replica_benchmark(dc, ordem)
    fig5_piso_de_poder()
    t = fig6_custo_de_federar_pretreino(*carrega_exp2())
    print("\nExp. 2 — custo de federar o pré-treino (tabela de apoio):")
    print(t.sort_values("mediana").to_string(index=False))
    print("pronto.")


if __name__ == "__main__":
    main()
