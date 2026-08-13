#!/usr/bin/env python3
"""Anexos da seção de resultados — o que as figuras 1–6 agregam, destrinchado.

Cada figura principal (`build_fedssl_slides.py`) responde a uma hipótese com um
número agregado. Estes seis anexos abrem esse agregado nas dimensões que ele
esconde — **encoder**, **domínio/alvo** e **regime de rótulos** — para o material
de apoio no fim da apresentação: quando perguntarem *"qual encoder é o pior?"* ou
*"isso vale nos dois alvos?"*, a resposta é um slide, não uma promessa.

  A ← fig. 1  custo do domain shift **por encoder**, em cada alvo.
  B ← figs. 2 e 3  o placar por par **por federação × alvo** (as 6 colunas que a
      fig. 3 funde em "n = 30").
  C ← figs. 2 e 3  o placar por par **por regime de rótulos**.
  D ← fig. 3  o placar por par **por dataset** (centralizado, 6 domínios).
  E ← fig. 6  custo de federar o pré-treino **por encoder**, em cada método.
  F ← fig. 4  a réplica do benchmark **regime a regime** (a fig. 4 usa a mediana
      sobre os 4 regimes).

⚠️ **Leia as células como descritivas, não como testes.** Uma célula destes anexos
tem n = 4, 5 ou 6 configurações pareadas; o piso do Wilcoxon (2/2ⁿ) fica em 0,125,
0,0625 e 0,031 — acima de α/8 em todos os casos. É exatamente o argumento da fig. 5:
aqui não há teste possível, só descrição. Os testes moram nas figuras principais,
onde o pareamento agrega essas células.

Uso:
    poetry run python scripts/analysis/build_fedssl_anexos.py \\
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analysis.build_fedssl_slides as S  # noqa: E402
from analysis.build_fedssl_slides import (  # noqa: E402
    INK, INK2, SURF, GRID, SHOTS, SHOTS_LAB, _style, carrega_exp2, carrega_sl_shift,
    p_minimo,
)
from analysis.wilcoxon_pares import (  # noqa: E402
    BENCH_T10, DS6, ENCODERS, LAB, carrega_centralizado, carrega_federado,
)

# ── paleta ────────────────────────────────────────────────────────────────────
# Os anexos precisam de uma quarta escala que as figuras principais não têm: a
# identidade do ENCODER. Não dá para reusar laranja/verde (são TF-C/LFR em toda a
# apresentação) nem roxo/vermelho (são os dois alvos na fig. 1) — a cor segue a
# entidade, e reciclar hue faria ler "método" onde está escrito "encoder". Sobram
# quatro slots da paleta de referência, nesta ordem fixa:
#   blue · yellow · magenta · green
# Validados com os seis checks, `pairs=all` (são small multiples), superfície
# #fcfcfb: pior CVD ΔE 13,0 · pior normal ΔE 19,6 · ambos acima do alvo. O amarelo
# (2,11:1) e o magenta (2,62:1) ficam abaixo de 3:1 → **regra de alívio**: toda
# série leva rótulo direto na ponta, além da legenda.
ENC_COR = {"resnetse5": "#2a78d6", "cnnpff": "#eda100",
           "rnn": "#e87ba4", "tstcc": "#008300"}

# Divergente para os mapas de calor: azul ↔ vermelho, cinza neutro no meio (o par
# documentado; azul↔verde foi rejeitado porque os dois são frios e o meio não lê
# como "nada"). Cada braço é interpolado em OKLab do cinza até o polo — nenhuma cor
# fora da linha do polo documentado.
POLO_POS, POLO_NEG, NEUTRO = "#2a78d6", "#e34948", "#f0efec"
CEN_SHOTS = ["1", "10", "100", "full"]


# ── OKLab (só o suficiente para interpolar o divergente) ──────────────────────
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _srgb_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _oklab(hexa):
    rgb = _srgb_lin(np.array([int(hexa.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]))
    return _M2 @ np.cbrt(_M1 @ rgb)


def _to_rgb(lab):
    return np.clip(_lin_srgb(_M1i @ (_M2i @ lab) ** 3), 0, 1)


_M1i, _M2i = np.linalg.inv(_M1), np.linalg.inv(_M2)


def cmap_divergente(n=256):
    a, z, m = _oklab(POLO_NEG), _oklab(POLO_POS), _oklab(NEUTRO)
    meio = n // 2
    cores = ([_to_rgb(a + (m - a) * t) for t in np.linspace(0, 1, meio)]
             + [_to_rgb(m + (z - m) * t) for t in np.linspace(0, 1, n - meio)])
    return LinearSegmentedColormap.from_list("div_pp", cores)


CMAP = None


# ── mapa de calor compartilhado (anexos B, C, D) ──────────────────────────────
def escala(*Ms):
    """vmax SATURADO, não o máximo.

    Um único par (TF-C + rnn, até +28 pp) domina a amplitude e empurraria todo o
    resto para dentro de dois passos da rampa. Satura no percentil 75 de |Δ| — a
    variação que interessa mora em ±5 pp — e o valor exato continua escrito em cada
    célula, que é o que a regra de alívio pede quando a cor sozinha não separa.
    """
    v = np.concatenate([m[~np.isnan(m)].ravel() for m in Ms])
    return max(5.0, float(np.ceil(np.percentile(np.abs(v), 75))))


def heat(ax, M, linhas, colunas, vmax, contagem=None, fs=8):
    """M[linha][coluna] em pp. `contagem` = string extra sob o valor (ex.: 4/5)."""
    im = ax.imshow(M, cmap=CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            # Tinta do texto por LUMINÂNCIA da célula, nunca pela cor da série.
            r, g, b = CMAP((v + vmax) / (2 * vmax))[:3]
            tinta = "#ffffff" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.42 else INK
            t = f"{v:+.1f}" + (f"\n{contagem[i][j]}" if contagem else "")
            ax.text(j, i, t, ha="center", va="center", fontsize=fs, color=tinta,
                    fontweight="bold" if abs(v) >= 2 else "normal", linespacing=1.35)
    ax.set_xticks(range(len(colunas)))
    ax.set_xticklabels(colunas, fontsize=8.5)
    ax.set_yticks(range(len(linhas)))
    ax.set_yticklabels(linhas, fontsize=9)
    ax.set_xticks(np.arange(-.5, len(colunas), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(linhas), 1), minor=True)
    # Fio da superfície entre células: separa marcas adjacentes sem desenhar grade.
    ax.grid(which="minor", color=SURF, lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def barra(fig, im, ax, rotulo):
    cb = fig.colorbar(im, ax=ax, fraction=.02, pad=.015)
    cb.set_label(rotulo, fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)


def rotula_pontas(ax, series, x, folga):
    """Rótulo direto na ponta direita, com de-colisão vertical.

    Obrigatório aqui: amarelo e magenta ficam abaixo de 3:1 contra a superfície,
    e a regra de alívio exige que a identidade não dependa só da cor.
    """
    ordem = sorted(series.items(), key=lambda kv: kv[1][1])
    ys = []
    for nome, (cor, v) in ordem:
        y = v if not ys else max(v, ys[-1] + folga)
        ys.append(y)
        ax.annotate(f" {nome}", (x, y), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7.5, color=cor, fontweight="bold",
                    annotation_clip=False)


# ── anexo A ← fig. 1 ──────────────────────────────────────────────────────────
def anexoA_shift_por_encoder(p):
    cols = [("sub", "SUBSTITUIR  (cross5+5 − in-domain)"),
            ("add", "ACRESCENTAR  (cross10+10 − in-domain)")]
    alvos = ["RealWorld_thigh", "MotionSense"]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), sharex=True, sharey=True)
    for li, alvo in enumerate(alvos):
        for ci, (col, tit) in enumerate(cols):
            ax = axes[li][ci]
            ax.axhline(0, color=INK2, lw=1.1, zorder=2)
            pontas, med = {}, {}
            for enc in ENCODERS:
                c = ENC_COR[enc]
                m = (p[(p.target == alvo) & (p.encoder == enc)]
                     .groupby("n_shots")[col].mean().reindex(SHOTS))
                ax.plot(range(len(SHOTS)), m.values, "-o", lw=2, ms=6, color=c,
                        mec=SURF, mew=1.2, zorder=3)
                pontas[enc] = (c, float(m.values[-1]))
                med[enc] = float(np.median(m.values))
            rotula_pontas(ax, pontas, len(SHOTS) - 1, folga=1.5)
            pior, melhor = min(med, key=med.get), max(med, key=med.get)
            ax.annotate(f"pior: {pior} {med[pior]:+.1f} pp   ·   melhor: {melhor} {med[melhor]:+.1f} pp",
                        (.5, .03), xycoords="axes fraction", ha="center", fontsize=8.5,
                        color=INK, fontweight="bold")
            ax.set_title(f"{alvo}  ·  {tit}", loc="left", fontsize=9.5, color=INK, pad=6)
            ax.set_xticks(range(len(SHOTS)))
            ax.set_xticklabels(SHOTS_LAB)
            ax.set_xlim(-.35, len(SHOTS) - .55)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("rótulos por classe, por cliente")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ acurácia vs in-domain (pp)")
    axes[0][0].set_ylim(-17, 9)
    fig.legend(handles=[Line2D([], [], marker="o", ls="-", lw=2, ms=6, color=ENC_COR[e],
                               mec=SURF, mew=1.2, label=e) for e in ENCODERS],
               loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(.5, -.035))
    fig.suptitle("ANEXO A — o custo do domain shift, encoder a encoder  (destrincha a fig. 1)\n"
                 "a fig. 1 mostra a média sobre estes quatro; aqui está de quem vem o sinal",
                 x=.008, ha="left", fontsize=12, y=1.015)
    fig.tight_layout()
    S.salva(fig, "anexoA_shift_por_encoder")


# ── Δ por par, sem agregar ────────────────────────────────────────────────────
def delta_cen():
    d = carrega_centralizado()
    g = d.groupby(["method", "encoder", "target", "n_shots"]).acc.mean().unstack("method")
    return pd.concat([(g[m] - g["sup"]).rename("d").reset_index().assign(metodo=LAB[m])
                      for m in ("lfr", "tfc")], ignore_index=True)


def delta_fed():
    d = carrega_federado()
    g = (d.groupby(["method", "encoder", "spec", "target", "n_shots"]).acc.mean()
          .unstack("method"))
    return pd.concat([(g[m] - g["sup"]).rename("d").reset_index().assign(metodo=LAB[m])
                      for m in ("lfr", "tfc")], ignore_index=True)


def ordena(dc):
    """Uma ordem de linhas para todos os anexos — a mesma das figs. 2–4."""
    o = (dc.groupby(["metodo", "encoder"]).d.median().sort_values())
    return [f"{m} + {e}" for m, e in o.index]


def matriz(d, ordem, colunas, chave):
    M = np.full((len(ordem), len(colunas)), np.nan)
    C = [["" for _ in colunas] for _ in ordem]
    for (met, enc, k), g in d.groupby(["metodo", "encoder", chave]):
        par, col = f"{met} + {enc}", k
        if col not in colunas:
            continue
        i, j = ordem.index(par), colunas.index(col)
        M[i, j] = g.d.median()
        C[i][j] = f"{int((g.d > 0).sum())}/{len(g)}"
    return M, C


# ── anexo B ← figs. 2 e 3 ─────────────────────────────────────────────────────
def anexoB_por_federacao_e_alvo(df, ordem):
    df = df.copy()
    df["cel"] = df.spec + "|" + df.target
    colunas = ["device:RealWorld_thigh:10|RealWorld_thigh",
               "device:MotionSense:10|MotionSense",
               "device:RealWorld_thigh+MotionSense:5+5|RealWorld_thigh",
               "device:RealWorld_thigh+MotionSense:5+5|MotionSense",
               "device:RealWorld_thigh+MotionSense:10+10|RealWorld_thigh",
               "device:RealWorld_thigh+MotionSense:10+10|MotionSense"]
    rot = ["in-domain\nRealWorld", "in-domain\nMotionSense", "cross5+5\nalvo RealWorld",
           "cross5+5\nalvo MotionSense", "cross10+10\nalvo RealWorld",
           "cross10+10\nalvo MotionSense"]
    M, C = matriz(df, ordem, colunas, "cel")
    v = escala(M)
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    im = heat(ax, M, ordem, rot, vmax=v, contagem=C, fs=7.5)
    barra(fig, im, ax, f"Δ mediano (pp) — cor saturada em ±{v:.0f}")
    ax.set_title("ANEXO B — o placar federado, coluna a coluna  (destrincha as figs. 2 e 3)\n"
                 "cada célula: mediana sobre os 5 regimes · abaixo, quantos regimes ficaram "
                 f"positivos (n = 5 → piso de p = {p_minimo(5):.3f}: nada aqui pode ser significante)",
                 loc="left", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    S.salva(fig, "anexoB_placar_por_federacao_e_alvo")


# ── anexo C ← figs. 2 e 3 ─────────────────────────────────────────────────────
def anexoC_por_regime(dc, df, ordem):
    Mc, Cc = matriz(dc, ordem, CEN_SHOTS, "n_shots")
    Mf, Cf = matriz(df, ordem, SHOTS, "n_shots")
    v = escala(Mc, Mf)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9),
                             gridspec_kw={"width_ratios": [4, 5]})
    im = heat(axes[0], Mc, ordem, ["1", "10", "100", "full"], v, Cc, fs=7.5)
    heat(axes[1], Mf, [""] * len(ordem), SHOTS_LAB, v, Cf, fs=7.5)
    for ax, M, cols, tit, sub in [
            (axes[0], Mc, CEN_SHOTS, "Centralizado", "n = 6 datasets por célula"),
            (axes[1], Mf, SHOTS, "Federado", "n = 6 (spec × alvo) por célula")]:
        col = np.nanmedian(M, axis=0)
        alto, baixo = cols[int(np.argmax(col))], cols[int(np.argmin(col))]
        ax.set_title(f"{tit}  ·  {sub}\nmediana dos 8 pares — máx: «{alto}» {col.max():+.1f} pp"
                     f"  ·  mín: «{baixo}» {col.min():+.1f} pp",
                     loc="left", fontsize=8.5, color=INK, pad=8)
    # Os dois eixos NÃO são a mesma escala: no centralizado o regime é do benchmark
    # (rótulos por classe, dataset inteiro); no federado é por cliente.
    axes[0].set_xlabel("rótulos por classe (regimes do benchmark)")
    axes[1].set_xlabel("rótulos por classe, por cliente")
    barra(fig, im, axes[1], f"Δ mediano (pp) — cor saturada em ±{v:.0f}")
    fig.suptitle("ANEXO C — onde no orçamento de rótulos cada par ajuda  (destrincha as figs. 2 e 3)\n"
                 "o número de baixo é quantos domínios ficaram positivos · os dois painéis não "
                 "compartilham a escala do eixo x (veja os rótulos)",
                 x=.008, ha="left", fontsize=11.5, y=1.05)
    fig.tight_layout()
    S.salva(fig, "anexoC_placar_por_regime")


# ── anexo D ← fig. 3 ──────────────────────────────────────────────────────────
def anexoD_por_dataset(dc, ordem):
    M, C = matriz(dc, ordem, DS6, "target")
    v = escala(M)
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    im = heat(ax, M, ordem, [d.replace("RealWorld_", "RW_") for d in DS6],
              vmax=v, contagem=C, fs=7.5)
    barra(fig, im, ax, f"Δ mediano (pp) — cor saturada em ±{v:.0f}")
    ax.set_xlabel("domínio (in-domain, centralizado)")
    col = np.nanmedian(M, axis=0)
    alto, baixo = DS6[int(np.argmax(col))], DS6[int(np.argmin(col))]
    ax.set_title("ANEXO D — o placar centralizado, domínio a domínio  (destrincha a fig. 3)\n"
                 "cada célula: mediana sobre os 4 regimes · abaixo, quantos regimes ficaram positivos\n"
                 f"quem mais responde ao SSL: {alto} ({col.max():+.1f} pp mediano sobre os 8 pares)"
                 f"  ·  quem menos: {baixo} ({col.min():+.1f} pp)",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    fig.tight_layout()
    S.salva(fig, "anexoD_placar_por_dataset")


# ── anexo E ← fig. 6 ──────────────────────────────────────────────────────────
def anexoE_custo_federar_por_encoder(cel):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, m in zip(axes, ["tfc", "lfr"]):
        ax.axhline(0, color=INK2, lw=1.1, zorder=2)
        pontas, med, um = {}, {}, {}
        for enc in ENCODERS:
            c = ENC_COR[enc]
            g = cel[(cel.method == m) & (cel.encoder == enc)]
            v = g.groupby("n_shots")["d"].mean().reindex(SHOTS)
            ax.plot(range(len(SHOTS)), v.values, "-o", lw=2, ms=6, color=c,
                    mec=SURF, mew=1.2, zorder=3)
            pontas[enc] = (c, float(v.values[-1]))
            med[enc] = float(np.median(v.values))
            um[enc] = float(v.values[0])
        rotula_pontas(ax, pontas, len(SHOTS) - 1, folga=1.1)
        # Duas leituras diferentes, e citar só a mediana esconde a outra: o TF-C+rnn
        # tem mediana POSITIVA e mesmo assim é um dos que mais afunda no 1-shot.
        pior, pior1 = min(med, key=med.get), min(um, key=um.get)
        neg1 = sum(u < -1 for u in um.values())
        ax.set_title(f"{LAB[m]}  ·  pior mediano: {pior} ({med[pior]:+.1f} pp)\n"
                     f"pior no 1-shot: {pior1} ({um[pior1]:+.1f} pp)  ·  {neg1} de 4 encoders "
                     "abaixo de −1 pp no 1-shot",
                     loc="left", fontsize=9, color=INK, pad=6)
        ax.set_xticks(range(len(SHOTS)))
        ax.set_xticklabels(SHOTS_LAB)
        ax.set_xlim(-.35, len(SHOTS) - .5)
        ax.set_xlabel("rótulos por classe, por cliente")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Δ acurácia: 10 clientes − 1 cliente (pp)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="-", lw=2, ms=6, color=ENC_COR[e],
                               mec=SURF, mew=1.2, label=e) for e in ENCODERS],
               loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(.5, -.075))
    fig.suptitle("ANEXO E — o custo de federar o pré-treino, encoder a encoder  (destrincha a fig. 6)\n"
                 "dentro de um mesmo método os encoders NÃO pagam igual: em cada painel há "
                 "encoder que ganha e encoder que afunda, e a linha da fig. 6 é a média deles",
                 x=.008, ha="left", fontsize=11.5, y=1.06)
    fig.tight_layout()
    S.salva(fig, "anexoE_custo_federar_por_encoder")


# ── anexo F ← fig. 4 ──────────────────────────────────────────────────────────
def anexoF_replica_por_regime(dc):
    from scipy.stats import spearmanr
    regimes = ["1", "10", "100", "full"]
    nosso = dc.groupby(["metodo", "encoder", "n_shots"]).d.median()
    rot = {"1": "1 rótulo", "10": "10 rótulos", "100": "100 rótulos", "full": "full"}
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9), sharex=True, sharey=True)
    stats = {}
    for ax, k in zip(axes, regimes):
        xs, ys = [], []
        for m in ("lfr", "tfc"):
            for e in ENCODERS:
                b = BENCH_T10[(m, e)][k] - BENCH_T10[("sup", e)][k]
                v = float(nosso.loc[(LAB[m], e, k)])
                xs.append(b)
                ys.append(v)
                ax.plot(b, v, "o", ms=8, color=S.CM[LAB[m]], mec=SURF, mew=1.4, zorder=3)
        lim = [-8, 25]
        ax.plot(lim, lim, ls=(0, (4, 3)), lw=1.2, color=GRID, zorder=1)
        ax.axhline(0, color=GRID, lw=1, zorder=1)
        ax.axvline(0, color=GRID, lw=1, zorder=1)
        rho = spearmanr(xs, ys).statistic
        sinal = int((np.sign(xs) == np.sign(ys)).sum())
        stats[k] = (rho, sinal)
        ax.set_title(rot[k], loc="left", fontsize=10, color=INK, pad=6)
        ax.annotate(f"ρ = {rho:+.2f}\nsinal {sinal}/8", (.04, .96), xycoords="axes fraction",
                    va="top", fontsize=8.5, color=INK, fontweight="bold")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("Δ do benchmark (pp)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Δ nosso (pp)")
    pior_rho = min(stats, key=lambda k: stats[k][0])
    pior_sin = min(stats, key=lambda k: stats[k][1])
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=S.CM[LAB[m]],
                               mec=SURF, mew=1.4, label=LAB[m]) for m in ("tfc", "lfr")]
               + [Line2D([], [], ls=(0, (4, 3)), lw=1.2, color=GRID, label="réplica exata (y = x)")],
               loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5, -.1))
    fig.suptitle("ANEXO F — a réplica do benchmark, regime a regime  (destrincha a fig. 4)\n"
                 "a fig. 4 usa a mediana sobre os 4 regimes; a concordância de ORDEM se sustenta em "
                 f"todos os quatro (o pior é «{rot[pior_rho]}», ρ = {stats[pior_rho][0]:+.2f})\n"
                 f"e a de SINAL cai só no regime de {rot[pior_sin]} ({stats[pior_sin][1]}/8) — "
                 "onde quase todo Δ está colado no zero e trocar de sinal custa nada",
                 x=.008, ha="left", fontsize=11, y=1.13)
    fig.tight_layout()
    S.salva(fig, "anexoF_replica_por_regime")


def main():
    global CMAP
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True,
                    help="pasta de destino (ex.: docs/apresentacao_11_08). "
                         "Obrigatório: nunca sobrescreve registro de apresentação entregue.")
    args = ap.parse_args()
    out = Path(args.outdir)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    S.OUT = out
    _style()
    CMAP = cmap_divergente()

    dc, df = delta_cen(), delta_fed()
    ordem = ordena(dc)

    print(f"Escrevendo em {out.relative_to(PROJECT_ROOT)}/")
    anexoA_shift_por_encoder(carrega_sl_shift())
    anexoB_por_federacao_e_alvo(df, ordem)
    anexoC_por_regime(dc, df, ordem)
    anexoD_por_dataset(dc, ordem)
    cel, _ = carrega_exp2()
    anexoE_custo_federar_por_encoder(cel.reset_index())
    anexoF_replica_por_regime(dc)
    print("pronto.")


if __name__ == "__main__":
    main()
