#!/usr/bin/env python3
"""Figuras da apresentação de 2026-08-05 (orientação).

Só LÊ os caches e escreve PNG no `--outdir`. Não treina, não escreve em results/.
As figuras espelham as do `fedssl_cross_device_avaliation.ipynb` (§4.2, §9.1, §9.2)
e do `cross_device_avaliation.ipynb` (§9.2), com fundo transparente e fonte maior,
para caber num slide 20x11,25 pol.

Uso:
    poetry run python scripts/analysis/build_assets_05_08.py \
        --outdir docs/apresentacao_05_08
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

PART_DIR = PROJECT_ROOT / "results" / "fedssl_cross_device_parts"
BASE_CSV = PROJECT_ROOT / "results" / "fed_cross_device.csv"
BASE_PARTS = PROJECT_ROOT / "results" / "fed_cross_device_parts"

CM = {"none": "#2a78d6", "tfc": "#eb6834", "lfr": "#1baf7a"}
LAB3 = {"none": "Supervisionado", "lfr": "LFR", "tfc": "TF-C"}
COLOR_ENC = dict(zip(["resnetse5", "cnnpff", "rnn", "tstcc"],
                     ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, INK2 = "#1a1a19", "#5c5b55"
ENCODERS = ["resnetse5", "cnnpff", "rnn", "tstcc"]
ORDEM_BARRA = ["none", "lfr", "tfc"]
SHOTS = ["1", "2", "5", "10"]
ALL_SHOTS = SHOTS + ["full"]
IN_SPEC = {"RealWorld_thigh": "device:RealWorld_thigh:10",
           "MotionSense": "device:MotionSense:10"}
C55 = "device:RealWorld_thigh+MotionSense:5+5"
C1010 = "device:RealWorld_thigh+MotionSense:10+10"
ARM = {"device:RealWorld_thigh:10": "in10-RW", "device:MotionSense:10": "in10-MS",
       C55: "cross5+5", C1010: "cross10+10"}
BRACO = {"in10-RW": "in10", "in10-MS": "in10",
         "cross5+5": "cross5+5", "cross10+10": "cross10+10"}
BRACOS = ["in10", "cross5+5", "cross10+10"]
TAG = {"RealWorld_thigh": "RW", "MotionSense": "MS"}
KEY = ["method", "encoder", "spec", "seed", "n_shots"]
PLATEAU = 20

# Fonte maior que a do notebook: no slide a figura é vista de longe.
plt.rcParams.update({
    "figure.dpi": 140, "savefig.transparent": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": "#e8e7e2",
    "font.size": 13, "legend.frameon": False,
})


def style(ax, axis="y"):
    ax.grid(axis=axis, lw=0.7)
    ax.set_axisbelow(True)
    return ax


# --------------------------------------------------------------- dados
def carregar():
    raw = pd.concat([pd.read_csv(f) for f in sorted(PART_DIR.glob("*.csv"))],
                    ignore_index=True)
    raw["n_shots"] = raw["n_shots"].astype(str)
    rmax = int(raw["round"].max())
    sel = (raw[raw["round"] > rmax - PLATEAU]
           .groupby(KEY + ["target"])["test_acc"].mean().reset_index())
    sel["arm"] = sel["spec"].map(ARM)
    return sel


def acuracia(sel):
    """(encoder, method, braco, alvo, degrau, seed) -> acuracia %. Sem mediar alvos."""
    b = pd.read_csv(BASE_CSV)
    b = b[(b.local_epochs == 5) & (b["round"] > b["round"].max() - PLATEAU)
          & b.spec.isin(ARM)]
    b = (b.groupby(["encoder", "spec", "target", "seed"])["test_acc"].mean()
         .reset_index().assign(method="none", n_shots="full"))
    a = pd.concat([sel, b.assign(arm=b.spec.map(ARM))], ignore_index=True)
    a["braco"] = a.arm.map(BRACO)
    a["test_acc"] *= 100
    return (a.groupby(["encoder", "method", "braco", "target", "n_shots",
                       "seed"])["test_acc"].mean().reset_index())


def stats(df, col):
    return df.groupby(["encoder", "method", col])["test_acc"].agg(["mean", "std"])


# --------------------------------------------------------------- figuras
def fig_barras(tab, grupos, titulo, sub, xlabel, out, divisor=None, larg=9.0):
    """Uma figura POR ENCODER: 3 barras (SL/LFR/TF-C) por grupo. ylim comum."""
    x, w = np.arange(len(grupos)), 0.8 / len(ORDEM_BARRA)
    topo = max(np.nansum([tab["mean"].get((e, m, g), np.nan),
                          tab["std"].get((e, m, g), 0)])
               for e in ENCODERS for m in ORDEM_BARRA for g in grupos)
    for enc in ENCODERS:
        fig, ax = plt.subplots(figsize=(larg, 5.0))
        style(ax)
        for k, m in enumerate(ORDEM_BARRA):
            mu = [tab["mean"].get((enc, m, g), np.nan) for g in grupos]
            sd = [tab["std"].get((enc, m, g), np.nan) for g in grupos]
            ax.bar(x + (k - 1) * w, mu, w * 0.88, color=CM[m], label=LAB3[m],
                   yerr=sd, error_kw=dict(ecolor=INK2, lw=1.1, capsize=0), zorder=3)
        if divisor is not None:
            ax.axvline(divisor, color="#c9c8c2", lw=1.2, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(grupos, fontsize=11)
        ax.set_ylim(0, topo * 1.06)
        ax.set_ylabel("acurácia de teste (%)")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{titulo} — {enc}\n{sub}", fontsize=13, loc="left", color=INK)
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        fig.tight_layout()
        fig.savefig(out / f"fig_{titulo.split()[0]}_{enc}.png", bbox_inches="tight")
        plt.close(fig)


def fig_h1_contrastes(out):
    """Os dois contrastes do baseline supervisionado — a hipotese-pilar."""
    d = pd.concat([pd.read_csv(f) for f in sorted(BASE_PARTS.glob("*.csv"))],
                  ignore_index=True)
    d = d[(d.local_epochs == 5)].dropna(subset=["val_acc"])
    d["arm"] = d.spec.map(ARM)
    d = d.dropna(subset=["arm"])
    v = d.groupby(["encoder", "arm", "seed", "round"])["val_acc"].mean().reset_index()
    best = d.merge(v.loc[v.groupby(["encoder", "arm", "seed"])["val_acc"].idxmax(),
                         ["encoder", "arm", "seed", "round"]],
                   on=["encoder", "arm", "seed", "round"])
    IN = {"RealWorld_thigh": "in10-RW", "MotionSense": "in10-MS"}
    rows = []
    for rot, arm in (("substituir\n(cross5+5 − in10)", "cross5+5"),
                     ("acrescentar\n(cross10+10 − in10)", "cross10+10")):
        for tgt in IN:
            for e in ENCODERS:
                s = best[(best.encoder == e) & (best.target == tgt)]
                x = s[s.arm == arm].set_index("seed")["test_acc"]
                y = s[s.arm == IN[tgt]].set_index("seed")["test_acc"]
                c = x.index.intersection(y.index)
                rows.append({"contraste": rot, "alvo": tgt, "encoder": e,
                             "delta": (x[c] - y[c]).mean() * 100,
                             "dp": (x[c] - y[c]).std() * 100})
    ct = pd.DataFrame(rows)
    tgts = list(IN)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    x, w = np.arange(len(tgts)), 0.8 / len(ENCODERS)
    for ax, rot in zip(axes, ct.contraste.unique()):
        style(ax)
        s = ct[ct.contraste == rot]
        for j, enc in enumerate(ENCODERS):
            k = s[s.encoder == enc].set_index("alvo").reindex(tgts)
            ax.bar(x + (j - 1.5) * w, k.delta, w * 0.86, color=COLOR_ENC[enc],
                   label=enc, yerr=k.dp,
                   error_kw=dict(ecolor=INK2, lw=1.0, capsize=0), zorder=3)
        ax.axhline(0, color=INK, lw=1.4, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([TAG[t] + "_thigh" if "Real" in t else TAG[t] for t in tgts])
        ax.set_title(rot, fontsize=12.5, loc="left", color=INK)
        ax.set_xlabel("domínio-alvo")
    axes[0].set_ylabel("Δ acurácia vs in-domain (pp)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Federações supervisionadas sob domain shift — barra de erro = dp "
                 "das diferenças pareadas", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out / "fig_h1_contrastes.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", required=True,
                    help="OBRIGATORIO — pasta da apresentacao (nunca sobrescrever registro).")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    sel = carregar()
    acc = acuracia(sel)

    d91 = acc[acc.n_shots == "full"].copy()
    d91["grupo"] = d91.target.map(TAG) + "_" + d91.braco
    fig_barras(stats(d91, "grupo"),
               [f"{t}_{b}" for t in ("RW", "MS") for b in BRACOS],
               "H2", "degrau `full` · cada barra é UMA federação · erro = ±1 dp entre 4 seeds",
               "federação (alvo_braço)", out, divisor=2.5, larg=9.5)

    d92 = acc[acc.braco == "cross5+5"].copy()
    d92["grupo"] = d92.target.map(TAG) + "_" + d92.n_shots
    fig_barras(stats(d92, "grupo"),
               [f"{t}_{s}" for t in ("RW", "MS") for s in ALL_SHOTS],
               "Ladder", "braço cross5+5 · cada bloco é um alvo · erro = ±1 dp",
               "alvo_rótulos por classe", out, divisor=4.5, larg=11.0)

    fig_h1_contrastes(out)
    print(f"[ASSETS] {len(list(out.glob('*.png')))} PNG em {out}")


if __name__ == "__main__":
    main()
