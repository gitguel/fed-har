#!/usr/bin/env python3
"""Matrizes Δ(SSL − supervisionado) no regime `full`, uma por spec.

Eixo x = método SSL (LFR, TF-C); eixo y = encoder; célula = Δ de acurácia em pp
contra o FedAvg supervisionado, com o desvio padrão entre seeds. Depois, a versão
binária das mesmas matrizes.

Só LÊ os caches. O braço supervisionado no `full` vem de `fed_cross_device.csv`
(a grade Fed-SSL não tem `none@full`); mesmo protocolo, outro mutirão.

Nos specs mistos (`5+5`, `10+10`) o Δ é mediado sobre os DOIS alvos. É seguro
porque Δ é diferença pareada dentro do mesmo alvo — a diferença de patamar entre
RealWorld_thigh e MotionSense (~15 pp) cancela. Mediar o NÍVEL seria outra coisa.

Uso:
    poetry run python scripts/analysis/matriz_delta_full.py --outdir <pasta>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PART = PROJECT_ROOT / "results" / "fedssl_cross_device_parts"
BASE = PROJECT_ROOT / "results" / "fed_cross_device.csv"

SPECS = {"device:RealWorld_thigh:10": "RW10",
         "device:MotionSense:10": "MS10",
         "device:RealWorld_thigh+MotionSense:5+5": "5+5",
         "device:RealWorld_thigh+MotionSense:10+10": "10+10"}
ENCODERS = ["resnetse5", "cnnpff", "rnn", "tstcc"]
METODOS = ["lfr", "tfc"]
LAB = {"lfr": "LFR", "tfc": "TF-C"}
PLATEAU = 20
INK, INK2 = "#1a1a19", "#5c5b55"
NEG, NEUTRO, POS = "#e34948", "#eeeeec", "#2a78d6"
# Diverging: dois tons + cinza NEUTRO no meio (nunca um hue no ponto médio).
CMAP = LinearSegmentedColormap.from_list("delta", [NEG, NEUTRO, POS])

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "text.color": INK,
                     "axes.edgecolor": "#c9c8c2", "xtick.color": INK2,
                     "ytick.color": INK2})


def tabela():
    """Δ por (método, encoder, spec): média e dp entre as 4 seeds."""
    raw = pd.concat([pd.read_csv(f) for f in sorted(PART.glob("*.csv"))],
                    ignore_index=True)
    raw["n_shots"] = raw["n_shots"].astype(str)
    r = int(raw["round"].max())
    ssl = (raw[(raw["round"] > r - PLATEAU) & (raw.n_shots == "full")
               & (raw.method != "none")]
           .groupby(["method", "encoder", "spec", "target", "seed"])["test_acc"]
           .mean() * 100)
    b = pd.read_csv(BASE)
    b = b[(b.local_epochs == 5) & (b["round"] > b["round"].max() - PLATEAU)
          & b.spec.isin(SPECS)]
    sl = b.groupby(["encoder", "spec", "target", "seed"])["test_acc"].mean() * 100
    d = (ssl.reset_index()
         .merge(sl.rename("sl").reset_index(),
                on=["encoder", "spec", "target", "seed"]))
    d["delta"] = d.test_acc - d.sl
    # média sobre alvos POR SEED: o dp resultante é variação de seed, não de alvo
    por_seed = d.groupby(["method", "encoder", "spec", "seed"])["delta"].mean()
    return por_seed.groupby(level=[0, 1, 2]).agg(["mean", "std"])


def _grade(g, spec, campo):
    return np.array([[g.loc[(m, e, spec), campo] for m in METODOS] for e in ENCODERS])


def fig_delta(g, out):
    lim = np.nanmax(np.abs([_grade(g, s, "mean") for s in SPECS]))
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.2))
    for ax, (spec, tag) in zip(axes, SPECS.items()):
        mu, sd = _grade(g, spec, "mean"), _grade(g, spec, "std")
        ax.imshow(mu, cmap=CMAP, vmin=-lim, vmax=lim)
        for i in range(len(ENCODERS)):
            for j in range(len(METODOS)):
                # texto claro só onde o fundo está saturado
                cor = "white" if abs(mu[i, j]) > 0.55 * lim else INK
                ax.text(j, i, f"{mu[i, j]:+.1f}\n±{sd[i, j]:.1f}", ha="center",
                        va="center", fontsize=9.5, color=cor, linespacing=1.35)
        ax.set_xticks(range(len(METODOS)))
        ax.set_xticklabels([LAB[m] for m in METODOS])
        ax.set_yticks(range(len(ENCODERS)))
        ax.set_yticklabels(ENCODERS if ax is axes[0] else [])
        ax.set_title(tag, fontsize=11.5, color=INK)
        ax.set_xticks(np.arange(-.5, len(METODOS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(ENCODERS), 1), minor=True)
        ax.grid(which="minor", color="white", lw=2.5)
        ax.tick_params(which="minor", length=0)
    fig.suptitle("Δ acurácia (SSL − FedAvg supervisionado), regime `full`, em pp\n"
                 "média ± 1 dp entre as 4 seeds · azul = SSL ganha, vermelho = perde",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out / "matriz_delta_full.png", bbox_inches="tight")
    plt.close(fig)


def fig_binaria(g, out):
    """Duas regras, porque 'dentro do desvio padrão' admite as duas leituras."""
    regras = [("sinal do Δ  (1 se Δ > 0)", lambda m, s: m > 0),
              ("Δ acima do ruído  (1 se Δ − dp > 0)", lambda m, s: m - s > 0)]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 8.0))
    for row, (nome, regra) in enumerate(regras):
        for col, (spec, tag) in enumerate(SPECS.items()):
            ax = axes[row, col]
            mu, sd = _grade(g, spec, "mean"), _grade(g, spec, "std")
            b = regra(mu, sd).astype(int)
            ax.imshow(b, cmap=LinearSegmentedColormap.from_list("bin", [NEG, POS]),
                      vmin=0, vmax=1)
            for i in range(len(ENCODERS)):
                for j in range(len(METODOS)):
                    ax.text(j, i, str(b[i, j]), ha="center", va="center",
                            fontsize=17, color="white", fontweight="bold")
            ax.set_xticks(range(len(METODOS)))
            ax.set_xticklabels([LAB[m] for m in METODOS])
            ax.set_yticks(range(len(ENCODERS)))
            ax.set_yticklabels(ENCODERS if col == 0 else [])
            if row == 0:
                ax.set_title(tag, fontsize=11.5, color=INK)
            if col == 0:
                ax.set_ylabel(nome, fontsize=10, color=INK2, labelpad=10)
            ax.set_xticks(np.arange(-.5, len(METODOS), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(ENCODERS), 1), minor=True)
            ax.grid(which="minor", color="white", lw=2.5)
            ax.tick_params(which="minor", length=0)
    fig.suptitle("Matriz binária — o SSL supera o supervisionado no `full`?\n"
                 "1 (azul) = sim · 0 (vermelho) = não · duas regras de decisão",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out / "matriz_binaria_full.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    g = tabela()
    fig_delta(g, out)
    fig_binaria(g, out)
    t = g.round(2).reset_index()
    t["spec"] = t.spec.map(SPECS)
    t["sinal"] = (t["mean"] > 0).astype(int)
    t["acima_do_ruido"] = (t["mean"] - t["std"] > 0).astype(int)
    t.to_csv(out / "matriz_delta_full.csv", index=False)
    print(t.to_string(index=False))
    print(f"\n[OK] 2 PNG + 1 CSV em {out}")
    print(f"células com 1 pela regra do sinal: {t.sinal.sum()}/32 | "
          f"acima do ruído: {t.acima_do_ruido.sum()}/32")


if __name__ == "__main__":
    main()
