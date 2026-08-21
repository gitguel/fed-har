"""Gera o diagrama do slide 1 do pitch (`docs/pitch_reuniao_semanal.md`).

Esquema conceitual, não lê cache nenhum: clientes (usuários, domínios
diferentes) -> pré-treino SSL federado (rodadas de agregação) -> fine-tuning com
poucos rótulos -> modelo HAR. A barreira tracejada marca o ponto do slide: o
dado bruto nunca sai do cliente, sobe só parâmetro.

Rodar: poetry run python scripts/analysis/build_pitch_diagram.py \\
           --outdir docs/pitch_assets
(`--outdir` é obrigatório, mesma disciplina de build_presentation_assets.py:
pasta de apresentação entregue não se regera)

Saída: fig_pitch_slide1_fedssl.{png,pdf}
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]

# paleta colorblind-safe, a mesma de build_presentation_assets.METHOD_COLORS
C_CLIENT, C_SERVER, C_DOWN = "#0173B2", "#DE8F05", "#029E73"
F_CLIENT, F_SERVER, F_DOWN = "#E8F1F8", "#FDF1DE", "#E4F3EC"
C_BARRIER = "#C44E52"
C_TEXT = "#222222"
C_ARROW = "#555555"

CLIENTS = [("Cliente · domínio A", 58.0), ("Cliente · domínio B", 35.5),
           ("Cliente · domínio C", 13.0)]
CX0, CX1, CH = 5.0, 24.0, 19.0        # cartões dos clientes
BARRIER_X, BARRIER_Y0, BARRIER_Y1 = 30.5, 12.0, 86.0
SX0, SX1, SY0, SY1 = 38.0, 61.0, 34.0, 62.0   # servidor
BX0, BX1 = 66.0, 78.0                 # backbone pré-treinado
MX0, MX1 = 89.0, 99.0                 # modelo HAR
RY0, RY1 = 34.0, 62.0                 # faixa vertical dos blocos da direita
RETURN_Y, RETURN_X = 4.5, 1.5         # canal de volta do modelo global


def box(ax, x0, x1, y0, y1, face, edge, lw=1.8):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.6",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))


def arrow(ax, xy_from, xy_to, color=C_ARROW, lw=2.0, style="-|>",
          connection="arc3,rad=0", ls="-"):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=18, linewidth=lw,
        color=color, linestyle=ls, connectionstyle=connection,
        shrinkA=0, shrinkB=0, zorder=3))


def txt(ax, x, y, s, size=11, weight="normal", color=C_TEXT, ha="center",
        va="center", style="normal", rotation=0):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha,
            va=va, style=style, rotation=rotation, zorder=4)


def signal(ax, x0, x1, y, amp=1.5):
    """Sketch de janela IMU dentro do cartão do cliente (6 canais, sem rótulo)."""
    t = np.linspace(0, 1, 220)
    rng = np.random.default_rng(7)
    for k, c in enumerate([C_CLIENT, "#7FB3D3"]):
        w = (np.sin(2 * np.pi * (2.6 + 1.4 * k) * t + 1.3 * k)
             + 0.35 * rng.standard_normal(t.size))
        ax.plot(x0 + t * (x1 - x0), y + amp * w / 2.6, color=c, linewidth=1.0,
                alpha=0.85, zorder=3)


def build(outdir: Path):
    fig, ax = plt.subplots(figsize=(14, 6.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---------------------------------------------------- fases (títulos topo)
    txt(ax, (CX0 + SX1) / 2, 95, "① Pré-treino SSL federado — sem rótulo",
        size=14, weight="bold", color=C_SERVER)
    txt(ax, (BX0 + MX1) / 2, 95, "② Fine-tuning — poucos rótulos",
        size=14, weight="bold", color=C_DOWN)
    ax.plot([CX0, SX1], [91.5, 91.5], color=C_SERVER, lw=1.2, alpha=0.5)
    ax.plot([BX0, MX1], [91.5, 91.5], color=C_DOWN, lw=1.2, alpha=0.5)

    # ------------------------------------------------------------- clientes
    for label, y0 in CLIENTS:
        box(ax, CX0, CX1, y0, y0 + CH, F_CLIENT, C_CLIENT)
        txt(ax, (CX0 + CX1) / 2, y0 + CH - 4.0, label, size=11.5, weight="bold",
            color=C_CLIENT)
        signal(ax, CX0 + 2.5, CX1 - 2.5, y0 + CH / 2 - 1.0, amp=6.0)
        txt(ax, (CX0 + CX1) / 2, y0 + 3.2,
            "dado não-rotulado · SSL local", size=10, color=C_TEXT)
    txt(ax, (CX0 + CX1) / 2, CLIENTS[-1][1] - 4.5,
        "muitos clientes, domínios diferentes", size=10, style="italic",
        color="#555555")

    # -------------------------------------------- barreira: o dado não sobe
    ax.plot([BARRIER_X, BARRIER_X], [BARRIER_Y0, BARRIER_Y1], color=C_BARRIER,
            lw=2.0, ls=(0, (6, 4)), zorder=2)
    txt(ax, BARRIER_X, 88.0, "o dado bruto nunca sai do cliente", size=12,
        weight="bold", color=C_BARRIER)

    # janelas brutas: seta barrada na barreira
    arrow(ax, (CX1 + 0.8, 82.0), (BARRIER_X - 1.2, 82.0), color=C_BARRIER,
          lw=1.6)
    txt(ax, BARRIER_X + 0.1, 82.0, "✕", size=17, weight="bold", color=C_BARRIER)
    txt(ax, CX1 + 0.2, 82.0, "janelas brutas", size=10, color=C_BARRIER,
        ha="right")

    # ------------------------------------------------------------- servidor
    box(ax, SX0, SX1, SY0, SY1, F_SERVER, C_SERVER)
    txt(ax, (SX0 + SX1) / 2, SY1 - 6.5, "Servidor", size=13, weight="bold",
        color=C_SERVER)
    txt(ax, (SX0 + SX1) / 2, SY1 - 13.5, "agrega os parâmetros\n(FedAvg)",
        size=11)
    txt(ax, (SX0 + SX1) / 2, SY0 + 4.5, "R rodadas", size=10.5, style="italic",
        color="#555555")

    # subida: Δ de parâmetros de cada cliente
    for _, y0 in CLIENTS:
        arrow(ax, (CX1 + 0.8, y0 + CH / 2), (SX0 - 0.8, (SY0 + SY1) / 2))
    txt(ax, 40.0, 71.0, "Δ parâmetros", size=12, weight="bold", color=C_ARROW)

    # descida: modelo global volta para todos os clientes (canal externo)
    ax.plot([(SX0 + SX1) / 2, (SX0 + SX1) / 2, RETURN_X, RETURN_X],
            [SY0, RETURN_Y, RETURN_Y, CLIENTS[0][1] + CH / 2],
            color="#8A8A8A", lw=1.8, solid_joinstyle="round", zorder=2)
    for _, y0 in CLIENTS:
        arrow(ax, (RETURN_X, y0 + CH / 2), (CX0 - 0.8, y0 + CH / 2),
              color="#8A8A8A", lw=1.8)
    txt(ax, 33.0, 7.2, "modelo global", size=11, color="#6E6E6E")

    # -------------------------------------------------- backbone -> modelo
    box(ax, BX0, BX1, RY0, RY1, F_DOWN, C_DOWN)
    txt(ax, (BX0 + BX1) / 2, RY1 - 8.0, "Backbone\npré-treinado", size=12,
        weight="bold", color=C_DOWN)
    txt(ax, (BX0 + BX1) / 2, RY0 + 7.0, "0 rótulos\naté aqui", size=10.5,
        style="italic", color="#555555")

    box(ax, MX0, MX1, RY0, RY1, F_DOWN, C_DOWN)
    txt(ax, (MX0 + MX1) / 2, RY1 - 8.0, "Modelo\nHAR", size=12.5,
        weight="bold", color=C_DOWN)
    txt(ax, (MX0 + MX1) / 2, RY0 + 7.0, "atividade\ndo usuário", size=10.5,
        style="italic", color="#555555")

    arrow(ax, (SX1 + 0.6, (SY0 + SY1) / 2), (BX0 - 0.6, (RY0 + RY1) / 2), lw=2.2)
    arrow(ax, (BX1 + 0.6, (RY0 + RY1) / 2), (MX0 - 0.6, (RY0 + RY1) / 2), lw=2.2)
    txt(ax, (BX1 + MX0) / 2, RY1 - 7.0, "fine-tuning\npoucos rótulos", size=10,
        weight="bold", color=C_DOWN)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for ext, kw in [("png", {"dpi": 300}), ("pdf", {})]:
        fig.savefig(outdir / f"fig_pitch_slide1_fedssl.{ext}",
                    bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print(f"escrito em {outdir}/fig_pitch_slide1_fedssl.{{png,pdf}}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", required=True, help="pasta de saída dos assets")
    a = p.parse_args()
    out = Path(a.outdir)
    build(out if out.is_absolute() else ROOT / out)
