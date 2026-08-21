"""Gera o diagrama de HAR do pitch (`docs/pitch_reuniao_semanal.md`).

Esquema conceitual, não lê cache nenhum. Apresenta o *dado* de HAR de forma
geral, sem detalhe de processamento:

  ① as atividades do usuário em volta do smartphone com IMU -> sinal contínuo;
  ② nota no canto inferior direito (~10% da figura): a mesma atividade medida em
     três posições de sensor dá sinais diferentes.

Os pictogramas são **Material Symbols Outlined** (Google, Apache 2.0), baixados
em `scripts/analysis/icons/*.svg` e convertidos aqui de path SVG para
`matplotlib.path.Path` — assim o ícone continua vetorial no PDF. Rebaixar um
ícone que falte:

    curl -o scripts/analysis/icons/<nome>.svg \\
      https://raw.githubusercontent.com/google/material-design-icons/master/\\
symbols/web/<nome>/materialsymbolsoutlined/<nome>_24px.svg

Os *sinais* são sintéticos (seno + harmônico + ruído com semente fixa), só
ilustração — nenhum vem de `datasets/` ou `results/`.

Complementa `build_pitch_diagram.py`, que desenha a federação (Fed-SSL); aqui é
só o dado de HAR.

Rodar: poetry run python scripts/analysis/build_har_diagram.py \\
           --outdir docs/pitch_assets
(`--outdir` é obrigatório, mesma disciplina de build_presentation_assets.py:
pasta de apresentação entregue não se regera)

Saída: fig_pitch_har.{png,pdf}
"""

import argparse
import re
from pathlib import Path as FsPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.transforms import Affine2D

ROOT = FsPath(__file__).resolve().parents[2]
ICONS = FsPath(__file__).resolve().parent / "icons"

# mesma paleta colorblind-safe de build_pitch_diagram.py
C_MAIN, C_WARN = "#0173B2", "#C44E52"
F_MAIN, F_WARN = "#E8F1F8", "#FAECEC"
C_TEXT, C_ARROW, C_MUTED = "#222222", "#555555", "#6E6E6E"

# ------------------------------------------------------------------ geometria
FIG_W, FIG_H = 13.0, 6.6
XR, YR = 100.0, 98.0                  # extensão dos eixos (ylim 2..100)
# uma unidade de x vale ~metade de uma de y em polegadas; os ícones são
# desenhados em proporção isométrica e corrigidos por este fator
ASPECT = (FIG_H / YR) / (FIG_W / XR)

CX, CY, RX, RY = 31.0, 50.0, 26.0, 31.0   # centro e raios da órbita
ICON_H, PHONE_H = 14.0, 20.0
ICON_SCALE = {"stairs_2": 0.85}       # formas cheias pesam mais que os bonecos
# raios onde as setas órbita->celular começam e terminam (folga nos dois lados)
AS_X, AS_Y, AE_X, AE_Y = 19.5, 21.5, 7.5, 13.5
# (ângulo, rótulo, ícone, lado do rótulo)
ORBIT = [(126.0, "sentado", "airline_seat_recline_normal", "cima"),
         (180.0, "em pé", "man", "baixo"),
         (234.0, "escada", "stairs_2", "baixo"),
         (306.0, "correndo", "directions_run", "baixo"),
         (54.0, "andando", "directions_walk", "cima")]

SG_X0, SG_X1, SG_Y0, SG_Y1 = 64.0, 99.0, 34.0, 64.0      # painel de sinal
NX0, NX1, NY0, NY1 = 64.0, 99.0, 4.0, 30.0               # nota (canto inf. dir.)

# amplitude / freq. base / harmônico (fator, múltiplo) / ruído por posição
WALK = {"cintura": (0.42, 4.0, 0.15, 2.0, 0.06),
        "bolso":   (1.00, 4.0, 0.45, 2.0, 0.09),
        "coxa":    (0.85, 4.0, 0.55, 3.0, 0.07)}

# ------------------------------------------------------- SVG path -> mpl Path
_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_SEG = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)")


def svg_path(d: str) -> MPath:
    """Converte um atributo `d` de SVG em Path do matplotlib (y invertido).

    Cobre M/L/H/V/C/S/Q/T/Z (abs. e rel.) — o suficiente para os Material
    Symbols. Arco elíptico (A) não aparece neles e é rejeitado explicitamente.
    """
    verts, codes = [], []
    cur = start = (0.0, 0.0)
    prev_ctrl, prev_cmd = None, ""

    def add(pt, code):
        verts.append(pt)
        codes.append(code)

    for cmd, raw in _SEG.findall(d):
        nums = [float(v) for v in _NUM.findall(raw)]
        rel = cmd.islower()
        c = cmd.upper()
        if c == "A":
            raise ValueError("arco elíptico não suportado neste conversor")
        if c == "Z":
            add(start, MPath.CLOSEPOLY)
            cur, prev_ctrl, prev_cmd = start, None, c
            continue
        step = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4,
                "T": 2}[c]
        for i in range(0, len(nums), step):
            a = nums[i:i + step]
            if c in ("M", "L"):
                pt = (cur[0] + a[0], cur[1] + a[1]) if rel else (a[0], a[1])
                add(pt, MPath.MOVETO if (c == "M" and i == 0) else MPath.LINETO)
                if c == "M" and i == 0:
                    start = pt
                cur, prev_ctrl = pt, None
            elif c in ("H", "V"):
                x = cur[0] + a[0] if rel else a[0]
                y = cur[1] + a[0] if rel else a[0]
                pt = (x, cur[1]) if c == "H" else (cur[0], y)
                add(pt, MPath.LINETO)
                cur, prev_ctrl = pt, None
            elif c in ("C", "S"):
                if c == "C":
                    p = [(a[0], a[1]), (a[2], a[3]), (a[4], a[5])]
                else:
                    refl = (cur if prev_ctrl is None or prev_cmd not in "CS"
                            else (2 * cur[0] - prev_ctrl[0],
                                  2 * cur[1] - prev_ctrl[1]))
                    p = [(0.0, 0.0), (a[0], a[1]), (a[2], a[3])]
                if rel:
                    p = [(cur[0] + x, cur[1] + y) for x, y in p]
                if c == "S":
                    p[0] = refl
                for pt in p:
                    add(pt, MPath.CURVE4)
                cur, prev_ctrl = p[2], p[1]
            else:                                   # Q / T (quadráticas)
                if c == "Q":
                    p = [(a[0], a[1]), (a[2], a[3])]
                    if rel:
                        p = [(cur[0] + x, cur[1] + y) for x, y in p]
                else:
                    ctrl = (cur if prev_ctrl is None or prev_cmd not in "QT"
                            else (2 * cur[0] - prev_ctrl[0],
                                  2 * cur[1] - prev_ctrl[1]))
                    end = ((cur[0] + a[0], cur[1] + a[1]) if rel
                           else (a[0], a[1]))
                    p = [ctrl, end]
                for pt in p:
                    add(pt, MPath.CURVE3)
                cur, prev_ctrl = p[1], p[0]
            prev_cmd = c
    pts = np.asarray(verts, dtype=float)
    pts[:, 1] *= -1.0                                # SVG cresce para baixo
    return MPath(pts, codes)


def load_icon(name: str) -> MPath:
    svg = (ICONS / f"{name}.svg").read_text()
    m = re.search(r'\sd="([^"]+)"', svg)
    if m is None:
        raise ValueError(f"sem atributo d em {name}.svg")
    return svg_path(m.group(1))


def draw_icon(ax, name, cx, cy, height, color=C_MAIN, zorder=3):
    """Desenha o ícone centrado em (cx, cy) com `height` unidades de altura."""
    p = load_icon(name)
    bb = p.get_extents()
    sy = height / bb.height
    sx = sy * ASPECT                                  # preserva a proporção
    tr = (Affine2D()
          .translate(-(bb.x0 + bb.x1) / 2, -(bb.y0 + bb.y1) / 2)
          .scale(sx, sy).translate(cx, cy))
    ax.add_patch(PathPatch(tr.transform_path(p), facecolor=color,
                           edgecolor="none", zorder=zorder))


# ------------------------------------------------------------------ desenho
def txt(ax, x, y, s, size=11, weight="normal", color=C_TEXT, ha="center",
        va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha,
            va=va, style=style, zorder=5)


def arrow(ax, xy_from, xy_to, color=C_ARROW, lw=2.0):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>",
                                 mutation_scale=16, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4))


def traces(ax, x0, x1, y, amp, freq, harm, harm_freq, noise, seed,
           colors=(C_MAIN, "#7FB3D3"), lw=1.2, n=300):
    """Trilhas de um sinal periódico — passada de caminhada estilizada."""
    t = np.linspace(0, 1, n)
    rng = np.random.default_rng(seed)
    for k, c in enumerate(colors):
        w = (np.sin(2 * np.pi * freq * t + 0.9 * k)
             + harm * np.sin(2 * np.pi * harm_freq * freq * t + 1.7 * k)
             + noise * rng.standard_normal(n))
        ax.plot(x0 + t * (x1 - x0), y + amp * w, color=c, lw=lw, alpha=0.92,
                zorder=3)


def draw_orbit(ax):
    for ang, label, icon, side in ORBIT:
        a = np.deg2rad(ang)
        x, y = CX + RX * np.cos(a), CY + RY * np.sin(a)
        draw_icon(ax, icon, x, y, ICON_H * ICON_SCALE.get(icon, 1.0))
        dy = ICON_H / 2 + 4.6
        txt(ax, x, y + dy if side == "cima" else y - dy, label, size=12,
            weight="bold", color=C_TEXT)
        # seta curta apontando para o smartphone
        arrow(ax, (CX + AS_X * np.cos(a), CY + AS_Y * np.sin(a)),
              (CX + AE_X * np.cos(a), CY + AE_Y * np.sin(a)),
              color="#9AA3AA", lw=1.6)


def draw_signal(ax):
    ax.add_patch(FancyBboxPatch(
        (SG_X0, SG_Y0), SG_X1 - SG_X0, SG_Y1 - SG_Y0,
        boxstyle="round,pad=0,rounding_size=1.8", facecolor=F_MAIN,
        edgecolor=C_MAIN, linewidth=1.8, zorder=2))
    traces(ax, SG_X0 + 2.0, SG_X1 - 2.0, SG_Y0 + 0.62 * (SG_Y1 - SG_Y0), 4.2,
           4.5, 0.30, 2.0, 0.07, 13, colors=(C_MAIN, "#7FB3D3", "#B6D3E8"),
           lw=1.4)
    txt(ax, (SG_X0 + SG_X1) / 2, SG_Y0 + 3.6,
        "sinal contínuo — acelerômetro + giroscópio, 6 canais", size=10,
        color=C_MUTED, style="italic")


def draw_note(ax):
    ax.add_patch(FancyBboxPatch(
        (NX0, NY0), NX1 - NX0, NY1 - NY0,
        boxstyle="round,pad=0,rounding_size=1.6", facecolor=F_WARN,
        edgecolor=C_WARN, linewidth=1.4, zorder=2))
    txt(ax, NX0 + 2.0, NY1 - 4.6,
        "a mesma atividade — andar —\nmuda com a posição do sensor", size=10.5,
        weight="bold", color=C_WARN, ha="left")
    for i, (name, seed) in enumerate((("cintura", 31), ("bolso", 47),
                                      ("coxa", 59))):
        y = NY1 - 11.5 - i * 5.8
        txt(ax, NX0 + 2.0, y, name, size=10, color=C_TEXT, ha="left")
        amp, f, h, hf, n = WALK[name]
        traces(ax, NX0 + 11.0, NX1 - 2.0, y, 1.9 * amp, f, h, hf, n, seed,
               colors=(C_WARN,), lw=1.1, n=220)


def build(outdir: FsPath):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    txt(ax, 3.0, 96.5, "Dados de HAR: a atividade do usuário vira sinal de IMU",
        size=14.5, weight="bold", color=C_MAIN, ha="left")

    draw_orbit(ax)
    draw_icon(ax, "smartphone", CX, CY, PHONE_H)
    txt(ax, CX, 6.5,
        "todas as atividades passam pelo mesmo sensor — o smartphone com IMU",
        size=10.5, color=C_MUTED, style="italic")

    arrow(ax, (CX + 7.5, CY), (SG_X0 - 1.6, CY), lw=2.2)
    draw_signal(ax)
    draw_note(ax)

    ax.set_xlim(0, 100)
    ax.set_ylim(2, 100)
    ax.axis("off")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for ext, kw in [("png", {"dpi": 300}), ("pdf", {})]:
        fig.savefig(outdir / f"fig_pitch_har.{ext}", bbox_inches="tight",
                    facecolor="white", **kw)
    plt.close(fig)
    print(f"escrito em {outdir}/fig_pitch_har.{{png,pdf}}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", required=True, help="pasta de saída dos assets")
    a = p.parse_args()
    out = FsPath(a.outdir)
    build(out if out.is_absolute() else ROOT / out)
