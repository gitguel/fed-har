"""Gera os assets da apresentação (slides SSL LFR/TF-C + comparação SSL vs SL).

Lê apenas os caches de results/ e escreve em docs/apresentacao/:
  - resultados_ssl.xlsx (abas: resumo_metodos, regimes, encoder_x_tecnica,
    cross_encoder, comb2target)
  - fig_lfr_data_efficiency.png, fig_tfc_data_efficiency.png,
    fig_tfc_comb2target.png, fig_ssl_vs_sl_cenarios.png, fig_ssl_vs_sl_fewshot.png
  - slides_ssl_texto.md (bullets sugeridos por slide)

Rodar: poetry run python scripts/analysis/build_presentation_assets.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "apresentacao"
OUT.mkdir(parents=True, exist_ok=True)

SHOT_ORDER = ["1", "10", "100", "full"]
SHOT_LABELS = ["1-shot", "10-shot", "100-shot", "100%"]
ENCODER_PRETTY = {
    "resnetse5": "ResNet-SE-5",
    "cnnpff": "CNN-PFF",
    "rnn": "BiGRU (RNN)",
    "tstcc": "TS-TCC",
}
METHOD_COLORS = {"SL": "#0173B2", "LFR": "#DE8F05", "TF-C": "#029E73"}
SCEN_PRETTY = {"in": "In-domain (especialista)", "comb": "Combined (multi-domínio)",
               "cross": "Cross-domain (transfer)"}


def load() -> pd.DataFrame:
    sup = pd.read_csv(RESULTS / "supervised_eval_transfer.csv")
    sup["method"], sup["protocol"] = "SL", "sl"
    frames = [sup]
    for slug, name in [("tfc", "TF-C"), ("lfr", "LFR")]:
        d = pd.read_csv(RESULTS / f"ssl_{slug}_eval_transfer.csv")
        d["method"] = name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["n_shots"] = df["n_shots"].astype(str)
    df["setting"] = np.where(
        df.source == "combined", "comb", np.where(df.source == df.target, "in", "cross")
    )
    return df


def load_c2t() -> pd.DataFrame:
    frames = []
    for slug, name in [("tfc", "TF-C"), ("lfr", "LFR")]:
        d = pd.read_csv(RESULTS / f"ssl_{slug}_comb2target_eval_transfer.csv")
        d["method"] = name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["n_shots"] = d["n_shots"].astype(str)
    return d[d.source == d.target]  # diagonal: finetune no próprio target


def agg(df, by):
    return df.groupby(by, observed=True)[["test_acc", "test_f1_macro"]].mean() * 100


# ---------------------------------------------------------------- planilha
def method_label(row):
    if row.method == "SL":
        return "SL (from scratch)"
    return f"{row.method} ({row.protocol})"


def sheet_resumo(df):
    full = df[df.n_shots == "full"].copy()
    full["Método"] = full.apply(method_label, axis=1)
    t = agg(full, ["Método", "setting"]).unstack("setting")
    t.columns = [f"{'Acc' if m == 'test_acc' else 'F1'} {SCEN_PRETTY[s].split(' ')[0]} (%)"
                 for m, s in t.columns]
    sl = t.loc["SL (from scratch)"]
    for c in [c for c in t.columns if c.startswith("Acc")]:
        t[c.replace("Acc", "Δ Acc vs SL,") + ""] = t[c] - sl[c]
    order = ["SL (from scratch)", "LFR (linear)", "LFR (finetune)",
             "TF-C (linear)", "TF-C (finetune)"]
    return t.loc[order].round(1)


def sheet_regimes(df):
    d = df[(df.protocol.isin(["sl", "finetune"])) & (df.setting != "comb")]
    t = agg(d, ["setting", "method", "n_shots"]).reset_index()
    t["Regime"] = t.n_shots.map(dict(zip(SHOT_ORDER, SHOT_LABELS)))
    piv = t.pivot_table(index=["setting", "method"], columns="Regime",
                        values="test_acc").reindex(columns=SHOT_LABELS)
    piv.index = [f"{SCEN_PRETTY[s].split(' ')[0]} — {m}" for s, m in piv.index]
    piv.columns = [f"Acc {c} (%)" for c in piv.columns]
    return piv.round(1)


def sheet_encoders(df, setting):
    d = df[(df.n_shots == "full") & (df.protocol.isin(["sl", "finetune"]))
           & (df.setting == setting)]
    t = agg(d, ["encoder", "method"])
    acc = t["test_acc"].unstack("method")[["SL", "LFR", "TF-C"]]
    f1 = t["test_f1_macro"].unstack("method")[["SL", "LFR", "TF-C"]]
    out = pd.DataFrame(index=[ENCODER_PRETTY[e] for e in acc.index])
    for m in ["SL", "LFR", "TF-C"]:
        out[f"Acc {m} (%)"] = acc[m].values
        out[f"F1 {m} (%)"] = f1[m].values
    for m in ["LFR", "TF-C"]:
        out[f"Δ Acc {m}−SL (pp)"] = out[f"Acc {m} (%)"] - out["Acc SL (%)"]
    return out.round(1)


def sheet_c2t(df, c2t):
    full = df[df.n_shots == "full"]
    esp = agg(full[(full.setting == "in") & (full.protocol == "finetune")],
              ["method", "target"])["test_acc"].rename("Especialista (%)")
    comb = agg(full[(full.setting == "comb") & (full.protocol == "finetune")],
               ["method", "target"])["test_acc"].rename("Combined (%)")
    c2 = agg(c2t[(c2t.n_shots == "full") & (c2t.protocol == "finetune")],
             ["method", "target"])["test_acc"].rename("Comb→target (%)")
    t = pd.concat([esp, c2, comb], axis=1)
    t["Δ pré-treino multi (pp)"] = t["Comb→target (%)"] - t["Especialista (%)"]
    t["Δ rótulos misturados (pp)"] = t["Combined (%)"] - t["Comb→target (%)"]
    return t.round(1)


def write_xlsx(sheets: dict):
    path = OUT / "resultados_ssl.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, t in sheets.items():
            t.to_excel(xw, sheet_name=name)
            ws = xw.sheets[name]
            ws.column_dimensions["A"].width = 30
            for col in ws.iter_cols(min_col=2):
                ws.column_dimensions[col[0].column_letter].width = 16
    print("wrote", path)


# ---------------------------------------------------------------- figuras
def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def fig_data_efficiency(df, ssl_name, fname, title):
    d = df[(df.setting == "in") & (df.method.isin(["SL", ssl_name]))
           & (df.protocol.isin(["sl", "finetune"]))]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    x = np.arange(4)
    for ax, enc in zip(axes, ["resnetse5", "cnnpff", "rnn", "tstcc"]):
        for m in ["SL", ssl_name]:
            y = (agg(d[(d.encoder == enc) & (d.method == m)], ["n_shots"])
                 ["test_acc"].reindex(SHOT_ORDER))
            ax.plot(x, y, marker="o", markersize=6, linewidth=2,
                    color=METHOD_COLORS[m], label="SL (do zero)" if m == "SL"
                    else f"{m} pré-treino + finetune")
        ax.set_xticks(x, SHOT_LABELS, fontsize=9)
        ax.set_title(ENCODER_PRETTY[enc], fontsize=11)
        style_ax(ax)
    axes[0].set_ylabel("Acurácia in-domain (%)")
    axes[0].legend(fontsize=9, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / fname)


def fig_cenarios(df):
    d = df[(df.n_shots == "full") & (df.protocol.isin(["sl", "finetune"]))]
    t = agg(d, ["setting", "method"])["test_acc"].unstack("method")
    t = t.loc[["in", "comb", "cross"], ["SL", "LFR", "TF-C"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x, w = np.arange(3), 0.26
    for i, m in enumerate(["SL", "LFR", "TF-C"]):
        bars = ax.bar(x + (i - 1) * w, t[m], w * 0.93, color=METHOD_COLORS[m],
                      label=m if m != "SL" else "SL (do zero)")
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
    ax.set_xticks(x, [SCEN_PRETTY[s] for s in t.index], fontsize=10)
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 92)
    ax.set_title("SL vs pré-treino SSL + finetune — 100% dos rótulos\n"
                 "(média de 4 encoders × 4 seeds; SSL em protocolo finetune)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ssl_vs_sl_cenarios.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_ssl_vs_sl_cenarios.png")


def fig_fewshot(df):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    x = np.arange(4)
    for ax, setting in zip(axes, ["in", "cross"]):
        d = df[(df.setting == setting) & (df.protocol.isin(["sl", "finetune"]))]
        for m in ["SL", "LFR", "TF-C"]:
            y = agg(d[d.method == m], ["n_shots"])["test_acc"].reindex(SHOT_ORDER)
            ax.plot(x, y, marker="o", markersize=6, linewidth=2,
                    color=METHOD_COLORS[m], label=m)
        ax.set_xticks(x, SHOT_LABELS, fontsize=9)
        ax.set_title(SCEN_PRETTY[setting], fontsize=11)
        style_ax(ax)
    axes[0].set_ylabel("Acurácia (%)")
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle("Eficiência de dados: SL vs SSL-finetune (média encoders × seeds)",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ssl_vs_sl_fewshot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_ssl_vs_sl_fewshot.png")


def fig_c2t(sheet):
    t = sheet.loc["TF-C"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x, w = np.arange(len(t)), 0.26
    cols = [("Especialista (%)", "#0173B2", "Especialista (pré-treino próprio)"),
            ("Comb→target (%)", "#029E73", "Pré-treino multi-domínio + finetune local"),
            ("Combined (%)", "#949494", "Combined (finetune com rótulos misturados)")]
    for i, (c, color, label) in enumerate(cols):
        bars = ax.bar(x + (i - 1) * w, t[c], w * 0.93, color=color, label=label)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, t.index, fontsize=9)
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 100)
    ax.set_title("TF-C: pré-treinar no corpus multi-domínio não degrada o especialista\n"
                 "(finetune, 100% dos rótulos; média 4 encoders × 4 seeds)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=3)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tfc_comb2target.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_tfc_comb2target.png")


def main():
    df = load()
    c2t = load_c2t()
    sheets = {
        "resumo_metodos": sheet_resumo(df),
        "regimes": sheet_regimes(df),
        "encoder_x_tecnica": sheet_encoders(df, "in"),
        "cross_encoder": sheet_encoders(df, "cross"),
        "comb2target": sheet_c2t(df, c2t),
    }
    write_xlsx(sheets)
    fig_data_efficiency(df, "LFR", "fig_lfr_data_efficiency.png",
                        "LFR vs SL — acurácia in-domain por regime de rótulos")
    fig_data_efficiency(df, "TF-C", "fig_tfc_data_efficiency.png",
                        "TF-C vs SL — acurácia in-domain por regime de rótulos")
    fig_cenarios(df)
    fig_fewshot(df)
    fig_c2t(sheets["comb2target"])


if __name__ == "__main__":
    main()
