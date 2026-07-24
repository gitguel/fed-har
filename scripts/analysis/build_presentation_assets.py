"""Gera os assets da apresentação (slides SSL LFR/TF-C + comparação SSL vs SL).

Lê apenas os caches de results/ e escreve em docs/apresentacao/:
  - resultados_ssl.xlsx (abas: resumo_metodos, regimes_finetune, regimes_linear,
    encoder_x_tecnica, cross_encoder, comb2target) — "média ± dp entre seeds" (2 casas)
  - tabelas/tab_resumo_metodos.{tex,pdf}, tabelas/tab_comb2target.{tex,pdf},
    tabelas/tab_regimes_finetune.{tex,pdf}, tabelas/tab_regimes_linear.{tex,pdf}
  - fig_lfr_data_efficiency.png, fig_tfc_data_efficiency.png,
    fig_tfc_comb2target.png, fig_ssl_vs_sl_cenarios.png, fig_ssl_vs_sl_fewshot.png
  - fig_transfer_matrix_acc.png (acurácia fonte×alvo por método),
    fig_transfer_matrix_delta.png (Δ SSL−SL fonte×alvo, LFR e TF-C)

As médias são sobre todos os runs agregados na célula (encoders × seeds ×
fontes); o desvio-padrão é ENTRE SEEDS (cada seed reduzida à sua média, dp
sobre as 4 seeds), com 2 casas decimais. Nos gráficos o desvio vira barra de
erro (gráficos de barras) ou faixa sombreada (gráficos de linha).

Rodar: poetry run python scripts/analysis/build_presentation_assets.py
(a compilação dos PDFs usa `tectonic` ou `pdflatex` se disponíveis no PATH)
"""

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../scripts

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "apresentacao"
TAB = OUT / "tabelas"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

DEC = 2  # casas decimais em toda a apresentação
NA = "—"  # marcador de "não se aplica" (planilha)
NA_TEX = "--"  # idem, na tabela LaTeX

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
SETTING_SHORT = {"in": "In-domain", "comb": "Combined", "cross": "Cross-domain"}
# ordem dos datasets nas matrizes de transfer (mesma de common.DATASETS)
DATASETS = ["UCI", "MotionSense", "KuHar", "WISDM", "RealWorld_thigh", "RealWorld_waist"]


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


def agg_std(df, by):
    # Desvio-padrão ENTRE SEEDS: cada seed é reduzida à sua média (sobre os
    # encoders/datasets/fontes agregados na célula) e o dp é tomado sobre as
    # médias por seed. Mede a variabilidade de treino, não a heterogeneidade
    # entre datasets/encoders. As médias reportadas (agg) não mudam.
    cols = ["test_acc", "test_f1_macro"]
    per_seed = df.groupby(list(by) + ["seed"], observed=True)[cols].mean()
    return per_seed.groupby(list(by), observed=True).std(ddof=1) * 100


# ------------------------------------------------------ formatação de células
def fmt_ms(mean, std, dec=DEC):
    """'média ± dp' para a planilha (str)."""
    if pd.isna(mean):
        return NA
    if pd.isna(std):
        return f"{mean:.{dec}f}"
    return f"{mean:.{dec}f} ± {std:.{dec}f}"


def method_label(row):
    if row.method == "SL":
        return "SL (from scratch)"
    return f"{row.method} ({row.protocol})"


# ----------------------------------------------------------------- planilha
def _resumo_frames(df):
    """Retorna (média, desvio) para métodos × cenário, no regime 100%."""
    full = df[df.n_shots == "full"].copy()
    full["Método"] = full.apply(method_label, axis=1)
    order = ["SL (from scratch)", "LFR (linear)", "LFR (finetune)",
             "TF-C (linear)", "TF-C (finetune)"]

    def relabel(t):
        t = t.unstack("setting")
        t.columns = [f"{'Acc' if m == 'test_acc' else 'F1'} {SETTING_SHORT[s]} (%)"
                     for m, s in t.columns]
        return t.reindex(order)

    tm = relabel(agg(full, ["Método", "setting"]))
    ts = relabel(agg_std(full, ["Método", "setting"]))
    return tm, ts


def _resumo_delta(tm):
    sl = tm.loc["SL (from scratch)"]
    delta = pd.DataFrame(index=tm.index)
    for s in ["Combined", "Cross-domain", "In-domain"]:
        c = f"Acc {s} (%)"
        delta[f"Δ Acc vs SL, {s} (%)"] = (tm[c] - sl[c])
    return delta


def sheet_resumo(df):
    tm, ts = _resumo_frames(df)
    delta = _resumo_delta(tm)
    out = pd.DataFrame(index=tm.index)
    for c in tm.columns:
        out[c] = [fmt_ms(mu, sd) for mu, sd in zip(tm[c], ts[c])]
    for c in delta.columns:
        out[c] = delta[c].round(DEC)
    out.index.name = "Método"
    return out


def _regime_frames(df, proto):
    """Média/desvio de acurácia por (setting, método, n_shots) para um protocolo.

    SL entra sempre do zero (protocolo 'sl'); LFR/TF-C no protocolo pedido
    ('finetune' ou 'linear'). Só cenários in-domain e cross-domain.
    """
    d = df[df.setting.isin(["in", "cross"])]
    d = d[((d.method == "SL") & (d.protocol == "sl"))
          | (d.method.isin(["LFR", "TF-C"]) & (d.protocol == proto))]
    by = ["setting", "method", "n_shots"]
    return agg(d, by)["test_acc"], agg_std(d, by)["test_acc"]


def sheet_regimes_proto(df, proto):
    m, s = _regime_frames(df, proto)
    rows, idx = [], []
    for st in ["in", "cross"]:
        sl = m.loc[(st, "SL")].reindex(SHOT_ORDER)
        for meth in ["SL", "LFR", "TF-C"]:
            mm = m.loc[(st, meth)].reindex(SHOT_ORDER)
            ss = s.loc[(st, meth)].reindex(SHOT_ORDER)
            rows.append([fmt_ms(a, b) for a, b in zip(mm, ss)])
            idx.append((SETTING_SHORT[st], meth))
        for meth in ["LFR", "TF-C"]:
            g = m.loc[(st, meth)].reindex(SHOT_ORDER) - sl
            rows.append([round(v, DEC) for v in g])
            idx.append((SETTING_SHORT[st], f"Δ {meth}−SL (pp)"))
    return pd.DataFrame(rows, columns=[f"Acc {l} (%)" for l in SHOT_LABELS],
                        index=pd.MultiIndex.from_tuples(idx, names=["Cenário", "Linha"]))


def sheet_encoders(df, setting):
    d = df[(df.n_shots == "full") & (df.protocol.isin(["sl", "finetune"]))
           & (df.setting == setting)]
    tm = agg(d, ["encoder", "method"])
    ts = agg_std(d, ["encoder", "method"])
    accm = tm["test_acc"].unstack("method")[["SL", "LFR", "TF-C"]]
    accs = ts["test_acc"].unstack("method")[["SL", "LFR", "TF-C"]]
    f1m = tm["test_f1_macro"].unstack("method")[["SL", "LFR", "TF-C"]]
    f1s = ts["test_f1_macro"].unstack("method")[["SL", "LFR", "TF-C"]]
    out = pd.DataFrame(index=[ENCODER_PRETTY[e] for e in accm.index])
    for m in ["SL", "LFR", "TF-C"]:
        out[f"Acc {m} (%)"] = [fmt_ms(a, b) for a, b in zip(accm[m], accs[m])]
        out[f"F1 {m} (%)"] = [fmt_ms(a, b) for a, b in zip(f1m[m], f1s[m])]
    for m in ["LFR", "TF-C"]:
        out[f"Δ Acc {m}−SL (pp)"] = (accm[m].values - accm["SL"].values).round(DEC)
    return out


def _c2t_frames(df, c2t, by=("method", "target")):
    """Retorna (média, desvio) de acurácia por `by`, regime 100%.

    `by=["method", "target"]` -> uma linha por dataset;
    `by=["method"]` -> agregado sobre todos os datasets (linha "Média").
    """
    by = list(by)
    full = df[df.n_shots == "full"]

    def mm(d):
        m = d.groupby(by, observed=True)["test_acc"].mean() * 100
        per_seed = d.groupby(by + ["seed"], observed=True)["test_acc"].mean()
        s = per_seed.groupby(by, observed=True).std(ddof=1) * 100  # dp entre seeds
        return m, s

    # SL entra em Especialista (in-domain do zero) e Combined (treino no mix);
    # não há "Comb->alvo" supervisionado -> fica NaN (protocolo "sl").
    esp_m, esp_s = mm(full[(full.setting == "in")
                           & (full.protocol.isin(["finetune", "sl"]))])
    c2_m, c2_s = mm(c2t[(c2t.n_shots == "full") & (c2t.protocol == "finetune")])
    cmb_m, cmb_s = mm(full[(full.setting == "comb")
                           & (full.protocol.isin(["finetune", "sl"]))])
    cols = ["Especialista (%)", "Comb→target (%)", "Combined (%)"]
    mean = pd.concat([esp_m, c2_m, cmb_m], axis=1, keys=cols)
    std = pd.concat([esp_s, c2_s, cmb_s], axis=1, keys=cols)
    return mean, std


def _delta_col(series):
    return [NA if pd.isna(v) else round(v, DEC) for v in series]


def _local_ref(mean):
    """Base 'local limpa' de cada método: Comb->alvo (SSL) ou Especialista (SL)."""
    return mean["Comb→target (%)"].fillna(mean["Especialista (%)"])


def _c2t_format(mean, std, sl_esp):
    # sl_esp: Especialista do SL no mesmo alvo (array alinhado às linhas de `mean`)
    out = pd.DataFrame(index=mean.index)
    for c in mean.columns:
        out[c] = [fmt_ms(mu, sd) for mu, sd in zip(mean[c], std[c])]
    out["Δ vs SL (pp)"] = _delta_col(mean["Especialista (%)"].to_numpy(float)
                                     - np.asarray(sl_esp, dtype=float))
    out["Δ pré-treino multi (pp)"] = _delta_col(mean["Comb→target (%)"]
                                                - mean["Especialista (%)"])
    out["Δ rótulos misturados (pp)"] = _delta_col(mean["Combined (%)"]
                                                  - _local_ref(mean))
    return out


def sheet_c2t(df, c2t):
    mean, std = _c2t_frames(df, c2t)
    mean_m, std_m = _c2t_frames(df, c2t, by=["method"])
    sl_esp = mean.loc["SL", "Especialista (%)"]        # Especialista do SL por alvo
    sl_esp_all = mean_m.loc["SL", "Especialista (%)"]  # idem, agregado
    blocks = []
    for method in ["SL", "LFR", "TF-C"]:
        bm, bs = mean.loc[[method]], std.loc[[method]]
        ref = sl_esp.reindex(bm.index.get_level_values("target")).to_numpy(float)
        blocks.append(_c2t_format(bm, bs, ref))
        avg = _c2t_format(mean_m.loc[[method]], std_m.loc[[method]], [sl_esp_all])
        avg.index = pd.MultiIndex.from_tuples([(method, "Média (todos)")])
        blocks.append(avg)
    out = pd.concat(blocks)
    out.index.names = ["method", "target"]
    return out


def write_xlsx(sheets: dict):
    path = OUT / "resultados_ssl.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, t in sheets.items():
            t.to_excel(xw, sheet_name=name)
            ws = xw.sheets[name]
            ws.column_dimensions["A"].width = 30
            for col in ws.iter_cols(min_col=2):
                ws.column_dimensions[col[0].column_letter].width = 18
            # colunas Δ ficam numéricas -> força 2 casas na exibição
            for row in ws.iter_rows(min_row=2, min_col=2):
                for cell in row:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = "0.00"
    print("wrote", path)


# --------------------------------------------------------------- tabelas TeX
def tex_num(x, dec=DEC):
    return f"{x:.{dec}f}".replace(".", "{,}")


def tex_signed(x, dec=DEC):
    if pd.isna(x):
        return NA_TEX
    r = round(x, dec)
    if r > 0:
        return "$+$" + tex_num(abs(r), dec)
    if r < 0:
        return "$-$" + tex_num(abs(r), dec)
    return tex_num(0.0, dec)  # zero exato: sem sinal


def tex_cell(mean, std, bold=False):
    if pd.isna(mean):
        return NA_TEX
    if pd.isna(std):
        s = tex_num(mean)
    else:
        s = f"{tex_num(mean)} $\\pm$ {tex_num(std)}"
    return r"\textbf{" + s + "}" if bold else s


RESUMO_TEMPLATE = r"""\documentclass[border=8pt,varwidth=\maxdimen]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{newtxtext,newtxmath}
\begin{document}

\centering
{\large\bfseries Resumo dos métodos: SL vs.\ SSL}\\[2pt]
{\small Acurácia e F1-macro (\%): média $\pm$ dp entre seeds (2 casas). \textbf{Negrito} = melhor por coluna.}\\[6pt]
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{Acurácia (\%)} & \multicolumn{3}{c}{F1-macro (\%)} & \multicolumn{3}{c}{$\Delta$ Acc vs.\ SL (pp)}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Método & Comb. & Cross & In-dom. & Comb. & Cross & In-dom. & Comb. & Cross & In-dom.\\
\midrule
<<BODY>>
\bottomrule
\end{tabular}

\end{document}
"""

RESUMO_TEX_NAME = {
    "SL (from scratch)": "SL (from scratch)",
    "LFR (linear)": "LFR (linear)",
    "LFR (finetune)": "LFR (fine-tune)",
    "TF-C (linear)": "TF-C (linear)",
    "TF-C (finetune)": "TF-C (fine-tune)",
}


def write_tex_resumo(df):
    tm, ts = _resumo_frames(df)
    delta = _resumo_delta(tm)
    settings = ["Combined", "Cross-domain", "In-domain"]
    metric_cols = [f"Acc {s} (%)" for s in settings] + [f"F1 {s} (%)" for s in settings]
    best = {c: tm[c].idxmax() for c in metric_cols}
    best_d = {c: delta[c].idxmax() for c in delta.columns}
    rows = []
    for m in tm.index:
        cells = [tex_cell(tm.loc[m, c], ts.loc[m, c], bold=best[c] == m)
                 for c in metric_cols]
        for c in delta.columns:
            s = tex_signed(delta.loc[m, c])
            cells.append(r"\textbf{" + s + "}" if best_d[c] == m else s)
        rows.append(f"{RESUMO_TEX_NAME[m]:<18}& " + " & ".join(cells) + r"\\")
    tex = RESUMO_TEMPLATE.replace("<<BODY>>", "\n".join(rows))
    path = TAB / "tab_resumo_metodos.tex"
    path.write_text(tex, encoding="utf-8")
    print("wrote", path)
    compile_tex(path)


C2T_TEMPLATE = r"""\documentclass[border=8pt,varwidth=\maxdimen]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs,multirow}
\usepackage{newtxtext,newtxmath}
\begin{document}

\centering
{\large\bfseries Pré-treino multi-dataset vs.\ especialista}\\[2pt]
{\scriptsize Acurácia (\%), média $\pm$ dp entre seeds. \textbf{Negrito} = melhor por linha; -- = n/a.}\\[6pt]
\begin{tabular}{llcccccc}
\toprule
& & \multicolumn{3}{c}{Acurácia (\%)} & \multicolumn{3}{c}{$\Delta$ (pp)}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Método & Alvo & Especialista & Comb$\to$alvo & Combined & vs.\ SL & Pré-treino multi & Rótulos mist.\\
\midrule
<<BODY>>
\bottomrule
\end{tabular}

\end{document}
"""


def write_tex_comb2target(df, c2t):
    mean, std = _c2t_frames(df, c2t)
    mean_m, std_m = _c2t_frames(df, c2t, by=["method"])
    acc_cols = ["Especialista (%)", "Comb→target (%)", "Combined (%)"]
    sl_esp = mean.loc["SL", "Especialista (%)"]        # Especialista do SL por alvo
    sl_esp_all = mean_m.loc["SL", "Especialista (%)"]  # idem, agregado

    def acc_delta_cells(m, s, sl_ref):
        best = m[acc_cols].idxmax()
        cells = [tex_cell(m[c], s[c], bold=(c == best)) for c in acc_cols]
        local = m["Comb→target (%)"] if pd.notna(m["Comb→target (%)"]) \
            else m["Especialista (%)"]
        cells += [tex_signed(m["Especialista (%)"] - sl_ref),
                  tex_signed(m["Comb→target (%)"] - m["Especialista (%)"]),
                  tex_signed(m["Combined (%)"] - local)]
        return cells

    blocks = []
    for method in ["SL", "LFR", "TF-C"]:
        mm, ss = mean.loc[method], std.loc[method]
        targets = sorted(mm.index)
        lines = [f"\\multirow{{{len(targets) + 1}}}{{*}}{{{method}}}"]
        for tgt in targets:
            cells = acc_delta_cells(mm.loc[tgt], ss.loc[tgt], sl_esp[tgt])
            tgt_tex = tgt.replace("_", r"\_")
            lines.append(f" & {tgt_tex:<16}& " + " & ".join(cells) + r"\\")
        # linha-resumo: média sobre todos os datasets do método
        avg = acc_delta_cells(mean_m.loc[method], std_m.loc[method], sl_esp_all)
        lines.append(r"\cmidrule(lr){2-8}")
        lines.append(r" & \textit{Média}    & " + " & ".join(avg) + r"\\")
        blocks.append("\n".join(lines))
    body = "\n\\midrule\n".join(blocks)
    tex = C2T_TEMPLATE.replace("<<BODY>>", body)
    path = TAB / "tab_comb2target.tex"
    path.write_text(tex, encoding="utf-8")
    print("wrote", path)
    compile_tex(path)


REGIMES_TEMPLATE = r"""\documentclass[border=8pt,varwidth=\maxdimen]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs,multirow}
\usepackage{newtxtext,newtxmath}
\begin{document}

\centering
{\large\bfseries Eficiência de dados — <<SUB>>}\\[2pt]
{\scriptsize Acurácia (\%), média $\pm$ dp entre seeds, por regime de rótulos. \textbf{Negrito} = melhor por coluna; $\Delta$ = ganho vs.\ SL (pp).}\\[6pt]
\begin{tabular}{llcccc}
\toprule
& & \multicolumn{4}{c}{Acurácia (\%)}\\
\cmidrule(lr){3-6}
Cenário & Método & 1-shot & 10-shot & 100-shot & 100\%\\
\midrule
<<BODY>>
\bottomrule
\end{tabular}

\end{document}
"""


def write_tex_regimes(df, proto, sub, fname):
    m, s = _regime_frames(df, proto)
    blocks = []
    for st in ["in", "cross"]:
        sl = m.loc[(st, "SL")].reindex(SHOT_ORDER)
        accs = {mth: m.loc[(st, mth)].reindex(SHOT_ORDER) for mth in ["SL", "LFR", "TF-C"]}
        best = {sh: max(["SL", "LFR", "TF-C"], key=lambda mth: accs[mth][sh])
                for sh in SHOT_ORDER}
        lines = [f"\\multirow{{5}}{{*}}{{{SETTING_SHORT[st]}}}"]
        for mth in ["SL", "LFR", "TF-C"]:
            mm = accs[mth]
            ss = s.loc[(st, mth)].reindex(SHOT_ORDER)
            cells = [tex_cell(mm[sh], ss[sh], bold=(best[sh] == mth)) for sh in SHOT_ORDER]
            lines.append(f" & {mth:<10}& " + " & ".join(cells) + r"\\")
        lines.append(r"\cmidrule(lr){2-6}")
        for mth in ["LFR", "TF-C"]:
            g = accs[mth] - sl
            cells = [tex_signed(g[sh]) for sh in SHOT_ORDER]
            lines.append(rf" & \textit{{$\Delta$ {mth}$-$SL}} & " + " & ".join(cells) + r"\\")
        blocks.append("\n".join(lines))
    tex = (REGIMES_TEMPLATE.replace("<<SUB>>", sub)
           .replace("<<BODY>>", "\n\\midrule\n".join(blocks)))
    path = TAB / fname
    path.write_text(tex, encoding="utf-8")
    print("wrote", path)
    compile_tex(path)


def compile_tex(path):
    tectonic = shutil.which("tectonic")
    try:
        if tectonic:
            subprocess.run([tectonic, "-X", "compile", "--outdir", str(path.parent),
                            str(path)], check=True, capture_output=True, text=True)
        elif shutil.which("pdflatex"):
            subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-output-directory", str(path.parent), str(path)],
                           check=True, capture_output=True, text=True, cwd=path.parent)
            for ext in (".aux", ".log"):
                path.with_suffix(ext).unlink(missing_ok=True)
        else:
            print("  (nenhum engine LaTeX no PATH; PDF de", path.name, "não gerado)")
            return
        print("wrote", path.with_suffix(".pdf"))
    except subprocess.CalledProcessError as e:
        print("  ERRO compilando", path.name, "\n", (e.stderr or e.stdout or "")[-800:])


# ---------------------------------------------------------------- figuras
def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def bar_labels(ax, bars):
    """Rótulos de valor (2 casas) no centro das barras, compatível c/ barra de erro."""
    ax.bar_label(bars, fmt=f"%.{DEC}f", label_type="center", rotation=90,
                 color="white", fontsize=7)


def fig_data_efficiency(df, ssl_name, fname, title):
    d = df[(df.setting == "in") & (df.method.isin(["SL", ssl_name]))
           & (df.protocol.isin(["sl", "finetune"]))]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    x = np.arange(4)
    for ax, enc in zip(axes, ["resnetse5", "cnnpff", "rnn", "tstcc"]):
        for m in ["SL", ssl_name]:
            sub = d[(d.encoder == enc) & (d.method == m)]
            y = agg(sub, ["n_shots"])["test_acc"].reindex(SHOT_ORDER)
            s = agg_std(sub, ["n_shots"])["test_acc"].reindex(SHOT_ORDER)
            ax.plot(x, y, marker="o", markersize=6, linewidth=2,
                    color=METHOD_COLORS[m], label="SL (do zero)" if m == "SL"
                    else f"{m} pré-treino + finetune")
            ax.fill_between(x, (y - s).to_numpy(dtype=float),
                            (y + s).to_numpy(dtype=float),
                            color=METHOD_COLORS[m], alpha=0.15, linewidth=0)
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
    rows, cols = ["in", "comb", "cross"], ["SL", "LFR", "TF-C"]
    tm = agg(d, ["setting", "method"])["test_acc"].unstack("method").loc[rows, cols]
    ts = agg_std(d, ["setting", "method"])["test_acc"].unstack("method").loc[rows, cols]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x, w = np.arange(3), 0.26
    for i, m in enumerate(cols):
        bars = ax.bar(x + (i - 1) * w, tm[m], w * 0.93, color=METHOD_COLORS[m],
                      yerr=ts[m], capsize=2, error_kw=dict(elinewidth=0.8, alpha=0.6),
                      label=m if m != "SL" else "SL (do zero)")
        bar_labels(ax, bars)
    ax.set_xticks(x, [SCEN_PRETTY[s] for s in tm.index], fontsize=10)
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 100)
    ax.set_title("SL vs pré-treino SSL + finetune — 100% dos rótulos\n"
                 "(média de 4 encoders × 4 seeds; erro = dp entre seeds; SSL finetune)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ssl_vs_sl_cenarios.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_ssl_vs_sl_cenarios.png")


def fig_fewshot(df, ssl_protocol="finetune", fname="fig_ssl_vs_sl_fewshot.png",
                subtitle="SL vs SSL-finetune"):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    x = np.arange(4)
    for ax, setting in zip(axes, ["in", "cross"]):
        methods = ["SL", "LFR", "TF-C"] if ssl_protocol == "finetune" else ["LFR", "TF-C"]
        d = df[(df.setting == setting) & (df.protocol == ssl_protocol)
               | (df.setting == setting) & (df.protocol == "sl") & (df.method == "SL")]
        for m in methods:
            sub = d[d.method == m]
            y = agg(sub, ["n_shots"])["test_acc"].reindex(SHOT_ORDER)
            s = agg_std(sub, ["n_shots"])["test_acc"].reindex(SHOT_ORDER)
            label = m if m == "SL" or ssl_protocol == "finetune" else f"{m} (linear)"
            ax.plot(x, y, marker="o", markersize=6, linewidth=2,
                    color=METHOD_COLORS[m], label=label)
            ax.fill_between(x, (y - s).to_numpy(dtype=float),
                            (y + s).to_numpy(dtype=float),
                            color=METHOD_COLORS[m], alpha=0.15, linewidth=0)
        ax.set_xticks(x, SHOT_LABELS, fontsize=9)
        ax.set_title(SCEN_PRETTY[setting], fontsize=11)
        style_ax(ax)
    axes[0].set_ylabel("Acurácia (%)")
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle(f"Eficiência de dados: {subtitle} (média encoders × seeds; faixa = dp entre seeds)",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / fname)


def fig_c2t(mean, std, method="TF-C", fname="fig_tfc_comb2target.png",
            headline="pré-treinar no corpus multi-domínio não degrada o especialista"):
    tm = mean.loc[method]
    ts = std.loc[method]
    tm = tm.loc[sorted(tm.index)]
    ts = ts.loc[tm.index]
    fig, ax = plt.subplots(figsize=(9, 4))
    x, w = np.arange(len(tm)), 0.26
    cols = [("Especialista (%)", "#0173B2", "Especialista"),
            ("Comb→target (%)", "#029E73", "Pré-treino multi + finetune local"),
            ("Combined (%)", "#949494", "Finetune misturado (combined)")]
    for i, (c, color, label) in enumerate(cols):
        bars = ax.bar(x + (i - 1) * w, tm[c], w * 0.93, color=color, label=label,
                      yerr=ts[c], capsize=2, error_kw=dict(elinewidth=0.8, alpha=0.6))
        bar_labels(ax, bars)
    ax.set_xticks(x, tm.index, fontsize=9)
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{method}: {headline}\n"
                 "(finetune, 100%; média de 4 encoders × 4 seeds; erro = dp entre seeds)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=3)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / fname)


def fig_cross_by_target(df):
    full = df[(df.n_shots == "full") & (df.protocol.isin(["sl", "finetune"]))]
    cross_m = (agg(full[full.setting == "cross"], ["target", "method"])["test_acc"]
               .unstack("method")[["SL", "LFR", "TF-C"]])
    cross_s = (agg_std(full[full.setting == "cross"], ["target", "method"])["test_acc"]
               .unstack("method")[["SL", "LFR", "TF-C"]])
    esp = agg(full[(full.setting == "in") & (full.method == "SL")],
              ["target"])["test_acc"].reindex(cross_m.index)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x, w = np.arange(len(cross_m)), 0.26
    for i, m in enumerate(["SL", "LFR", "TF-C"]):
        bars = ax.bar(x + (i - 1) * w, cross_m[m], w * 0.93, color=METHOD_COLORS[m],
                      yerr=cross_s[m], capsize=2,
                      error_kw=dict(elinewidth=0.8, alpha=0.6),
                      label=m if m != "SL" else "SL (do zero)")
        bar_labels(ax, bars)
    ax.hlines(esp, x - 1.5 * w, x + 1.5 * w, color="#333333", linewidth=1.6,
              linestyle="--", label="Nível in-domain (SL especialista)")
    ax.set_xticks(x, cross_m.index, fontsize=9)
    ax.set_ylabel("Acurácia no target (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Transfer cross-domain: treinar em outro domínio derruba o desempenho\n"
                 "(média das 5 fontes ≠ target × 4 encoders × 4 seeds; "
                 "erro = dp entre seeds; finetune, 100%)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=4)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_domain_por_target.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_cross_domain_por_target.png")


# ------------------------------------------ matrizes de transfer fonte × alvo
_MASK = np.eye(len(DATASETS), dtype=bool)  # diagonal (in-domain) mascarada


def _mat_ms(d):
    """Matriz fonte×alvo: (média, dp entre seeds) da acurácia (%)."""
    d = d[d.source.isin(DATASETS)]
    m = (d.groupby(["source", "target"])["test_acc"].mean() * 100).unstack()
    per_seed = d.groupby(["source", "target", "seed"])["test_acc"].mean()
    s = (per_seed.groupby(["source", "target"]).std(ddof=1) * 100).unstack()
    ix = dict(index=DATASETS, columns=DATASETS)
    return m.reindex(**ix), s.reindex(**ix)


def _delta_seed(df, method, protocol, n_shots):
    """Δ(método − SL) por (fonte, alvo, seed): média sobre encoders e depois diferença."""
    def per_seed(mth, proto):
        d = df[(df.method == mth) & (df.protocol == proto) & (df.n_shots == n_shots)
               & (df.source.isin(DATASETS))]
        return d.groupby(["source", "target", "seed"])["test_acc"].mean()
    return (per_seed(method, protocol) - per_seed("SL", "sl")) * 100


def _annot(m, s, delta=False):
    a = np.full(m.shape, "", dtype=object)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            mv, sv = m.iat[i, j], s.iat[i, j]
            if pd.notna(mv):
                head = f"{mv:+.1f}" if delta else f"{mv:.1f}"
                a[i, j] = head + (f"\n±{sv:.1f}" if pd.notna(sv) else "")
    return a


def _tick_axes(ax, first):
    ax.set_xlabel("Target (alvo)")
    ax.set_ylabel("Source (fonte)" if first else "")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.tick_params(labelsize=8)


def fig_transfer_matrix(df, protocol="finetune", n_shots="full"):
    """Acurácia de transfer out-domain fonte→alvo, por método (SL, LFR, TF-C)."""
    series = [("SL", "sl"), ("LFR", protocol), ("TF-C", protocol)]
    mats = [_mat_ms(df[(df.method == mth) & (df.protocol == pr) & (df.n_shots == n_shots)])
            for mth, pr in series]
    vals = np.concatenate([np.asarray(m, float)[~_MASK] for m, _ in mats])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for ax, (mth, _), (m, s) in zip(axes, series, mats):
        sns.heatmap(m, mask=_MASK, annot=_annot(m, s), fmt="", cmap="viridis",
                    vmin=vmin, vmax=vmax, square=True, linewidths=0.5, ax=ax,
                    cbar=(ax is axes[-1]), cbar_kws={"label": "Acurácia (%)"},
                    annot_kws={"fontsize": 7})
        ax.set_title(mth, fontsize=12)
        _tick_axes(ax, ax is axes[0])
    fig.suptitle("Transfer out-domain fonte → alvo — acurácia (%), finetune @ 100% dos rótulos\n"
                 "(média ± dp entre seeds; 4 encoders × 4 seeds; diagonal in-domain omitida)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig_transfer_matrix_acc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_transfer_matrix_acc.png")


def fig_transfer_delta(df, protocol="finetune", n_shots="full"):
    """Δ(SSL − SL) no transfer out-domain fonte→alvo, para LFR e TF-C."""
    mats = []
    for mth in ["LFR", "TF-C"]:
        ds = _delta_seed(df, mth, protocol, n_shots)
        ix = dict(index=DATASETS, columns=DATASETS)
        m = ds.groupby(["source", "target"]).mean().unstack().reindex(**ix)
        s = ds.groupby(["source", "target"]).std(ddof=1).unstack().reindex(**ix)
        mats.append((mth, m, s))
    peak = np.nanmax([np.abs(np.asarray(m, float)[~_MASK]).max() for _, m, _ in mats])
    vmax = max(5.0, np.ceil(peak / 5) * 5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (mth, m, s) in zip(axes, mats):
        sns.heatmap(m, mask=_MASK, annot=_annot(m, s, delta=True), fmt="", cmap="RdBu",
                    center=0, vmin=-vmax, vmax=vmax, square=True, linewidths=0.5, ax=ax,
                    cbar=(ax is axes[-1]), cbar_kws={"label": "Δ pp"},
                    annot_kws={"fontsize": 7})
        ax.set_title(f"Δ({mth} − SL)", fontsize=12)
        _tick_axes(ax, ax is axes[0])
    fig.suptitle("Ganho do SSL no transfer out-domain — Δ(SSL − SL) fonte → alvo, finetune @ 100%\n"
                 "(azul = SSL melhor; média ± dp entre seeds; 4 encoders × 4 seeds; diagonal omitida)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig_transfer_matrix_delta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_transfer_matrix_delta.png")


def main():
    df = load()
    c2t = load_c2t()
    sheets = {
        "resumo_metodos": sheet_resumo(df),
        "regimes_finetune": sheet_regimes_proto(df, "finetune"),
        "regimes_linear": sheet_regimes_proto(df, "linear"),
        "encoder_x_tecnica": sheet_encoders(df, "in"),
        "cross_encoder": sheet_encoders(df, "cross"),
        "comb2target": sheet_c2t(df, c2t),
    }
    write_xlsx(sheets)
    write_tex_resumo(df)
    write_tex_comb2target(df, c2t)
    write_tex_regimes(df, "finetune", "full fine-tuning", "tab_regimes_finetune.tex")
    write_tex_regimes(df, "linear", "linear readout", "tab_regimes_linear.tex")
    fig_data_efficiency(df, "LFR", "fig_lfr_data_efficiency.png",
                        "LFR vs SL — acurácia in-domain por regime de rótulos")
    fig_data_efficiency(df, "TF-C", "fig_tfc_data_efficiency.png",
                        "TF-C vs SL — acurácia in-domain por regime de rótulos")
    fig_cenarios(df)
    fig_cross_by_target(df)
    fig_fewshot(df)
    fig_fewshot(df, ssl_protocol="linear",
                fname="fig_ssl_vs_sl_fewshot_linear.png",
                subtitle="LFR vs TF-C sob linear readout")
    c2t_mean, c2t_std = _c2t_frames(df, c2t)
    fig_c2t(c2t_mean, c2t_std)
    fig_c2t(c2t_mean, c2t_std, method="LFR", fname="fig_lfr_comb2target.png",
            headline="pré-treino multi-domínio ~neutro no próprio domínio")
    fig_transfer_matrix(df)
    fig_transfer_delta(df)


if __name__ == "__main__":
    main()
