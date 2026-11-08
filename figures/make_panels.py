# -*- coding: utf-8 -*-
"""Render every panel of Fig 1-6 / Fig S1 as a STANDALONE figure (not a crop).

How it works
------------
Each figX_make.py has two parts separated by a "---- figure ----" marker:
  (1) data preparation  -> executed once into a namespace
  (2) figure section    -> split into per-panel blocks by their comment markers

For each panel we create a fresh figure sized for a single panel, bind the
axes names that the block uses, and re-execute only that block.  So every
output is genuinely re-drawn at its own canvas size -- axes, ticks, legends
and colour bars are laid out for that panel alone (no pixel cropping).

Per panel we also export the numbers actually plotted:
  panels/data/figX_panelY.csv        tidy table behind the panel
  panels/figX_panels_data.xlsx       one sheet per panel

Usage:  python make_panels.py fig1 [fig2 ...]      (run in this directory)
"""
from pathlib import Path
import runpy
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DPI = 600

# --------------------------------------------------------------------------- #
# panel specifications
# --------------------------------------------------------------------------- #
FIG1_BOX = '''
def box(ax, x, y, w, h, text, fc="#EAF2FA", ec="#2E5E8C", fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
'''

SPEC = {
    "fig1": {
        "script": "fig1/code/fig1_make.py",
        "split": "# ---------------- figure ----------------",
        "pre": FIG1_BOX,
        "panels": [
            dict(letter="A", axes=["axA"], marker="# --- Panel A: cohort, analysis set and centre-based split ---",
                 figsize=(7.2, 3.8), axis_off=True,
                 pre_axes="fig.subplots_adjust(left=0.005,right=0.995,top=0.96,bottom=0.02)",
                 data=lambda ns: pd.DataFrame([
                     dict(item="cohort", value=f"{len(ns['df'])}"),
                     dict(item="centres", value="13"),
                     dict(item="proteins_assayed", value="883"),
                     dict(item="proteins_excluded", value="0"),
                     dict(item="events_28d", value=ns["ev"]["28-day"]["composite"]),
                     dict(item="events_90d", value=ns["ev"]["90-day"]["composite"]),
                     dict(item="events_180d", value=ns["ev"]["6-month"]["composite"]),
                     dict(item="events_1y", value=ns["ev"]["1-year"]["composite"]),
                     dict(item="deaths_90d", value=ns["ev"]["90-day"]["death"]),
                     dict(item="transplants_90d", value=ns["ev"]["90-day"]["tx"]),
                     dict(item="no_ACLF", value=int((ns["df"][ns["col_aclf"]] == 0).sum())),
                     dict(item="ACLF_on_admission", value=int((ns["df"][ns["col_aclf"]] == 1).sum())),
                     dict(item="developed_ACLF", value=int((ns["df"][ns["col_aclf"]] == 2).sum())),
                     dict(item="cirrhosis_yes", value=int((ns["df"]["cirrhosis"] == 1).sum())),
                     dict(item="cirrhosis_no", value=int((ns["df"]["cirrhosis"] == 0).sum())),
                     dict(item="HBV_yes", value=int((ns["df"]["HBV"] == 1).sum())),
                     dict(item="HBV_no", value=int((ns["df"]["HBV"] == 0).sum())),
                 ])),
            dict(letter="B", axes=["axB"], marker="# --- Panel B: pre-specified analysis architecture ---",
                 figsize=(7.4, 3.6), axis_off=True,
                 pre_axes="fig.subplots_adjust(left=0.01,right=0.99,top=0.96,bottom=0.04)",
                 data=lambda ns: pd.DataFrame([
                     dict(aim=lane[0], step=i + 1, step_text=s.replace("\n", " "), figure=lane[4])
                     for lane in ns["LANES"] for i, s in enumerate(lane[3])])),
            dict(letter="C", axes=["axC"],
                 marker="# --- Panel C: design benchmarking against the two metabolomics papers ---",
                 figsize=(7.6, 4.4), axis_off=True,
                 data=lambda ns: pd.DataFrame(ns["BENCH"], columns=ns["BENCH"][0]).iloc[1:]),
            dict(letter="D", axes=["axD"], marker="# --- Panel D: Table 1 ---",
                 figsize=(7.2, 5.0), axis_off=True,
                 data=lambda ns: ns["t1"]),
        ],
    },
    "fig2": {
        "script": "fig2/code/fig2_make.py",
        "split": "# ---------------- figure ----------------",
        "pre": "",
        "panels": [
            dict(letter="A", axes=["axA1", "axA2"], twin=None,
                 sharey_pair=("axA1", "axA2"), marker="# A: PCA before/after",
                 figsize=(6.6, 3.0),
                 data=lambda ns: pd.DataFrame(dict(
                     sample=np.arange(len(ns["sc_raw"])),
                     group=ns["grp_col"],
                     PC1_before=ns["sc_raw"][:, 0], PC2_before=ns["sc_raw"][:, 1],
                     PC1_after=ns["sc_adj"][:, 0], PC2_after=ns["sc_adj"][:, 1])),
                 pre_axes='plt.setp(axA2.get_yticklabels(), visible=False)'),
            dict(letter="B", axes=["axB"], marker="# B: volcano", figsize=(3.9, 3.3),
                 data=lambda ns: ns["vol"][["protein", "beta", "p", "FDR"]].assign(
                     significant=lambda d: d["FDR"] < 0.05)),
            dict(letter="C", axes=["axC"], marker="# C: gradient heatmap", figsize=(3.8, 3.6),
                 data=lambda ns: ns["gmeans"][ns["top30"]].T.rename_axis("protein").reset_index()),
            dict(letter="D", axes=["axD"], marker="# D: module-trait heatmap", figsize=(4.4, 3.4),
                 data=lambda ns: ns["mt"][["module", "n", "hubs", "trend_r"] + list(ns["traits"].columns)]),
        ],
    },
    "fig3": {
        "script": "fig3/code/fig3_make.py",
        "split": "# ---------------- figure ----------------",
        "pre": "",
        "panels": [
            dict(letter="A", axes=["axA"], marker="# A forest", figsize=(5.4, 3.7),
                 data=lambda ns: ns["cox"].nsmallest(14, "p").iloc[::-1]
                 .merge(ns["annot"], on="protein", how="left")
                 [["protein", "geneID", "HR", "beta", "se", "p", "FDR"]]
                 .assign(CI_low=lambda d: np.exp(d["beta"] - 1.96 * d["se"]),
                         CI_high=lambda d: np.exp(d["beta"] + 1.96 * d["se"]))),
            dict(letter="B", axes=["axB"], marker="# B RCS", figsize=(4.8, 3.5),
                 data=lambda ns: pd.concat([
                     pd.DataFrame(dict(protein=p, z=xs, HR=hr, CI_low=lo, CI_high=hi,
                                       P_nonlinear=[r for r in ns["rcs_rows"]
                                                    if r["protein"] == p][0]["nonlin_p"]))
                     for p, (xs, hr, lo, hi) in ns["fig_rcs"].items()], ignore_index=True)),
            dict(letter="C", axes=["axC"], marker="# C Cox vs Fine-Gray", figsize=(4.2, 3.9),
                 data=lambda ns: ns["fg"].dropna().merge(ns["cox"][["protein", "HR"]], on="protein")
                 [["protein", "HR", "fg_HR"]]),
            dict(letter="D", axes=["axD"], marker="# D enrichment", figsize=(4.8, 3.3),
                 data=lambda ns: ns["enr"].nsmallest(10, "Adjusted P-value").iloc[::-1]
                 [["Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]]
                 if not ns["enr"].empty else pd.DataFrame()),
        ],
    },
    "fig4": {
        "script": "fig4/code/fig4_make.py",
        "split": "# ---------------- figure ----------------",
        "pre": "",
        "panels": [
            dict(letter="A", axes=["axA"], marker="# --- A: LASSO path ---",
                 figsize=(4.6, 3.6),
                 data=lambda ns: ns["path_df"]),
            dict(letter="B", axes=["axB"], marker="# --- B: AUC90 across models ---",
                 figsize=(5.2, 3.1),
                 data=lambda ns: ns["met"].assign(
                     dAUC_lo=lambda d: d.model.map(
                         {"MELD+3P": ns["ci"]["3vM"][0], "MELD+5P": ns["ci"]["5vM"][0]}),
                     dAUC_hi=lambda d: d.model.map(
                         {"MELD+3P": ns["ci"]["3vM"][1], "MELD+5P": ns["ci"]["5vM"][1]}),
                     p_dAUC=lambda d: d.model.map(
                         {"MELD+3P": ns["ci"]["3vM"][2], "MELD+5P": ns["ci"]["5vM"][2]}))),
            dict(letter="C", axes=["axC"], marker="# --- C: HR derivation vs validation ---",
                 figsize=(4.6, 3.4), data=lambda ns: ns["hr"]),
            dict(letter="D", axes=["axD"], marker="# --- D: calibration + metrics ---",
                 figsize=(4.0, 3.3),
                 data=lambda ns: ns["cal"].reset_index().assign(
                     c_index=float(ns["met"].loc[ns["met"].model == "MELD+5P", "c_validation"].iloc[0]),
                     meld_c_index=float(ns["meld_c"]),
                     auc90=float(ns["met"].loc[ns["met"].model == "MELD+5P", "auc90"].iloc[0]),
                     dAUC_lo=ns["ci"]["5vM"][0], dAUC_hi=ns["ci"]["5vM"][1],
                     nri=float(ns["nri5"]), idi=float(ns["idi5"]),
                     brier90=float(ns["met"].loc[ns["met"].model == "MELD+5P", "brier90"].iloc[0]))),
        ],
    },
    "fig5": {
        "script": "fig5/code/fig5_make.py",
        "split": "# ---------------- figure ----------------",
        "pre": "",
        "panels": [
            dict(letter="A", axes=["axA"], marker="# A grouped bars per center",
                 figsize=(7.0, 3.7), data=lambda ns: ns["loco"]),
            dict(letter="B", axes=["axB"], marker="# B forest", figsize=(4.8, 3.2),
                 data=lambda ns: ns["sub"]),
            dict(letter="C", axes=["axC"], marker="# C heatmap", figsize=(9.2, 4.3),
                 data=lambda ns: pd.DataFrame(ns["M"], index=ns["list"](ns["ORGAN"]),
                                              columns=ns["SHOW"]).rename_axis("organ").reset_index()
                 .melt(id_vars="organ", var_name="protein", value_name="rho")
                 .merge(pd.DataFrame(ns["Q"], index=ns["list"](ns["ORGAN"]), columns=ns["SHOW"])
                        .rename_axis("organ").reset_index()
                        .melt(id_vars="organ", var_name="protein", value_name="q"),
                        on=["organ", "protein"])),
        ],
    },
    "fig6": {
        "script": "fig6/code/fig6_make.py",
        "split": "plt.rcParams.update",
        "pre": "",
        "panels": [
            dict(letter="A", axes=["axA"], marker="# ---- A: MR forest ----", figsize=(4.8, 3.3),
                 data=lambda ns: ns["mr"]),
            dict(letter="B", axes=["axB"], marker="# ---- B: coloc PP ----", figsize=(4.8, 3.5),
                 data=lambda ns: ns["co"]),
            dict(letter="C", axes=["axC"], marker="# ---- C: druggability ----", figsize=(5.2, 2.9),
                 data=lambda ns: ns["dr"]),
            dict(letter="D", axes=["axD"], marker="# ---- D: cell-type expression ----",
                 figsize=(5.6, 3.5), data=lambda ns: ns["sc"]),
        ],
    },
    "figS1": {
        "script": "figS1/code/figS1_make.py",
        "split": "# ------------------------------------------------------------------ figure",
        "pre": "",
        "panels": [
            dict(letter="A", axes=["axA"], marker="# A detection-rate distribution",
                 figsize=(3.6, 3.1),
                 data=lambda ns: pd.DataFrame(dict(protein=ns["ff_all"].index,
                                                   floor_fraction=ns["ff_all"].to_numpy(),
                                                   retained=ns["ff_all"].to_numpy() <= 0.20))),
            dict(letter="B", axes=["axB"], marker="# B prognostic strength vs detection",
                 figsize=(3.7, 3.3),
                 data=lambda ns: ns["cox_all"][["protein", "ff", "z", "p", "retained"]]
                 .assign(is_primary=lambda d: d["protein"].isin(ns["prim_sig"]))),
            dict(letter="C", axes=["axC"], marker="# C threshold sweep", figsize=(3.8, 3.1),
                 data=lambda ns: ns["sweep"]),
            dict(letter="D", axes=["axD"], marker="# D model variants", figsize=(5.4, 3.3),
                 data=lambda ns: ns["var"]),
            dict(letter="E", axes=["axE"], marker="# E bootstrap stability", figsize=(4.4, 3.3),
                 data=lambda ns: ns["bootf"]),
            dict(letter="F", axes=["axF"], marker="# F full re-derivation in the reversed split",
                 figsize=(5.0, 3.7),
                 data=lambda ns: pd.read_csv(ns["OUT"] / "matrices" /
                                             "reversed_split_replication.csv")),
        ],
    },
    "figS2": {
        "script": "figS2/code/figS2_make.py",
        "split": "# ---------------- figure ----------------",
        "pre": "",
        "panels": [
            dict(letter="A", axes=["axA"], marker="# A  selection funnel",
                 figsize=(4.2, 3.2),
                 data=lambda ns: ns["funnel"]),
            dict(letter="B", axes=["axB"], marker="# B  panel score within MELD tertiles",
                 figsize=(5.4, 3.2),
                 data=lambda ns: ns["tert"].assign(p_interaction=ns["p_int"])),
            dict(letter="C", axes=["axC"], marker="# C  AUC comparison with CRP",
                 figsize=(4.6, 3.4),
                 data=lambda ns: ns["crp"]),
            dict(letter="D", axes=["axD"], marker="# D  win ratio", figsize=(4.8, 3.4),
                 data=lambda ns: ns["wr"]),
            dict(letter="E", axes=["axE"], marker="# E  decision-curve analysis",
                 figsize=(4.6, 3.3),
                 data=lambda ns: ns["dca"]),
            dict(letter="F", axes=["axF"],
                 marker="# F  net-benefit difference (panel minus MELD)",
                 figsize=(5.8, 2.8),
                 data=lambda ns: ns["dca"]
                 .pivot_table(index="threshold", columns="strategy", values="net_benefit")
                 .assign(delta=lambda d: d["MELD + 5-protein"] - d["MELD"])[["delta"]]
                 .rename_axis("threshold").reset_index()),
        ],
    },
}


# --------------------------------------------------------------------------- #
def split_source(text, split_marker):
    idx = text.index(split_marker)
    return text[:idx], text[idx:]


def extract_block(fig_part, marker, next_marker):
    start = fig_part.index(marker)
    end = fig_part.index(next_marker) if next_marker else len(fig_part)
    code = fig_part[start:end]
    supt = code.find("fig.suptitle(")
    if supt != -1:
        code = code[:supt]               # drop composite suptitle / savefig tail
    save = code.find("fig.savefig(")
    if save != -1:
        code = code[:save]
    return code


def strip_layout(code):
    """Drop axes/gridspec creation lines and composite-panel letter tags:
    the standalone canvas already has its axes, and the user re-labels
    panels during assembly, so A/B/C... letters must not appear."""
    import re
    keep = []
    for line in code.splitlines():
        if "add_subplot(" in line or "add_gridspec(" in line:
            continue
        m = re.search(r"\.text\((.*)\)", line)
        if m and re.search(r"[\"']([A-E])[\"']", m.group(1)) \
                and len(re.search(r"[\"']([A-E])[\"']", m.group(1)).group(0)) == 3:
            continue
        keep.append(line)
    return "\n".join(keep)


def build_axes(fig, names, sharey_pair=None):
    """Create the axes a panel block expects and return them by name."""
    out = {}
    if len(names) == 1:
        out[names[0]] = fig.add_subplot(111)
    else:
        for i, n in enumerate(names):
            kw = {}
            if sharey_pair and n == sharey_pair[1]:
                kw["sharey"] = out[sharey_pair[0]]
            out[n] = fig.add_subplot(1, len(names), i + 1, **kw)
    return out


def render_panel(spec, data_ns, panel, outdir, prefix, figdir):
    letter = panel["letter"]
    ns = dict(data_ns)
    if spec["pre"]:
        exec(compile(spec["pre"], "<pre>", "exec"), ns)

    fig = plt.figure(figsize=panel["figsize"])
    plt.figure(fig.number)          # make it current (plt.colorbar relies on gcf)
    axmap = build_axes(fig, panel["axes"], panel.get("sharey_pair"))
    ns.update(axmap)
    ns["fig"] = fig
    if panel.get("axis_off"):
        for a in axmap.values():
            a.axis("off")
    if panel.get("pre_axes"):
        exec(compile(panel["pre_axes"], "<pre_axes>", "exec"), ns)

    exec(compile(panel["code"], "<panel%s>" % letter, "exec"), ns)

    outs = []
    for ext in ("png", "tif", "pdf", "eps"):
        p = outdir / f"{prefix}_panel{letter}.{ext}"
        if ext in ("png", "tif"):
            fig.savefig(p, dpi=DPI, bbox_inches="tight")
        else:
            fig.savefig(p, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)

    df = None
    try:
        df = panel["data"](data_ns)
    except Exception as e:                                   # pragma: no cover
        print("   ! data export failed for %s: %r" % (letter, e))

    if isinstance(df, pd.DataFrame) and not df.empty:
        (outdir / "data").mkdir(exist_ok=True)
        df.to_csv(outdir / "data" / f"{prefix}_panel{letter}.csv",
                  index=False, encoding="utf-8-sig", na_rep="NA")
    return df, outs[2], outs[0]


def run(figkey):
    spec = SPEC[figkey]
    script = HERE / spec["script"]
    text = script.read_text(encoding="utf-8")
    data_part, fig_part = split_source(text, spec["split"])

    print("== %s: executing data section (%d chars)" % (figkey, len(data_part)))
    data_ns = {"__name__": "__panel_data__", "__file__": str(script)}
    exec(compile(data_part, str(script), "exec"), data_ns)
    if "list" not in data_ns:
        data_ns["list"] = list

    figdir = HERE / figkey
    outdir = figdir / "panels"
    outdir.mkdir(parents=True, exist_ok=True)

    markers = [p["marker"] for p in spec["panels"]]
    sheets = {}
    for i, panel in enumerate(spec["panels"]):
        nxt = markers[i + 1] if i + 1 < len(markers) else None
        panel = dict(panel)
        panel["code"] = strip_layout(extract_block(fig_part, panel["marker"], nxt))
        df, pdf, png = render_panel(spec, data_ns, panel, outdir, figkey, figdir)
        if isinstance(df, pd.DataFrame) and not df.empty:
            sheets[panel["letter"]] = df
        print("  PANEL %s -> %s" % (panel["letter"], png.name))

    if sheets:
        xlsx = outdir / f"{figkey}_panels_data.xlsx"
        with pd.ExcelWriter(xlsx) as xw:
            for k, v in sheets.items():
                v.to_excel(xw, sheet_name="Panel_" + k, index=False)
        print("  data workbook ->", xlsx.name)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SPEC)
    for k in keys:
        try:
            run(k)
        except Exception:
            traceback.print_exc()
            print("!! failed:", k)
