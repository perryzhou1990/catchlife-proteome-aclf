# -*- coding: utf-8 -*-
"""Fig 6 — Causal direction, druggability and cellular origin of the 5-protein panel.
A  cis-pQTL Mendelian randomisation (cirrhosis, GCST90319878): OR per SD, IVW
B  Colocalisation posterior probabilities (PP.H4)
C  Druggability annotation (ChEMBL + Open Targets + UniProt + ClinicalTrials.gov)
D  Cell-type expression in human liver atlas (CELLxGENE, 73,295 cells)

Rev 2026-11-06: MR weighted-median SE replaced by parametric bootstrap and coloc
aggregation fixed (log-sum-exp); panel A annotates the 80%-power MDE, panel B is
retitled to the actual finding (no shared variant supported, PP.H4 <= 0.05).
Rev 2026-11-07: panel C rebuilt from analysis/09_druggability_assessment.py
(druggability_v2.csv): localisation, modality feasibility, highest clinical
stage, direction implied by the Cox model and registered trials in liver
disease. GH is now shown; it had been silently dropped from the figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[3]
FIG6 = BASE / "figures" / "fig6"
MAT, OUT = FIG6 / "matrices", FIG6 / "output"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["TNFRSF1A", "REN", "TNFRSF21", "SCARF2", "ANGPT2", "CCL14", "CRP"]
ORDER_C = ["TNFRSF1A", "TNFRSF21", "REN", "SCARF2", "GH1", "ANGPT2", "CCL14", "CRP"]
ALIAS = {"TNFRSF1A": "TNF-RI", "REN": "Renin", "TNFRSF21": "DR6", "SCARF2": "SREC-II",
         "GH1": "GH", "ANGPT2": "ANG-2", "CCL14": "HCC-1", "CRP": "CRP"}
CORE = {"TNFRSF1A", "REN", "TNFRSF21", "SCARF2", "GH1"}

mr = pd.read_csv(MAT / "mr_results.csv")
co = pd.read_csv(MAT / "coloc_results.csv")
dr = pd.read_csv(MAT / "druggability_v2.csv")
sc = pd.read_csv(MAT / "scrna_expression.csv")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig = plt.figure(figsize=(10.4, 8.2))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34,
                      width_ratios=[1.0, 1.15], height_ratios=[1.0, 1.0])
axA, axB, axC, axD = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                      fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))

# ---- A: MR forest ----
axA.text(-0.34, 1.06, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")
sub = mr.set_index("gene").reindex([g for g in ORDER if g in set(mr.gene)]).dropna(subset=["b_ivw"])
y = np.arange(len(sub))[::-1]
for k, (g, r) in enumerate(sub.iterrows()):
    col = "#C0504D" if g in CORE else "#8C8C8C"
    axA.plot([r["or_lo"], r["or_hi"]], [y[k], y[k]], color=col, lw=1.3)
    axA.plot(r["or_ivw"], y[k], "o", color=col, ms=5)
    axA.text(0.99, y[k], f'{int(r["n_snp"])} SNP' + ("s" if r["n_snp"] > 1 else ""),
             transform=axA.get_yaxis_transform(), fontsize=5.6, va="center", ha="right",
             color="#444444")
axA.axvline(1, color="#888888", ls="--", lw=0.8)
axA.set_yticks(y, [ALIAS[g] for g in sub.index], fontsize=7)
axA.set_xscale("log"); axA.set_xticks([0.5, 0.7, 1, 1.5, 2])
axA.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
axA.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axA.set_xlabel("OR for cirrhosis per SD higher protein (cis-pQTL, IVW)", fontsize=6.8)
axA.text(0.0, 1.075, "all 95% CIs include 1;  80%-power MDE 1.08 (HCC-1) to 1.78 (TNF-RI) per SD",
         transform=axA.transAxes, fontsize=5.0, color="#666666", ha="left")
axA.text(0.0, 1.025, "filled: core panel;  grey: exploratory comparators",
         transform=axA.transAxes, fontsize=5.0, color="#666666", ha="left")
axA.set_title("Causal effect on cirrhosis susceptibility (cis-MR)", fontsize=8.5, loc="left", pad=22)
axA.tick_params(labelsize=6)
axA.text(0.99, 0.03, " ", transform=axA.transAxes, fontsize=1)

# ---- B: coloc PP ----
axB.text(-0.14, 1.06, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
cs = co.set_index("gene").reindex([g for g in ORDER if g in set(co.gene)])
xpos = np.arange(len(cs))
ppt = cs[["PPH0", "PPH1", "PPH2", "PPH3", "PPH4"]].fillna(0).to_numpy()
cols = ["#E8E8E8", "#C9D6E8", "#9FB6D4", "#E8C98C", "#C0504D"]
bottom = np.zeros(len(cs))
for j, c in enumerate(cols):
    axB.bar(xpos, ppt[:, j], bottom=bottom, color=c, width=0.66,
            label=["H0", "H1", "H2", "H3", "H4"][j])
    bottom += ppt[:, j]
for i, g in enumerate(cs.index):
    v = cs.loc[g, "PPH4"]
    axB.text(i, 1.03, f"{v:.2f}", ha="center", fontsize=6.2,
             color="#7A2E24" if v > 0.5 else "#555555")
axB.set_xticks(xpos, [ALIAS[g] for g in cs.index], fontsize=6.6, rotation=25, ha="right")
axB.set_ylim(0, 1.12); axB.set_ylabel("posterior probability", fontsize=7)
axB.legend(fontsize=5.6, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
axB.set_title("Colocalisation: mass on protein-only signal, PP.H4 \u2264 0.05", fontsize=8.5, loc="left")
axB.tick_params(labelsize=6)

# ---- C: druggability ----
axC.text(-0.34, 1.06, "C", transform=axC.transAxes, fontsize=12, fontweight="bold")
axC.axis("off")
dd = dr.set_index("gene").reindex(ORDER_C)
STAGE = {0: "-", 1: "ph 1", 2: "ph 2", 3: "ph 3", 4: "approved"}
cells, cellcols = [], []
for g, r in dd.iterrows():
    loc = "membrane" if "membrane" in str(r.get("uniprot_loc", "")).lower() else \
          ("secreted" if "secreted" in str(r.get("uniprot_loc", "")).lower() else "-")
    sm = r.get("ot_sm_druggable_family") is True or r.get("ot_sm_pocket") is True or \
         (not pd.isna(r.get("n_mechanisms")) and r.get("n_mechanisms", 0) > 0
          and r.get("target_class", "") and "protease" in str(r.get("target_class", "")).lower())
    ab = r.get("ot_ab_extracellular") is True
    if g == "GH1":
        mod = "replacement\nbiologic"
    elif loc == "membrane" and ab:
        mod = "antibody"
    elif loc == "membrane" and sm:
        mod = "small molecule"
    elif loc == "secreted" and sm:
        mod = "small molecule"
    elif ab:
        mod = "antibody"
    else:
        mod = "no tool\nreported"
    ph = max(0.0 if pd.isna(r.get("max_phase")) else r.get("max_phase"),
             0.0 if pd.isna(r.get("self_drug_phase")) else r.get("self_drug_phase"))
    stage = STAGE.get(int(ph), "-")
    if stage.startswith("approved") and g == "GH1":
        stage = "approved\n(replacement)"
    direction = "inhibit" if r.get("implied_direction") == "inhibit" else "augment"
    n_tr = r.get("n_trials_liver")
    n_tr = "-" if pd.isna(n_tr) else f"{int(n_tr)}"
    cells.append([ALIAS[g] + (" *" if r.get("role") == "panel" else ""), loc, mod, stage, direction, n_tr])
    rowcol = []
    for j in range(6):
        rowcol.append("#FFFFFF" if r.get("role") == "panel" else "#F4F4F4")
    cellcols.append(rowcol)

tb = axC.table(cellText=cells, colLabels=["Protein", "Localisation", "Modality",
                                          "Highest stage", "Direction\n(data)", "Liver\ntrials"],
               loc="center", cellLoc="center",
               colWidths=[0.15, 0.16, 0.21, 0.20, 0.16, 0.12], bbox=[0, 0.18, 1, 0.80])
tb.auto_set_font_size(False); tb.set_fontsize(5.6)
for i in range(len(cells)):
    for j in range(6):
        tb[i + 1, j].set_facecolor(cellcols[i][j])
for j in range(6):
    tb[0, j].set_facecolor("#2E5E8C"); tb[0, j].set_text_props(color="white", fontweight="bold")
# highlight: available agents vs none, and zero vs non-zero liver programmes
for i, g in enumerate(dd.index):
    r = dd.loc[g]
    if str(cells[i][3]).startswith("-"):
        tb[i + 1, 3].set_facecolor("#F2F2F2"); tb[i + 1, 3].set_text_props(color="#999999")
    else:
        tb[i + 1, 3].set_facecolor("#DCE6F1")
    if cells[i][5] == "0":
        tb[i + 1, 5].set_facecolor("#F2F2F2"); tb[i + 1, 5].set_text_props(color="#999999")
    elif cells[i][5] != "-":
        tb[i + 1, 5].set_facecolor("#C6D9EC")
axC.text(0.0, 0.10, "* panel protein;  unmarked rows are exploratory comparators.  "
                    "Highest stage = furthest clinical stage of any agent annotated to the\n"
                    "target (ChEMBL / Open Targets);  liver trials = registered studies of those agents in "
                    "cirrhosis, liver failure or\nhepatic insufficiency (ClinicalTrials.gov, 2026-11-07)",
         transform=axC.transAxes, fontsize=4.8, color="#666666", va="top")
axC.set_title("Druggability: tractable targets, but no liver programme for any of the five",
              fontsize=8.5, loc="left")

# ---- D: cell-type expression ----
axD.text(-0.30, 1.06, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")
piv = sc.pivot(index="gene", columns="cell_type", values="pct_expr").reindex(ORDER)
piv = piv[[c for c in piv.columns if piv[c].max() > 0]]
# order cell types by max expression of the panel
piv = piv[piv.max(axis=0).sort_values(ascending=False).index]
im = axD.imshow(piv.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0, vmax=np.nanmax(piv.to_numpy()) * 0.9)
SHORT = {"erythroid lineage cell": "erythroid", "midzonal region hepatocyte": "midzonal hep.",
         "periportal region hepatocyte": "periportal hep.", "centrilobular region hepatocyte": "centrilob. hep.",
         "hepatic stellate cell": "HSC", "cholangiocyte": "cholangio.", "natural killer cell": "NK cell",
         "M1 macrophage": "M1 macro.", "Kupffer cell": "Kupffer",
         "endothelial cell of pericentral hepatic sinusoid": "LSEC", "B cell": "B cell",
         "blood vessel endothelial cell": "vasc. EC"}
axD.set_xticks(range(piv.shape[1]), [SHORT.get(c, c) for c in piv.columns],
               fontsize=5.8, rotation=40, ha="right")
axD.set_yticks(range(len(piv)), [ALIAS[g] for g in piv.index], fontsize=6.6)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.iloc[i, j]
        if np.isfinite(v) and v >= 5:
            axD.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.2,
                     color="white" if v > np.nanmax(piv.to_numpy()) * 0.55 else "#333333")
cb = fig.colorbar(im, ax=axD, fraction=0.022, pad=0.01)
cb.set_label("% cells expressing", fontsize=6.2); cb.ax.tick_params(labelsize=5.5)
axD.set_title("Cellular origin (human liver atlas, 73,295 cells)", fontsize=8.5, loc="left")
axD.tick_params(length=0)
axD.text(0, -0.42, "GH1: not detected in any of the 12 hepatic cell types (0% of cells), consistent with an "
                   "extrahepatic, pituitary source", transform=axD.transAxes, fontsize=5.2, color="#666666")

fig.suptitle("Figure 6. No detectable causal effect on cirrhosis liability, tractable targets without a liver programme, "
             "and an endothelial/cholangiocyte source",
             fontsize=9.6, fontweight="bold", y=0.985)
fig.savefig(OUT / "fig6.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig6.png", dpi=300, bbox_inches="tight")
print("DONE")
