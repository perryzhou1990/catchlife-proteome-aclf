# -*- coding: utf-8 -*-
"""Fig S2 — second robustness programme (added after the metabolomics review).

A  Selection funnel (protocol v1.2): 883 assayed -> 128 full-cohort prognostic
   -> 91 derivation pool -> 6 stability-selected -> 3 | 5 lockings -> 5 locked (union)
B  Locked panel score within MELD tertiles (HR per SD, centre-stratified) + interaction test
C  Does an acute-phase protein replace the panel? MELD +/- CRP +/- 5-protein, AUC at 90 d
D  Win ratio for the 90-day composite endpoint (MELD-decile stratified, Finkelstein-Schoenfeld)
E  Decision-curve analysis in the external validation set

Numbers are read from analysis/output/*.csv so that the figure and the text
cannot drift apart.  Every panel is re-drawn standalone by make_panels.py.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[3]
OUT = BASE / "figures" / "figS2"
for sub in ("raw_data", "matrices", "code", "output"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

SRC = BASE / "analysis" / "output"
funnel = pd.read_csv(SRC / "funnel.csv")
tert = pd.read_csv(SRC / "meld_tertile.csv")
crp = pd.read_csv(SRC / "crp_model.csv")
wr = pd.read_csv(SRC / "win_ratio.csv")
dca = pd.read_csv(SRC / "dca.csv")
S = json.loads((SRC / "robustness2_summary.json").read_text(encoding="utf-8"))
p_int = S["meld_tertile_interaction_p"]

for d, name in ((funnel, "funnel"), (tert, "meld_tertile"), (crp, "crp_model"),
                (wr, "win_ratio"), (dca, "dca")):
    d.to_csv(OUT / "matrices" / f"{name}.csv", index=False, encoding="utf-8-sig")

BLUE, ORANGE, GREY, RED = "#2E5E8C", "#C4622D", "#6E6E6E", "#A03A3A"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---------------- figure ----------------
fig = plt.figure(figsize=(11.2, 8.0))
gs = fig.add_gridspec(3, 6, hspace=0.55, wspace=1.15,
                      left=0.075, right=0.985, top=0.905, bottom=0.075)
axA = fig.add_subplot(gs[0, :2])
axB = fig.add_subplot(gs[0, 2:])
axC = fig.add_subplot(gs[1, :2])
axD = fig.add_subplot(gs[1, 2:4])
axE = fig.add_subplot(gs[1, 4:])
axF = fig.add_subplot(gs[2, :])   # reserved: net-benefit difference

# =========================================================================== #
# A  selection funnel
# =========================================================================== #
lab = ["Assayed", "Prognostic\n(full cohort)", "Candidate pool\n(derivation set)",
       "Stability-\nselected", "Primary lock\n(883)", "Sensitivity lock\n(665)", "Locked panel\n(union)"]
n = funnel["n"].tolist()
cols = ["#DCE7F1", "#C3D6E8", "#A5C1DC", "#7FA3C4", "#6D93B8", BLUE, ORANGE]
ypos = np.arange(len(n))[::-1]
axA.barh(ypos, n, height=0.68, color=cols, edgecolor="white", lw=0.8)
for y, v in zip(ypos, n):
    axA.text(v + 18, y, f"{v:,}", va="center", ha="left", fontsize=8, color="#222222")
for y, s in zip(ypos, funnel["detail"]):
    pass  # details live in funnel.csv, keep the panel clean
axA.set_yticks(ypos)
axA.set_yticklabels(lab)
axA.set_xlim(0, 1060)
axA.set_xlabel("Number of proteins")
axA.set_title("Selection funnel", loc="left", fontweight="bold")
for s in ("top", "right"):
    axA.spines[s].set_visible(False)

# =========================================================================== #
# B  panel score within MELD tertiles
# =========================================================================== #
ypos = np.arange(len(tert))[::-1]
axB.errorbar(tert["HR"], ypos,
             xerr=[tert["HR"] - tert["CI_low"], tert["CI_high"] - tert["HR"]],
             fmt="o", ms=5.5, color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=2.5)
axB.axvline(1, color=GREY, ls=":", lw=1)
axB.set_yticks(ypos)
axB.set_yticklabels([f"{r.tertile}  (MELD {r.meld_low:.0f}-{r.meld_high:.0f})\n"
                     f"{r.events}/{r.n} events" for r in tert.itertuples()], fontsize=7)
axB.set_xscale("log")
axB.set_xlabel("HR per SD of locked panel score")
axB.set_title("Panel score within MELD tertiles", loc="left", fontweight="bold")
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.text(0.99, 0.06, ("P interaction < 0.001" if p_int < 0.001 else f"P interaction = {p_int:.3f}"), transform=axB.transAxes,
         ha="right", va="bottom", fontsize=7.5, color=GREY)

# =========================================================================== #
# C  AUC comparison with CRP
# =========================================================================== #
order = ["MELD", "MELD + CRP", "MELD + 5-protein", "MELD + 5-protein + CRP"]
cc = crp.set_index("model").loc[order].reset_index()
x = np.arange(len(cc))
colors = [GREY, "#B9B9B9", BLUE, ORANGE]
axC.bar(x, cc["AUC90"], color=colors, width=0.62, edgecolor="white")
for xi, r in zip(x, cc.itertuples()):
    axC.text(xi, r.AUC90 + 0.008, f"{r.AUC90:.3f}", ha="center", va="bottom", fontsize=7.5)
axC.set_xticks(x)
axC.set_xticklabels(["MELD", "MELD\n+ CRP", "MELD\n+ 5-protein", "MELD\n+ 5-protein\n+ CRP"], fontsize=7)
axC.set_ylim(0.60, 1.05)
axC.set_ylabel("AUC at 90 days (validation)")
axC.set_title("Is CRP a substitute?", loc="left", fontweight="bold")
for s in ("top", "right"):
    axC.spines[s].set_visible(False)
d = S["dAUC_crp"]
axC.text(0.5, 0.985, f"$\\Delta$AUC (5P+CRP vs 5P) = {d['est']:+.3f}  "
                     f"(95%CI {d['CI'][0]:+.3f} to {d['CI'][1]:+.3f})",
         transform=axC.transAxes, ha="center", va="top", fontsize=6.6, color="#333333")

# =========================================================================== #
# D  win ratio
# =========================================================================== #
wd = wr.copy()
ypos = np.arange(len(wd))[::-1]
for i, r in zip(ypos, wd.itertuples()):
    if np.isfinite(r.CI_low) and np.isfinite(r.CI_high):
        axD.plot([r.CI_low, r.CI_high], [i, i], color=BLUE, lw=1.2, solid_capstyle="butt")
ok = np.isfinite(wd["win_ratio"].to_numpy())
axD.plot(wd["win_ratio"][ok], np.asarray(ypos)[ok], "o", ms=5.5, color=BLUE)
axD.axvline(1, color=GREY, ls=":", lw=1)
axD.set_yticks(ypos)
axD.set_yticklabels(["All\n(MELD decile)", "MELD T1\n(quartile)", "MELD T2\n(quartile)",
                     "MELD T3\n(quartile)"], fontsize=7)
axD.set_xlabel("Win ratio, score ≤ median vs > median")
axD.set_title("Win ratio, 90-day composite", loc="left", fontweight="bold")
axD.set_xscale("log")
axD.set_xlim(0.8, 60)
for s in ("top", "right"):
    axD.spines[s].set_visible(False)
# T1: no low-score patient had an event -> not estimable
axD.text(1.9, np.asarray(ypos)[1], "∞ (0 losses)", va="center", ha="left",
         fontsize=7, color=GREY)
axD.text(0.02, 0.06, f"overall WR {S['win_ratio']['WR']:.2f} "
                     f"({S['win_ratio']['CI'][0]:.2f}-{S['win_ratio']['CI'][1]:.2f})",
         transform=axD.transAxes, ha="left", va="bottom", fontsize=7, color="#333333")

# =========================================================================== #
# E  decision-curve analysis
# =========================================================================== #
styles = {"MELD": dict(color=GREY, ls="--", lw=1.2),
          "MELD + 5-protein": dict(color=BLUE, ls="-", lw=1.6),
          "treat all": dict(color="#B0B0B0", ls=":", lw=1.0)}
for name, st in styles.items():
    dd = dca[dca.strategy == name]
    axE.plot(dd["threshold"], dd["net_benefit"], label=name.replace(" + ", "+"), **st)
axE.plot([0, 0.6], [0, 0], color="#B0B0B0", ls=":", lw=1.0)
axE.set_xlim(0.05, 0.60)
axE.set_ylim(-0.05, 0.30)
axE.set_xlabel("Threshold probability")
axE.set_ylabel("Net benefit")
axE.set_title("Decision curves (validation)", loc="left", fontweight="bold")
axE.legend(loc="upper right", fontsize=6.8, handlelength=1.6)
for s in ("top", "right"):
    axE.spines[s].set_visible(False)

# =========================================================================== #
# F  net-benefit difference (panel minus MELD)
# =========================================================================== #
piv = dca.pivot_table(index="threshold", columns="strategy", values="net_benefit")
diff = (piv["MELD + 5-protein"] - piv["MELD"])
axF.axhline(0, color=GREY, ls=":", lw=1)
axF.fill_between(diff.index, 0, diff.values, where=diff.values > 0, color=BLUE, alpha=0.25, lw=0)
axF.plot(diff.index, diff.values, color=BLUE, lw=1.4)
axF.set_xlim(0.05, 0.60)
axF.set_xlabel("Threshold probability")
axF.set_ylabel("$\\Delta$ net benefit\n(panel − MELD)")
axF.set_title("Net benefit gained by the panel", loc="left", fontweight="bold")
for s in ("top", "right"):
    axF.spines[s].set_visible(False)

fig.suptitle("Figure S2. Selection funnel and second robustness programme "
             "(MELD stratification, CRP comparator, win ratio, decision curves)",
             fontsize=10, fontweight="bold", x=0.075, ha="left", y=0.975)

# ---- letters (composite only; standalone panels are exported unlabelled) ----
for ax, letter in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"), (axE, "E"), (axF, "F")):
    ax.text(-0.22 if ax in (axA, axC) else -0.16, 1.06, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")

fig.savefig(OUT / "output" / "figS2.pdf", bbox_inches="tight")
fig.savefig(OUT / "output" / "figS2.png", dpi=600, bbox_inches="tight")
print("saved", OUT / "output" / "figS2.png")
