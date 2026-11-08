# -*- coding: utf-8 -*-
"""Fig 1 — Study design, cohort flow and baseline characteristics (CATCH-LIFE, n=1390)
Outputs: figures/fig1/output/fig1.pdf + fig1.png (>=300 dpi), matrices/table1_baseline.csv
All figure text in English (GR13).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).resolve().parents[3]          # .../选题A_蛋白组炎症内皮轴预后
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
OUT = BASE / "figures" / "fig1"
for sub in ["raw_data", "matrices", "code", "output"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# ---------------- load ----------------
df = pd.read_excel(CLIN, sheet_name="队列数据1390人")
df.columns = [str(c).strip() for c in df.columns]
# age lives in the companion scoring workbook (same IDs)
_age = pd.read_excel(r"D://科研//博士课题//蛋白质组多中心数据-to 周沛//蛋白质组数据及临床信息//catch-life队列蛋白质组患者1390人-年龄-疾病评分.xlsx")
df = df.merge(_age[["ID", "AGE"]], on="ID", how="left")
df["center"] = df["ID"].astype(str).str.split("-").str[0]

col_aclf = "COSSH_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"
aclf_map = {0: "No ACLF", 1: "ACLF on admission", 2: "Developed ACLF"}
df["aclf_grp"] = df[col_aclf].map(aclf_map)

# events (0 alive, 1 death, 2 transplant, 4 lost)
def events(col):
    v = df[col]
    return dict(alive=int((v == 0).sum()), death=int((v == 1).sum()),
                tx=int((v == 2).sum()), lost=int((v == 4).sum()),
                composite=int(v.isin([1, 2]).sum()))
ev = {lab: events(c) for lab, c in
      [("28-day", "O28"), ("90-day", "O90_2020.10更新后"),
       ("6-month", "O180_twt根据更新的365更正"), ("1-year", "O365_更新")]}

# ---------------- Table 1 ----------------
def fmt_med(s, nd=1):
    s = s.dropna()
    return f"{s.median():.{nd}f} ({s.quantile(.25):.{nd}f}\u2013{s.quantile(.75):.{nd}f})" if len(s) else "NA"


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def p_cont(vals):
    from scipy.stats import kruskal
    return _p(kruskal(*vals).pvalue)


def p_cat(cols_by_g, level):
    from scipy.stats import chi2_contingency
    tab = [[int((c == level).sum()), int((c != level).sum())] for c in cols_by_g]
    return _p(chi2_contingency(tab).pvalue)

def fmt_cat(s, level):
    n = (s == level).sum()
    return f"{n} ({100 * n / s.notna().sum():.1f})"

rows = []
groups = ["No ACLF", "ACLF on admission", "Developed ACLF"]
for name, col, kind, arg in (
        [("n", None, "n", None)] +
        [("Age, years", "AGE", "cont", 1)] +
        [("Bilirubin, μmol/L", "TB(umol/L)", "cont", 1), ("INR", "INR", "cont", 2),
         ("Creatinine, mg/dL", "CR(mg/dL)", "cont", 2), ("Sodium, mmol/L", "Na", "cont", 1),
         ("Albumin, g/L", "ALB", "cont", 1), ("ALT, U/L", "ALT", "cont", 0),
         ("Platelets, ×10⁹/L", "PLT", "cont", 0), ("WBC, ×10⁹/L", "WBC", "cont", 2),
         ("CRP, clinical chemistry, mg/L", "CRP", "cont", 1), ("MAP, mmHg", "MAP", "cont", 0)] +
        [("Male sex", "Gender", "cat", 1),
         ("Cirrhosis", "cirrhosis", "cat", 1),
         ("HBV aetiology", "HBV", "cat", 1),
         ("Infection at admission", "AD_infection(0.no1.yes)", "cat", 1)]):
    r = {"Variable": name}
    subs = [df[df["aclf_grp"] == g] for g in groups]
    for g, sub in zip(groups, subs):
        if kind == "n":
            r[g] = str(len(sub))
        elif kind == "cont":
            r[g] = fmt_med(pd.to_numeric(sub[col], errors="coerce"), arg)
        else:
            r[g] = fmt_cat(sub[col], arg)
    if kind == "cont":
        r["P"] = p_cont([pd.to_numeric(s[col], errors="coerce").dropna() for s in subs])
    elif kind == "cat":
        r["P"] = p_cat([s[col] for s in subs], arg)
    else:
        r["P"] = ""
    rows.append(r)
t1 = pd.DataFrame(rows)
t1.to_csv(OUT / "matrices" / "table1_baseline.csv", index=False, encoding="utf-8-sig")

# center counts
cc = df["center"].value_counts().sort_values(ascending=False)
# anonymised centre labels C1-C13 by sample size (consistent with Fig 5)
cc.index = [f"C{i+1}" for i in range(len(cc))]
cc.to_csv(OUT / "matrices" / "center_counts.csv", header=["n"])

BLUE, GREEN, PURPLE, AMBER = "#2E5E8C", "#2E7D4F", "#6A4C93", "#B07A2A"

LANES = [
    ("Aim 1  Landscape", GREEN, "#E8F4EC",
     ["883 proteins\nacross the AD\u2013ACLF gradient",
      "Linear models\nBH-FDR across 883 tests",
      "Co-expression modules\n(k-means, k=8)",
      "Pathway enrichment\n(Enrichr, GO/KEGG)"], "Fig 2"),
    ("Aim 2  Prognosis", BLUE, "#EAF2FA",
     ["Derivation-only screen\nsex + MELD adjusted, FDR<0.05\n(91 proteins)",
      "LASSO-Cox, 10-fold CV\n\u03bb by held-out C-index",
      "Stability selection\n300 bootstraps, \u226570%",
      "Locking rules \u2460\u2013\u2463\nthen internal\u2013external validation"], "Figs 3\u20135"),
    ("Aim 3  Mechanism", PURPLE, "#EFE9F6",
     ["cis-pQTL MR\n+ colocalisation",
      "Druggability\n(ChEMBL)",
      "Cellular source\n(liver scRNA-seq)",
      "Interpretation:\nseverity marker, not cause"], "Fig 6"),
]

BENCH = [
    ["Design element", "JHEP 2023 (metabolomics)", "JCTH 2026 (metabolomics)", "Present study (proteomics)"],
    ["Population / n", "1,024 (HBV subgroup)", "1,314 (all aetiologies)", "1,390 (all aetiologies)"],
    ["Platform", "Metabolon HD4", "Metabolon HD4", "883-plex antibody panel"],
    ["Features analysed", "730 metabolites", "metabolite library", "883 proteins, none excluded"],
    ["Derivation / validation", "1:1 random split", "7:3 stratified random split", "5 vs 8 centres (no overlap)"],
    ["Validation strength", "internal", "internal", "internal–external + leave-one-centre-out"],
    ["Centre / batch adjustment", "not applied", "not applied", "centre + sex residualisation"],
    ["Selection stability", "not reported", "not reported", "300 bootstrap resamples"],
    ["Composite endpoint", "\u2014", "win ratio", "Fine-Gray + win ratio + DCA"],
    ["Targeted confirmation", "4-metabolite LC-MS/MS kit", "not performed", "not performed (split-invariance analysis)"],
    ["Severity confounding", "comparator model", "comparator model", "MELD tertiles + CRP (null)"],
]
# ---------------- figure ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig = plt.figure(figsize=(7.4, 11.4))
gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 1.15, 0.95, 1.05], hspace=0.30,
                      left=0.03, right=0.97, top=0.965, bottom=0.02)



def box(ax, x, y, w, h, text, fc="#EAF2FA", ec="#2E5E8C", fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.35)

# --- Panel A: cohort, analysis set and centre-based split ---
axA = fig.add_subplot(gs[0]); axA.axis("off")
axA.text(-0.02, 1.05, "A", transform=axA.transAxes, fontsize=12, fontweight="bold", va="top")
axA.set_xlim(0, 1); axA.set_ylim(0, 1)
box(axA, 0.22, 0.80, 0.56, 0.17,
    "CATCH-LIFE prospective multicentre cohort (13 centres)\n"
    "1,390 patients with acutely decompensated chronic liver disease\n"
    "plasma proteomics: 883 proteins quantified at admission", fs=7.2)
box(axA, 0.18, 0.55, 0.64, 0.13,
    "Analysis set: all 883 proteins \u2014 no detection filter, no protein excluded\n"
    "(the 218 low-detection proteins are analysed and flagged, not discarded)",
    fc="#E8F4EC", ec=GREEN, fs=7.0)
axA.add_patch(FancyArrowPatch((0.5, 0.795), (0.5, 0.685), arrowstyle="-|>",
                              color=BLUE, lw=1.2, shrinkA=0, shrinkB=0))
box(axA, 0.02, 0.26, 0.46, 0.20,
    "Derivation set (5 centres)\nn = 995;  116 events\n"
    "screening \u2192 LASSO \u2192 stability \u2192 locking", fc="#EAF2FA", ec=BLUE, fs=6.8)
box(axA, 0.52, 0.26, 0.46, 0.20,
    "Validation set (8 centres)\nn = 395;  46 events\n"
    "locked model applied unchanged, nothing refitted", fc="#EAF2FA", ec=BLUE, fs=6.8)
n_no = int((df[col_aclf] == 0).sum()); n_on = int((df[col_aclf] == 1).sum())
n_dev = int((df[col_aclf] == 2).sum())
col_easl = "EASL_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"
n_easl = int((df[col_easl] == 1).sum())
box(axA, 0.02, 0.01, 0.30, 0.19,
    "Endpoints (transplant-free survival)\n"
    f"28 d {ev['28-day']['composite']}  |  90 d {ev['90-day']['composite']}  |  "
    f"6 mo {ev['6-month']['composite']}  |  1 y {ev['1-year']['composite']} events\n"
    f"90 d: {ev['90-day']['death']} deaths, {ev['90-day']['tx']} transplantations, 0 lost",
    fc="#FDF3E3", ec=AMBER, fs=6.3)
box(axA, 0.35, 0.01, 0.30, 0.19,
    "ACLF status (COSSH primary; EASL-CLIF sensitivity)\n"
    f"No ACLF {n_no}  |  on admission {n_on}  |  developed {n_dev}\n"
    f"EASL-CLIF ACLF: {n_easl}",
    fc="#FDF3E3", ec=AMBER, fs=6.1)
box(axA, 0.68, 0.01, 0.30, 0.19,
    "Key subgroups (pre-defined)\n"
    f"HBV {int((df.HBV == 1).sum())}  |  non-HBV {int((df.HBV == 0).sum())}\n"
    f"cirrhosis {int((df.cirrhosis == 1).sum())}  |  no cirrhosis {int((df.cirrhosis == 0).sum())}",
    fc="#FDF3E3", ec=AMBER, fs=6.1)
for x in (0.25, 0.75):
    axA.add_patch(FancyArrowPatch((0.5, 0.545), (x, 0.465), arrowstyle="-|>",
                                  color=BLUE, lw=1.0, shrinkA=0, shrinkB=0))

# --- Panel B: pre-defined analysis architecture ---
axB = fig.add_subplot(gs[1]); axB.axis("off")
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.text(-0.02, 1.04, "B", transform=axB.transAxes, fontsize=12, fontweight="bold", va="top")

lanes = LANES
y0, hgt, gap = 0.76, 0.22, 0.25
for r, (title, ec, fc, steps, figref) in enumerate(lanes):
    y = y0 - r * gap
    axB.text(0.0, y + hgt + 0.025, title, fontsize=7.4, fontweight="bold", color=ec,
             ha="left", va="bottom")
    axB.text(0.985, y + hgt + 0.025, figref, fontsize=6.6, color=ec, ha="right", va="bottom")
    n = len(steps); w = 0.225; gp = (1 - n * w) / (n - 1)
    for i, s in enumerate(steps):
        x = i * (w + gp)
        box(axB, x, y, w, hgt, s, fc=fc, ec=ec, fs=6.0)
        if i < n - 1:
            axB.add_patch(FancyArrowPatch((x + w + 0.003, y + hgt / 2),
                                          (x + w + gp - 0.003, y + hgt / 2),
                                          arrowstyle="-|>", color="#555555", lw=0.9))
axB.text(0.0, 0.155,
         "Pre-defined: analysis set (883, no filter) \u00b7 centre-based split \u00b7 screening rule \u00b7 \u03bb selection \u00b7 "
         "stability threshold \u00b7 locking rules \u2460\u2013\u2463 \u00b7 validation metrics",
         fontsize=5.9, color="#222222", ha="left", va="top")
axB.text(0.0, 0.105,
         "Exploratory and reported as such: 25 low-detection proteins with FDR<0.05 \u00b7 MR and single-cell annotations",
         fontsize=5.9, color="#555555", ha="left", va="top")
axB.text(0.0, 0.055,
         "Negative controls built into the design: CRP comparator (no gain) \u00b7 cis-pQTL MR (all null) \u00b7 "
         "death-only and reversed-split variants",
         fontsize=5.9, color="#555555", ha="left", va="top")

# --- Panel C: design benchmarking against the two metabolomics papers ---
axC = fig.add_subplot(gs[2]); axC.axis("off")
axC.text(-0.04, 1.06, "C", transform=axC.transAxes, fontsize=12, fontweight="bold", va="top")
axC.text(0.5, 1.06, "Design benchmarking against the two metabolomic studies of the same cohort",
         transform=axC.transAxes, ha="center", fontsize=8.6, fontweight="bold")
bench = BENCH
tab = axC.table(cellText=bench, loc="center", cellLoc="left",
                colWidths=[0.24, 0.245, 0.245, 0.27], bbox=[0, 0.0, 1, 0.97])
tab.auto_set_font_size(False); tab.set_fontsize(5.9)
for j in range(4):
    tab[0, j].set_facecolor("#2E5E8C"); tab[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(bench)):
    for j in range(4):
        tab[i, j].set_facecolor("#F5F8FB" if i % 2 else "white")
    tab[i, 3].set_text_props(fontweight="bold")
axC.text(0.0, -0.06, "Em dash, not reported. DCA, decision-curve analysis.",
         transform=axC.transAxes, fontsize=5.8, color="#555555")

# --- Panel D: Table 1 ---
axD = fig.add_subplot(gs[3]); axD.axis("off")
axD.text(-0.04, 1.05, "D", transform=axD.transAxes, fontsize=12, fontweight="bold", va="top")
axD.text(0.5, 1.05, "Table 1. Baseline characteristics by ACLF status (COSSH)",
         transform=axD.transAxes, ha="center", fontsize=9.2, fontweight="bold")
cells = [list(t1.columns)] + t1.values.tolist()
tab2 = axD.table(cellText=cells, loc="center", cellLoc="center",
                 colWidths=[0.34, 0.20, 0.20, 0.20, 0.06], bbox=[0, 0.02, 1, 0.95])
tab2.auto_set_font_size(False); tab2.set_fontsize(6.3); tab2.scale(1, 1.18)
for j in range(len(cells[0])):
    tab2[0, j].set_facecolor("#2E5E8C"); tab2[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(cells)):
    for j in range(len(cells[0])):
        tab2[i, j].set_facecolor("#F5F8FB" if i % 2 else "white")
axD.text(0, -0.04, "Continuous variables: median (IQR), Kruskal\u2013Wallis test; categorical: n (%), \u03c7\u00b2 test. "
         "CRP, C-reactive protein; MAP, mean arterial pressure.",
         transform=axD.transAxes, fontsize=6.0, color="#555555")

fig.suptitle("Figure 1. Study design and pre-defined analysis architecture, with design benchmarking",
             fontsize=10.2, fontweight="bold", y=0.995)
fig.savefig(OUT / "output" / "fig1.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "output" / "fig1.png", dpi=300, bbox_inches="tight")
print("Table 1:\n", t1.to_string(index=False))
print("centers:\n", cc.to_string())
print("events:", ev)
print("DONE")
