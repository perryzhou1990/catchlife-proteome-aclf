# -*- coding: utf-8 -*-
"""Supplementary Fig S1 — Sensitivity to the protein detection filter and to modelling choices.

Background: the primary pipeline drops 218/883 proteins detected in <80% of patients
(floor-value fraction ff > 0.20). All 14 pre-specified panel candidates survive
(ff <= 0.141), so the panel itself is filter-independent; the questions are
  (i)  was the data-driven prognostic list (89 Cox-FDR proteins, Fig 3) shaped by the filter?
  (ii) does the fitted model survive endpoint / split / covariate re-specification?

A  Distribution of floor-value fraction over all 883 proteins, 20% cut-off marked
B  Prognostic strength (|z| from MELD-adjusted Cox) vs detection fraction, all 883
C  Threshold sweep: proteins retained, and number FDR<0.05, across cut-offs 1%-60%
D  Model robustness: AUC90 / C-index under 6 analysis variants (validation set)
E  Bootstrap selection stability of the 14-candidate LASSO-Cox (200 resamples)
F  Full re-derivation in the reversed split (8 small centres, n=395, 46 events):
   per-protein HR in the reversed derivation vs the 5 large centres

NOTE: Cox fits for the 665 retained proteins are IDENTICAL to Fig 3 (fit is per-protein),
      so only the 218 excluded proteins are refitted here and merged with cox90_all.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[3]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\proteinarray_matrix_1390人蛋白质组结果（质控后883蛋白）.xlsx")
ANNOT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\宋晶分组分析结果-供参考\cirrhosis_vs_noncirrhosis.csv")
OUT = BASE / "figures" / "figS1"
for sub in ["raw_data", "matrices", "code", "output"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ data
clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
clin.columns = [str(c).strip() for c in clin.columns]
clin["center"] = clin["ID"].astype(str).str.split("-").str[0]
prot = pd.read_excel(PROT, sheet_name="1390已测蛋白质组")
prot.columns = [str(c).strip() for c in prot.columns]
prot = prot.set_index(prot.columns[0]).loc[clin["ID"]].astype(float)
ff_all = (prot == prot.min()).mean()          # floor-value fraction, all 883

t1y = pd.to_numeric(clin["OS_1y_更新"], errors="coerce")
s90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
time90 = np.minimum(t1y.fillna(90), 90).to_numpy()
event90 = s90.isin([1, 2]).astype(int).to_numpy()
death90 = (s90 == 1).astype(int).to_numpy()          # death only, transplant censored
alive90 = (s90 == 0).to_numpy()
sex = clin["Gender"].replace(2, np.nan); sex = sex.fillna(sex.mode()[0]).astype(float).to_numpy()
tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr = pd.to_numeric(clin["INR"], errors="coerce"); inr = inr.fillna(inr.median())
cr = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce"); cr = cr.fillna(cr.median())
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.clip(lower=1))
        + 9.57 * np.log(cr.clip(lower=1)) + 6.43).clip(6, 40).to_numpy()
center = clin["center"].to_numpy()
ev1y = pd.to_numeric(clin["O365_更新"], errors="coerce")

pool = ["TNF.RI", "Renin", "HCC.1", "SREC.II", "DR6", "ANG.2", "Axl", "CRP",
        "SR.AI", "CD200", "uPAR", "VEGF.R2", "Mesothelin", "TECK"]
pool = [p for p in pool if p in prot.columns]
annot = pd.read_csv(ANNOT)[["protein", "geneID"]].drop_duplicates("protein")
g2sym = lambda p: (annot.loc[annot["protein"] == p, "geneID"].iloc[0]
                   if (annot["protein"] == p).any() else p)

base = pd.DataFrame({"time": time90, "event": event90, "sex": sex, "meld": meld, "center": center})

# ------------------------------------------------------------------ 1. Cox for the 218 excluded
prim = pd.read_csv(BASE / "figures" / "fig3" / "matrices" / "cox90_all.csv")
CACHE = OUT / "matrices" / "cox90_all883.csv"
Zall = np.log2(prot.clip(lower=1))
Zall = (Zall - Zall.mean()) / Zall.std(ddof=1)
if CACHE.exists():                      # 218 Cox fits take ~3.5 min -> cache
    cox_all = pd.read_csv(CACHE)
    print("loaded cached Cox table:", cox_all.shape)
else:
    dropped = [p for p in ff_all.index if ff_all[p] > 0.20 and p not in set(prim["protein"])]
    print("excluded proteins to refit:", len(dropped))
    rows = []
    for p in dropped:
        d = base.copy(); d["protein"] = Zall[p].to_numpy()
        try:
            cph = CoxPHFitter(penalizer=0.0)
            cph.fit(d, "time", "event", strata=["center"])
            s = cph.summary.loc["protein"]
            rows.append(dict(protein=p, HR=np.exp(s["coef"]), beta=s["coef"], se=s["se(coef)"],
                             z=s["z"], p=s["p"], FDR=np.nan,
                             cindex=concordance_index(time90, -cph.predict_partial_hazard(d), event90)))
        except Exception:
            rows.append(dict(protein=p, HR=np.nan, beta=np.nan, se=np.nan, z=np.nan,
                             p=np.nan, FDR=np.nan, cindex=np.nan))
    cox_all = pd.concat([prim[["protein", "HR", "beta", "se", "z", "p", "FDR", "cindex"]],
                         pd.DataFrame(rows)], ignore_index=True)
    cox_all["ff"] = cox_all["protein"].map(ff_all)
    cox_all["retained"] = cox_all["ff"] <= 0.20
    cox_all.to_csv(CACHE, index=False)
print("all-883 Cox:", cox_all.shape, "| failed:", int(cox_all["p"].isna().sum()))
# top exploratory hits among the excluded proteins
cox_all[~cox_all["retained"]].nsmallest(15, "p")[
    ["protein", "HR", "z", "p", "ff"]].to_csv(OUT / "matrices" / "top_excluded_hits.csv", index=False)


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    q = p[ok].copy(); q[q <= 0] = 1e-300
    o = np.argsort(q); r = q[o] * len(q) / (np.arange(len(q)) + 1)
    f = np.minimum.accumulate(r[::-1])[::-1].clip(max=1)
    tmp = np.empty_like(f); tmp[o] = f; out[ok] = tmp
    return out


# ------------------------------------------------------------------ 2. threshold sweep
ths = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
prim_sig = set(prim.loc[prim["p"] < 0.05 / max(1, (prim["p"].rank().max() / 1)), "protein"])  # placeholder
prim_sig = set(prim.loc[prim["FDR"] < 0.05, "protein"])
sweep = []
for t in ths:
    sub = cox_all[cox_all["ff"] <= t].copy()
    sub["FDR_t"] = bh(sub["p"].to_numpy())
    sig = set(sub.loc[sub["FDR_t"] < 0.05, "protein"])
    sweep.append(dict(threshold=t, n_kept=len(sub), n_sig=len(sig),
                      pct_sig=100 * len(sig) / len(sub),
                      n_sig_recovered=len(sig & prim_sig),
                      n_sig_new=len(sig - prim_sig),
                      jaccard=len(sig & prim_sig) / max(1, len(sig | prim_sig))))
sweep = pd.DataFrame(sweep)
sweep.to_csv(OUT / "matrices" / "threshold_sweep.csv", index=False)
print(sweep.to_string(index=False))

# is prognostic strength related to detection?
from scipy import stats
r_all = stats.spearmanr(cox_all["ff"], cox_all["z"].abs(), nan_policy="omit")
r_sig = stats.spearmanr(cox_all.loc[cox_all["retained"], "ff"],
                        cox_all.loc[cox_all["retained"], "z"].abs(), nan_policy="omit")
ex_max = cox_all.loc[~cox_all["retained"]].nlargest(1, "z")
print(f"Spearman ff vs |z|: all rho={r_all.statistic:.3f} p={r_all.pvalue:.3g}; "
      f"retained rho={r_sig.statistic:.3f} p={r_sig.pvalue:.3g}")
print("strongest excluded protein:", ex_max[["protein", "HR", "z", "p"]].to_dict("records"))
n_ex_sig_raw = int((cox_all.loc[~cox_all["retained"], "p"] < 0.05).sum())
print("excluded with raw p<0.05:", n_ex_sig_raw, "/", int((~cox_all["retained"]).sum()))

# ------------------------------------------------------------------ 3. model variants
# Protocol v1.2 / v0.3: locked panel = union of the two pre-specified lockings
sel = ["TNF.RI", "Renin", "SREC.II", "DR6", "GH"]
sel = [p for p in sel if p in prot.columns]
P3 = ["TNF.RI", "DR6", "Renin"]               # parsimonious alternative
dev_mask = np.isin(center, ["nfa", "xna", "rja", "dta", "xha"])

Z = Zall[prot.columns]                        # primary standardized matrix (883, no filter)
d0 = base.copy()
for p in sorted(set(pool) | set(sel)):
    d0[p] = Zall[p].to_numpy()
dev, val = d0[dev_mask].copy(), d0[~dev_mask].copy()


def fit(df, cols, strata=True, extra=None):
    frame = df[["time", "event"] + (["center"] if strata else []) + cols].copy()
    if extra is not None:
        for k, v in extra.items():
            frame[k] = v
    cc = CoxPHFitter(penalizer=0.0)
    cc.fit(frame, "time", "event", strata=["center"] if strata else None)
    return cc, list(frame.columns)


def evaluate(cc, cols, test, label="", y=None, keep=None):
    x = test[["time", "event", "center"] + cols].copy()
    lin = np.log(np.asarray(cc.predict_partial_hazard(x)))
    y = test["event"].to_numpy().astype(int) if y is None else np.asarray(y).astype(int)
    t_all = test["time"].to_numpy()
    e_all = test["event"].to_numpy().astype(int)
    if keep is not None:
        keep = np.asarray(keep, bool)
        lin, y, t_all, e_all = lin[keep], y[keep], t_all[keep], e_all[keep]
    return dict(variant=label,
                auc90=roc_auc_score(y, lin),
                cindex=concordance_index(t_all, -lin, e_all),
                n_test=len(lin))


variants = []
# 1 primary
cc, cols = fit(dev, ["meld"] + sel)
variants.append(evaluate(cc, cols, val, "Primary (5 proteins, centre-stratified)"))
# 2 parsimonious 3-protein alternative (TNF-RI, DR6, Renin)
cc, cols = fit(dev, ["meld"] + P3)
variants.append(evaluate(cc, cols, val, "Parsimonious (3 proteins)"))
# 3 death-only endpoint (transplant censored, excluded from the AUC contrast)
dev_d = dev.copy(); dev_d["event"] = death90[dev_mask]
val_d = val.copy(); val_d["event"] = death90[~dev_mask]
cc, cols = fit(dev_d, ["meld"] + sel)
variants.append(evaluate(cc, cols, val_d, "Death only (transplant censored)",
                         y=death90[~dev_mask], keep=(alive90 | (death90 == 1))[~dev_mask]))
# 4 centre as covariate instead of strata
cc, cols = fit(dev, ["meld"] + sel, strata=False)
variants.append(evaluate(cc, cols, val, "Centre as covariate (no strata)"))
# 5 reversed split: train on the 8 validation centres, test on the 5 derivation centres
cc, cols = fit(val, ["meld"] + sel)
variants.append(evaluate(cc, cols, dev, "Reversed split (train n=395)"))
# 6 random 70/30 split
rng = np.random.default_rng(42)
idx = np.arange(len(d0))
tr = []
for e in [0, 1]:
    ii = idx[(d0["event"] == e).to_numpy()]
    tr.extend(rng.permutation(ii)[:int(round(0.7 * len(ii)))])
tr = np.array(sorted(tr)); te = np.array([i for i in idx if i not in set(tr)])
cc, cols = fit(d0.iloc[tr], ["meld"] + sel)
variants.append(evaluate(cc, cols, d0.iloc[te], "Random 70/30 split (seed 42)"))
var = pd.DataFrame(variants)
var.to_csv(OUT / "matrices" / "model_variants.csv", index=False)
print(var.to_string(index=False))

# ------------------------------------------------------------------ 4. bootstrap selection stability (protocol v1.2)
# Reuse the reselect_panel stability run: 91-protein derivation pool, 300
# bootstrap resamples, penalised fits at lambda_1se, largest-gap threshold.
bootf = pd.read_csv(BASE / "analysis" / "output2" / "stability_primary883.csv")
bootf["gene"] = [g2sym(pr) for pr in bootf["protein"]]
bootf["selected_primary"] = bootf["protein"].isin(sel)
bootf.sort_values("freq", ascending=False).to_csv(OUT / "matrices" / "bootstrap_selection.csv", index=False)
FREQ_CUT = 0.795
print(bootf.head(10).to_string(index=False))

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig = plt.figure(figsize=(9.4, 9.0))
gs = fig.add_gridspec(3, 6, height_ratios=[1, 1.05, 0.92], hspace=0.55, wspace=0.95)
axA = fig.add_subplot(gs[0, 0:2]); axB = fig.add_subplot(gs[0, 2:4]); axC = fig.add_subplot(gs[0, 4:6])
axD = fig.add_subplot(gs[1, 0:3]); axE = fig.add_subplot(gs[1, 3:6])
axF = fig.add_subplot(gs[2, 1:5])

# A detection-rate distribution
axA.text(-0.26, 1.09, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")
axA.hist(ff_all[ff_all <= 0.20] * 100, bins=np.arange(0, 21, 1), color="#4E79A7", alpha=0.9,
         label=f"would be retained, n={int((ff_all <= 0.20).sum())}")
axA.hist(ff_all[ff_all > 0.20] * 100, bins=np.arange(20, 62, 1.5), color="#C0504D", alpha=0.85,
         label=f"would be excluded, n={int((ff_all > 0.20).sum())}")
axA.axvline(20, color="#333333", ls="--", lw=0.9)
axA.text(22, axA.get_ylim()[1] * 0.55, "80% detection\ncut-off", fontsize=6, color="#333333")
axA.set_xlabel("Floor-value fraction (%)", fontsize=7)
axA.set_ylabel("Proteins", fontsize=7)
axA.legend(fontsize=6, frameon=False, loc="upper right")
axA.set_title("Hypothetical detection filter on 883 proteins\n(primary analysis applies no filter)", fontsize=8, loc="left")
axA.tick_params(labelsize=6)

# B prognostic strength vs detection
axB.text(-0.26, 1.09, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
ret = cox_all["retained"].to_numpy()
sig89 = cox_all["protein"].isin(prim_sig).to_numpy()
axB.scatter(cox_all.loc[ret & ~sig89, "ff"] * 100, cox_all.loc[ret & ~sig89, "z"].abs(),
            s=5, c="#BBBBBB", alpha=0.6, lw=0, label="not FDR<0.05")
axB.scatter(cox_all.loc[sig89, "ff"] * 100, cox_all.loc[sig89, "z"].abs(),
            s=14, c="#C0504D", alpha=0.9, lw=0, label="FDR<0.05 (primary)")
axB.scatter(cox_all.loc[~ret, "ff"] * 100, cox_all.loc[~ret, "z"].abs(),
            s=8, c="none", edgecolors="#2E5E8C", lw=0.5, label="excluded (re-tested)")
axB.axvline(20, color="#333333", ls="--", lw=0.9)
axB.set_xlabel("Floor-value fraction (%)", fontsize=7)
axB.set_ylabel("|z| of MELD-adjusted Cox", fontsize=7)
axB.legend(fontsize=5.6, frameon=False, loc="upper right")
axB.set_title(f"Prognostic strength vs detection\nall 883: ρ={r_all.statistic:.2f} (P={r_all.pvalue:.2f})",
              fontsize=7.5, loc="left")
axB.text(0.02, 0.015, f"retained only: ρ={r_sig.statistic:.2f}", transform=axB.transAxes,
         fontsize=5.4, color="#555555")
axB.tick_params(labelsize=6)

# C threshold sweep
axC.text(-0.26, 1.09, "C", transform=axC.transAxes, fontsize=12, fontweight="bold")
axC.plot(sweep["threshold"] * 100, sweep["n_kept"], "-o", ms=3, lw=1.1, color="#4E79A7",
         label="proteins retained")
axC.plot(sweep["threshold"] * 100, sweep["n_sig"], "-s", ms=3, lw=1.1, color="#C0504D",
         label="FDR<0.05")
axC.axvline(20, color="#333333", ls="--", lw=0.9)
axC.set_xlabel("Detection cut-off (% of patients with value above floor)", fontsize=7)
axC.set_ylabel("Number of proteins", fontsize=7)
axC2 = axC.twinx()
axC2.plot(sweep["threshold"] * 100, sweep["jaccard"], ":", lw=1.0, color="#2E7D4F")
axC2.set_ylabel("Jaccard vs primary list", fontsize=6.5, color="#2E7D4F")
axC2.tick_params(labelsize=6, colors="#2E7D4F"); axC2.set_ylim(0.5, 1.02)
axC.legend(fontsize=5.8, frameon=False, loc="upper left")
axC.set_title("Prognostic list is stable across cut-offs", fontsize=8, loc="left")
axC.tick_params(labelsize=6)

# D model variants
axD.text(-0.14, 1.09, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")
yv = np.arange(len(var))[::-1]
same_set = np.array([r["n_test"] == 395 for _, r in var.iterrows()])
axD.barh(yv, var["auc90"], height=0.58,
         color=["#C0504D"] + ["#4E79A7" if s else "#9FB3C8" for s in same_set[1:]])
for yy, (_, r) in zip(yv, var.iterrows()):
    axD.text(r["auc90"] + 0.004, yy, f"{r['auc90']:.3f} (C={r['cindex']:.3f}, n={r['n_test']})",
             va="center", fontsize=5.6)
axD.set_yticks(yv, var["variant"], fontsize=6.2)
axD.set_xlim(0.6, 1.07); axD.set_xlabel("AUC at 90 days (test set)", fontsize=7)
axD.plot([0.904, 0.904], [2.65, 5.4], color="#999999", ls="--", lw=0.8)
axD.text(0.906, 5.45, "MELD alone 0.904", fontsize=5.8, color="#777777")
axD.set_title("Model robustness across analysis choices", fontsize=8, loc="left")
axD.tick_params(labelsize=6)

# E bootstrap stability (91-protein pool, 300 resamples at lambda_1se)
axE.text(-0.14, 1.09, "E", transform=axE.transAxes, fontsize=12, fontweight="bold")
bs = bootf.sort_values("freq", ascending=False)
binsE = np.arange(0, 1.025, 0.05)
axE.hist(bs.loc[bs["freq"] < FREQ_CUT, "freq"], bins=binsE, color="#B9C6D2",
         label="below threshold (n=%d)" % int((bs.freq < FREQ_CUT).sum()))
axE.hist(bs.loc[bs["freq"] >= FREQ_CUT, "freq"], bins=binsE, color="#C0504D",
         label="selected (n=%d)" % int((bs.freq >= FREQ_CUT).sum()))
axE.axvline(FREQ_CUT, color="#2E7D4F", ls="--", lw=1.0)
axE.text(FREQ_CUT + 0.012, 0.97, "largest-gap threshold 0.795", transform=axE.get_xaxis_transform(),
         fontsize=6, ha="left", va="top", color="#2E7D4F")
axE.set_xticks(np.arange(0, 1.01, 0.2))
axE.set_xlim(0, 1.0)
axE.set_xlabel("Bootstrap selection frequency", fontsize=7)
axE.set_ylabel("Proteins (91-protein pool)", fontsize=7)
axE.legend(fontsize=6, frameon=False, loc="upper left")
axE.set_title("Stability selection, 300 resamples at $\\lambda_{1SE}$", fontsize=8, loc="left")
axE.tick_params(labelsize=6)
axE.text(0.98, 0.60, "Selected: " + ", ".join(bs.head(6)["gene"]), transform=axE.transAxes,
         fontsize=5.8, ha="right", color="#333333")

# F full re-derivation in the reversed split
axF.text(-0.075, 1.16, "F", transform=axF.transAxes, fontsize=12, fontweight="bold")
rep = pd.read_csv(OUT / "matrices" / "reversed_split_replication.csv")
lx = np.log2(rep["HR_dev"].to_numpy())
ly = np.log2(rep["HR_val"].to_numpy())
ok = rep["same_direction"].astype(bool).to_numpy()
pan = rep["in_reversed_panel"].astype(bool).to_numpy()
lim = 2.6
axF.axhline(0, color="#999999", lw=0.6, ls=":")
axF.axvline(0, color="#999999", lw=0.6, ls=":")
axF.plot([-lim, lim], [-lim, lim], color="#333333", lw=0.9, ls="--", zorder=1)
axF.scatter(lx[ok], ly[ok], s=18, facecolor="#4E79A7", edgecolor="white", lw=0.4, zorder=3,
            label=f"same direction, {int(ok.sum())}/29")
axF.scatter(lx[~ok], ly[~ok], s=26, marker="X", facecolor="#C0504D", edgecolor="white",
            lw=0.4, zorder=3, label=f"opposite direction, {int((~ok).sum())}/29")
axF.scatter(lx[pan], ly[pan], s=78, facecolor="none", edgecolor="#B07A2A", lw=1.1, zorder=4,
            label="in reversed-split panel")
OFFS = {"TNF.RI": (7, 3, "left"), "Renin": (9, -2, "left"),
        "TIMP.4": (-9, -4, "right"), "CEACAM.1": (-8, -12, "right")}
for _, r in rep.iterrows():
    if r["protein"] in OFFS:
        dx, dy, ha = OFFS[r["protein"]]
        axF.annotate(r["protein"].replace(".", "-"), (np.log2(r["HR_dev"]), np.log2(r["HR_val"])),
                     textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=6, color="#333333")
axF.set_xlim(-lim, lim)
axF.set_ylim(-lim, lim)
axF.set_aspect("equal")
axF.set_xlabel("log$_2$ HR in reversed derivation (8 small centres, 46 events)", fontsize=7)
axF.set_ylabel("log$_2$ HR in the 5 large centres\n(995 patients, 116 events)", fontsize=7)
axF.legend(fontsize=5.8, frameon=False, loc="lower right")
axF.set_title("Reversed split: effects replicate, but the newly derived panel adds nothing to MELD "
              "(ΔAUC −0.003)", fontsize=8, loc="left")
axF.tick_params(labelsize=6)

fig.suptitle("Supplementary Figure 1. The prognostic signature is robust to the protein detection filter and to modelling choices",
             fontsize=9.5, fontweight="bold", y=0.995)
fig.subplots_adjust(left=0.075, right=0.985, top=0.945, bottom=0.055)
fig.savefig(OUT / "output" / "figS1.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "output" / "figS1.png", dpi=300, bbox_inches="tight")
print("DONE")
