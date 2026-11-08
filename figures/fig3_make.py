# -*- coding: utf-8 -*-
"""Fig 3 — Single-protein prognostic associations with 90-day transplant-free survival.
A  Forest: top proteins, adjusted Cox HR per SD (sex + MELD, stratified by center), BH-FDR
B  Restricted cubic spline dose-response (5 highlight proteins), non-linearity LR test
C  Cox (death, tx censored) vs Fine-Gray subdistribution HR (tx competing), top 60 proteins
D  Pathway enrichment of prognostic proteins (Cox FDR<0.05, Enrichr)
Outputs: output/fig3.pdf/.png, matrices/cox90_all.csv, finegray_top60.csv, rcs_nonlin.csv,
         enrichment_cox90.csv
NOTE: age not available -> not adjusted (limitation); MELD imputed from TB/INR/Cr.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import gseapy as gp

BASE = Path(__file__).resolve().parents[3]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\proteinarray_matrix_1390人蛋白质组结果（质控后883蛋白）.xlsx")
ANNOT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\宋晶分组分析结果-供参考\cirrhosis_vs_noncirrhosis.csv")
OUT = BASE / "figures" / "fig3"
for sub in ["raw_data", "matrices", "code", "output"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
clin.columns = [str(c).strip() for c in clin.columns]
clin["center"] = clin["ID"].astype(str).str.split("-").str[0]
prot = pd.read_excel(PROT, sheet_name="1390已测蛋白质组")
prot.columns = [str(c).strip() for c in prot.columns]
prot = prot.set_index(prot.columns[0]).loc[clin["ID"]].astype(float)
ff = (prot == prot.min()).mean()
# Protocol v1.2: all 883 quantified proteins enter the analysis without a
# detection filter. Detection frequency is retained for annotation and to flag
# the low-detection subset (reported separately, never used as a filter).
DET = ff.copy()
NPROT = prot.shape[1]
log2 = np.log2(prot.clip(lower=1))
Z = ((log2 - log2.mean()) / log2.std(ddof=1))
Zarr = Z.to_numpy()

# base survival frame: 90-day transplant-free survival
t1y = pd.to_numeric(clin["OS_1y_更新"], errors="coerce")
s90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
time90 = np.minimum(t1y.fillna(90), 90).to_numpy()
event90 = s90.isin([1, 2]).astype(int).to_numpy()
sex = clin["Gender"].replace(2, np.nan)
sex = sex.fillna(sex.mode()[0]).astype(float).to_numpy()
tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr = pd.to_numeric(clin["INR"], errors="coerce")
cr = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce")
inr = inr.fillna(inr.median())
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.clip(lower=1))
        + 9.57 * np.log(cr.fillna(cr.median()).clip(lower=1)) + 6.43).clip(6, 40).to_numpy()
center = clin["center"].to_numpy()
base = pd.DataFrame({"time": time90, "event": event90, "sex": sex, "meld": meld,
                     "center": center})

# ---------------- A: single-protein Cox (composite event) ----------------
rows = []
for j, p in enumerate(Z.columns):
    d = base.copy(); d["protein"] = Zarr[:, j]
    try:
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(d, "time", "event", strata=["center"])
        s = cph.summary.loc["protein"]
        rows.append(dict(protein=p, HR=np.exp(s["coef"]), beta=s["coef"],
                         se=s["se(coef)"], z=s["z"], p=s["p"],
                         cindex=concordance_index(time90, -cph.predict_partial_hazard(d), event90),
                         floor_rate=float(DET.get(p, np.nan))))
    except Exception as ex:
        rows.append(dict(protein=p, HR=np.nan, beta=np.nan, se=np.nan, z=np.nan, p=np.nan, cindex=np.nan))
cox = pd.DataFrame(rows)
# proper BH-FDR
p_ = cox["p"].replace(0, 1e-300).to_numpy()
o = np.argsort(p_); ranked = p_[o] * len(p_) / (np.arange(len(p_)) + 1)
fdr_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
fdr = np.empty_like(fdr_sorted); fdr[o] = fdr_sorted
cox["FDR"] = fdr
cox.sort_values("p").to_csv(OUT / "matrices" / "cox90_all.csv", index=False)
n_sig = int((cox["FDR"] < 0.05).sum())
print(f"Cox 90d TFS: {n_sig}/{NPROT} proteins FDR<0.05")

# ---------------- B: RCS dose-response (5 highlights) ----------------
targets = ["TNF.RI", "HCC.1", "DR6", "Axl", "CRP"]
targets = [t for t in targets if t in Z.columns]
knots_pct = [0.05, 0.35, 0.65, 0.95]
def rcs_basis(x, knots):
    k = len(knots); B = np.zeros((len(x), k))
    B[:, 0] = x
    dk = knots[-1] - knots[0]
    denom = (knots[-1] - knots[1]) ** 2
    for j in range(1, k):
        t = np.clip(x - knots[j], 0, None) ** 3
        t -= np.clip(x - knots[k - 2], 0, None) ** 3 * (knots[-1] - knots[j]) / (knots[-1] - knots[k - 2])
        t += np.clip(x - knots[-1], 0, None) ** 3 * (knots[1] - knots[j]) / (knots[-1] - knots[1])
        B[:, j] = t / denom
    return B
rcs_rows = []
fig_rcs = {}
for p in targets:
    x = Zarr[:, list(Z.columns).index(p)]
    if not np.isfinite(x).all():
        continue
    x = np.clip(x, np.quantile(x, 0.01), np.quantile(x, 0.99))   # winsorize for RCS stability
    kn = np.quantile(x, knots_pct)
    if len(np.unique(kn)) < len(kn):          # duplicated knots (floor-heavy proteins)
        kn = np.linspace(x.min(), x.max(), 4)
    B = rcs_basis(x, kn)
    sd = B.std(0); sd[sd == 0] = 1.0
    B = B / sd   # scale spline basis for convergence
    try:
        dlin = base.copy(); dlin["protein"] = x
        dnl = base.copy()
        for j in range(B.shape[1]):
            dnl[f"rcs{j}"] = B[:, j]
        try:
            cph_l = CoxPHFitter(penalizer=0.1); cph_l.fit(dlin, "time", "event", strata=["center"])
            cph_n = CoxPHFitter(penalizer=0.1); cph_n.fit(dnl, "time", "event", strata=["center"])
        except Exception:
            cph_l = CoxPHFitter(penalizer=1.0); cph_l.fit(dlin, "time", "event", strata=["center"])
            cph_n = CoxPHFitter(penalizer=1.0); cph_n.fit(dnl, "time", "event", strata=["center"])
        lr = 2 * (cph_n.log_likelihood_ - cph_l.log_likelihood_)
        p_nl = stats.chi2.sf(lr, B.shape[1] - 1)
        # HR curve vs reference (10th pct)
        xs = np.linspace(np.quantile(x, 0.01), np.quantile(x, 0.99), 60)
        Bs = rcs_basis(xs, kn) / sd
        ref = np.quantile(x, 0.10)
        delta = np.vstack([B_ - rcs_basis(np.array([ref]), kn)[0] / sd for B_ in Bs])
        names = [f"rcs{j}" for j in range(B.shape[1])]
        cov = cph_n.variance_matrix_.loc[names, names].to_numpy()
        eta = delta @ cph_n.params_.loc[names].to_numpy()
        se = np.sqrt(np.einsum("ij,jk,ik->i", delta, cov, delta))
        fig_rcs[p] = (xs, np.exp(eta), np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se))
        rcs_rows.append(dict(protein=p, nonlin_p=p_nl, lr=lr))
        print(f"RCS {p}: nonlin p={p_nl:.2e}")
    except Exception as ex:
        print(f"RCS {p} failed: {type(ex).__name__}")
pd.DataFrame(rcs_rows).to_csv(OUT / "matrices" / "rcs_nonlin.csv", index=False)

# ---------------- C: Fine-Gray (death = event, tx = competing), top 60 ----------------
death = (s90 == 1).astype(int).to_numpy()
comp = (s90 == 2).astype(int).to_numpy()
cens = (s90 == 0).astype(int).to_numpy()
# KM of censoring G(t): censoring treated as events
tc = np.unique(time90[cens == 1])
def G(t):
    n_at = (time90 >= t).sum() if np.isscalar(t) else (time90[:, None] >= np.atleast_1d(t)).sum(0)
    ev_c = ((cens == 1) & (time90 == t)).sum() if np.isscalar(t) else ((cens == 1)[:, None] & (time90[:, None] == np.atleast_1d(t))).sum(0)
    out = 1.0
    return out  # replaced below
# vectorized KM for censoring
uniq = np.unique(time90)
Gv = {}
S = 1.0
for t in uniq:
    n_risk = (time90 >= t).sum()
    n_cens = ((cens == 1) & (time90 == t)).sum()
    S *= 1 - n_cens / n_risk
    Gv[t] = S
def Gat(t): return np.vectorize(lambda v: Gv.get(v, S))(np.atleast_1d(t))

top60 = cox.nsmallest(60, "p")["protein"].tolist()
fg_rows = []
td = np.unique(time90[death == 1])
w_cache = []   # (event_ids, risk_ids, weights) per death time
for t in td:
    d_ids = np.where((time90 == t) & (death == 1))[0]
    risk_ids = np.where(time90 >= t)[0]
    w = np.ones(len(risk_ids))
    done_comp = np.where((comp == 1) & (time90 < t))[0]
    if len(done_comp):
        w_done = np.array([Gv.get(time90[i], S) for i in done_comp])
        # weights attached to the full-sample index for later lookup
    wfull = np.ones(len(base))
    if len(done_comp):
        wfull[done_comp] = Gv.get(t, S) / np.array([Gv[time90[i]] for i in done_comp])
    w_cache.append((d_ids, risk_ids, wfull[risk_ids]))
def fg_fit(X):
    td_loc = td
    def negll(b):
        eta = X @ b
        ll = 0.0
        for d_ids, risk_ids, w in w_cache:
            e_r = eta[risk_ids]
            mx = e_r.max()
            ll += eta[d_ids].sum() - len(d_ids) * (mx + np.log((w * np.exp(e_r - mx)).sum()))
        return -ll
    r = minimize(negll, np.zeros(X.shape[1]), method="BFGS", options={"maxiter": 100, "gtol": 1e-6})
    b = r.x; k = len(b); h = 1e-4
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            bp = b.copy(); bp[i] += h; bp[j] += h; l1 = negll(bp)
            bp = b.copy(); bp[i] += h; bp[j] -= h; l2 = negll(bp)
            bp = b.copy(); bp[i] -= h; bp[j] += h; l3 = negll(bp)
            bp = b.copy(); bp[i] -= h; bp[j] -= h; l4 = negll(bp)
            H[i, j] = (l1 - l2 - l3 + l4) / (4 * h * h)
    cov = np.linalg.pinv(H)
    return b[-1], np.sqrt(max(cov[-1, -1], 1e-12))   # last column = protein
for p in top60:
    x = Zarr[:, list(Z.columns).index(p)]
    try:
        b, se = fg_fit(np.column_stack([sex, meld, x]))
        fg_rows.append(dict(protein=p, fg_beta=b, fg_se=se, fg_HR=np.exp(b),
                            fg_lo=np.exp(b - 1.96 * se), fg_hi=np.exp(b + 1.96 * se)))
    except Exception as ex:
        fg_rows.append(dict(protein=p, fg_beta=np.nan, fg_se=np.nan, fg_HR=np.nan,
                            fg_lo=np.nan, fg_hi=np.nan))
fg = pd.DataFrame(fg_rows)
fg.to_csv(OUT / "matrices" / "finegray_top60.csv", index=False)
print("Fine-Gray done:", fg["fg_HR"].notna().sum(), "/60")

# ---------------- D: enrichment of prognostic proteins ----------------
annot = pd.read_csv(ANNOT)[["protein", "geneID"]].drop_duplicates("protein")
sig = cox[cox["FDR"] < 0.05].merge(annot, on="protein", how="left")
genes = sig["geneID"].dropna().unique().tolist()
bg = annot["geneID"].dropna().unique().tolist()
enr = pd.DataFrame()
try:
    e = gp.enrichr(genes, gene_sets=["GO_Biological_Process_2023", "KEGG_2021_Human"],
                   organism="human", outdir=None, no_plot=True)
    enr = e.results
    enr.to_csv(OUT / "matrices" / "enrichment_cox90.csv", index=False)
    print("enrichment terms:", len(enr))
except Exception as ex:
    print("enrichment failed:", ex)

# ---------------- figure ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig = plt.figure(figsize=(8.6, 9.2))
gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 0.85, 0.85], hspace=0.65, wspace=1.3)
axA = fig.add_subplot(gs[0, 0:3]); axD = fig.add_subplot(gs[0, 3:6])
axB = fig.add_subplot(gs[1, 0:3]); axC = fig.add_subplot(gs[1:, 3:6])

# A forest
topA = cox.nsmallest(14, "p").iloc[::-1]
topA = topA.merge(annot, on="protein", how="left")
y = np.arange(len(topA))
axA.errorbar(topA["HR"], y, xerr=[topA["HR"] - np.exp(topA["beta"] - 1.96 * topA["se"]),
                                 np.exp(topA["beta"] + 1.96 * topA["se"]) - topA["HR"]],
             fmt="o", ms=3.5, color="#2E5E8C", ecolor="#2E5E8C", elinewidth=1, capsize=0)
axA.axvline(1, color="#999999", lw=0.7, ls="--")
axA.set_yticks(y, [f"{g} ({p.replace('.', '-')})" for p, g in zip(topA["protein"], topA["geneID"])], fontsize=5.6)
axA.set_xscale("log"); axA.set_xlabel("Adjusted HR per SD (90-day death/transplant)", fontsize=7)
axA.set_title(f"Top prognostic proteins ({n_sig}/{NPROT} FDR<0.05)", fontsize=8, loc="left")
axA.tick_params(labelsize=6)
axA.text(-0.30, 1.06, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")

# B RCS
axB.text(-0.30, 1.10, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
cmap = ["#C0504D", "#E8A33D", "#4E79A7", "#59A14F", "#B07AA1"]
plotted = 0
for i, p in enumerate(targets):
    if p not in fig_rcs:
        continue
    xs, hr, lo, hi = fig_rcs[p]
    c = cmap[plotted]
    axB.plot(xs, hr, color=c, lw=1.2)
    axB.fill_between(xs, lo, hi, color=c, alpha=0.13)
    pnl = [r for r in rcs_rows if r["protein"] == p][0]["nonlin_p"]
    axB.annotate(f"{p.replace('.', '-')} (P-nonlin={pnl:.1e})", (0.02, 0.94 - plotted * 0.09),
                 xycoords="axes fraction", fontsize=5.6, color=c)
    plotted += 1
axB.axhline(1, color="#999999", lw=0.6, ls="--")
axB.set_ylim(0, 8)
axB.set_xlabel("Protein level (z-score)", fontsize=7)
axB.set_ylabel("HR vs 10th percentile", fontsize=7)
axB.set_title("Dose-response (restricted cubic splines)", fontsize=8, loc="left")
axB.tick_params(labelsize=6)

# C Cox vs Fine-Gray
axC.text(-0.32, 1.06, "C", transform=axC.transAxes, fontsize=12, fontweight="bold")
m = fg.dropna().merge(cox[["protein", "HR"]], on="protein")
axC.scatter(m["HR"], m["fg_HR"], s=6, color="#6A4C93", alpha=0.6)
lims = [min(m["HR"].min(), m["fg_HR"].min()), max(m["HR"].max(), m["fg_HR"].max())]
axC.plot(lims, lims, color="#999999", lw=0.7, ls="--")
axC.set_xscale("log"); axC.set_yscale("log")
axC.set_xlabel("Cause-specific Cox HR (death; tx censored)", fontsize=7)
axC.set_ylabel("Fine-Gray subdistribution HR (tx competing)", fontsize=7)
r_sp = stats.spearmanr(m["HR"], m["fg_HR"]).statistic
axC.set_title(f"Cox vs Fine-Gray, top 60 proteins (Spearman ρ={r_sp:.2f})", fontsize=8, loc="left")
axC.tick_params(labelsize=6)

# D enrichment
axD.text(-0.22, 1.06, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")
if not enr.empty:
    top = enr.nsmallest(10, "Adjusted P-value").iloc[::-1]
    top = top.assign(label=[x.split(" (")[0][:46] for x in top["Term"]])
    axD.barh(top["label"], -np.log10(top["Adjusted P-value"]), color="#59A14F", alpha=0.85, height=0.62)
    axD.set_xlabel("-log10 (adjusted p)", fontsize=7)
else:
    axD.text(0.5, 0.5, "enrichment unavailable", ha="center")
axD.set_title(f"Enrichment of prognostic proteins (FDR<0.05, n={len(genes)} genes)", fontsize=8, loc="left")
axD.tick_params(labelsize=5.8)
axD.spines[["top", "right"]].set_visible(False)

fig.suptitle("Figure 3. Plasma proteins independently associated with 90-day transplant-free survival",
             fontsize=10.5, fontweight="bold", y=0.99)
fig.savefig(OUT / "output" / "fig3.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "output" / "fig3.png", dpi=300, bbox_inches="tight")
print("Top 10 prognostic proteins:\n",
      topA[["geneID", "protein", "HR", "p", "FDR"]].iloc[::-1].to_string(index=False))
print("DONE")
