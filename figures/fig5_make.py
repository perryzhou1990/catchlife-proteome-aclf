# -*- coding: utf-8 -*-
"""Fig 5 — Robustness and subgroup specificity of the 5-protein prognostic panel (v0.3).
A  Leave-one-center-out CV: MELD vs MELD+panel (5 locked proteins), per-center C-index
B  Subgroup analysis (forest): delta C-index vs MELD, bootstrap 95% CI
   HBV+/HBV-, cirrhosis+/cirrhosis-, COSSH-ACLF, EASL-ACLF  (models trained on complement)
C  Organ-dysfunction x protein association heatmap (center-adjusted partial Spearman)
Outputs: output/fig5.pdf/.png, matrices/loco_results.csv, subgroup_results.csv,
         organ_correlation.csv
NOTE: age unavailable; respiration organ not analyzable (P/F recorded only in 16/1390 with
      impairment) -> 5 organ axes used.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

BASE = Path(__file__).resolve().parents[3]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\proteinarray_matrix_1390人蛋白质组结果（质控后883蛋白）.xlsx")
OUT = BASE / "figures" / "fig5"
for sub in ["raw_data", "matrices", "code", "output"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
clin.columns = [str(c).strip() for c in clin.columns]
clin["center"] = clin["ID"].astype(str).str.split("-").str[0]
prot = pd.read_excel(PROT, sheet_name="1390已测蛋白质组")
prot.columns = [str(c).strip() for c in prot.columns]
prot = prot.set_index(prot.columns[0]).loc[clin["ID"]].astype(float)
ff = (prot == prot.min()).mean()
prot = prot[ff[ff <= 0.20].index]
log2 = np.log2(prot.clip(lower=1))
Z = (log2 - log2.mean()) / log2.std(ddof=1)

t1y = pd.to_numeric(clin["OS_1y_更新"], errors="coerce")
s90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
time90 = np.minimum(t1y.fillna(90), 90).to_numpy(dtype=float)
event90 = s90.isin([1, 2]).astype(int).to_numpy()
tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr = pd.to_numeric(clin["INR"], errors="coerce")
inr = inr.fillna(inr.median())
cr = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce")
cr = cr.fillna(cr.median())
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.clip(lower=1))
        + 9.57 * np.log(cr.clip(lower=1)) + 6.43).clip(6, 40).to_numpy(dtype=float)
center = clin["center"].to_numpy()
sex = clin["Gender"].replace(2, np.nan)
sex = sex.fillna(sex.mode()[0]).astype(float).to_numpy()

PANEL = ["TNF.RI", "Renin", "DR6", "SREC.II", "GH"]   # locked panel (protocol v1.2 union)
PANEL = [p for p in PANEL if p in Z.columns]
d = pd.DataFrame({"time": time90, "event": event90, "meld": meld,
                  "center": center, "sex": sex})
for p in PANEL:
    d[p] = Z[p].to_numpy()

def fit(df, cols):
    cc = CoxPHFitter()
    use_c = df["center"].nunique() > 1
    cc.fit(df[["time", "event", "center"] + cols], "time", "event",
           strata=["center"] if use_c else None)
    return cc

def cidx(cc, df, cols):
    lp = np.asarray(cc.predict_partial_hazard(df[["time", "event", "center"] + cols]))
    try:
        return concordance_index(df["time"], -lp, df["event"])
    except ZeroDivisionError:
        return np.nan

# ---------------- A: leave-one-center-out ----------------
loco = []
for cen in sorted(pd.unique(center)):
    tr = d[d["center"] != cen]
    te = d[d["center"] == cen]
    if te["event"].sum() < 5:
        loco.append(dict(center=cen, n=len(te), events=int(te["event"].sum()),
                         c_meld=np.nan, c_prot=np.nan, dC=np.nan))
        continue
    m0, m1 = fit(tr, ["meld"]), fit(tr, ["meld"] + PANEL)
    c0, c1 = cidx(m0, te, ["meld"]), cidx(m1, te, ["meld"] + PANEL)
    loco.append(dict(center=cen, n=len(te), events=int(te["event"].sum()),
                     c_meld=c0, c_prot=c1, dC=c1 - c0))
loco = pd.DataFrame(loco).sort_values("n", ascending=False).reset_index(drop=True)
# anonymised centre labels: C1-C13 ordered by sample size (standard multicentre reporting)
loco["center_label"] = [f"C{i+1}" for i in range(len(loco))]
loco[["center_label", "center", "n", "events"]].to_csv(
    OUT / "matrices" / "center_mapping.csv", index=False)
loco.to_csv(OUT / "matrices" / "loco_results.csv", index=False)
ok = loco.dropna(subset=["dC"])
w = ok["events"] / ok["events"].sum()
print(loco.to_string(index=False))
print(f"weighted mean dC = {(ok['dC'] * w).sum():.3f}; "
      f"c_meld={np.nansum(ok['c_meld']*w):.3f} c_prot={np.nansum(ok['c_prot']*w):.3f}")

# ---------------- B: subgroups ----------------
hbv = (pd.to_numeric(clin["HBV"], errors="coerce") == 1).to_numpy()
cir = (pd.to_numeric(clin["cirrhosis"], errors="coerce") == 1).to_numpy()
cossh = (pd.to_numeric(clin["COSSH_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"],
                       errors="coerce") >= 1).to_numpy()
easl = (pd.to_numeric(clin["EASL_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"],
                      errors="coerce") >= 1).to_numpy()
subs = [("HBV aetiology", hbv), ("Non-HBV aetiology", ~hbv),
        ("Cirrhosis", cir), ("No cirrhosis", ~cir),
        ("COSSH-ACLF", cossh), ("EASL-ACLF", easl)]
rows = []
for label, mask in subs:
    tr, te = d[~mask], d[mask]
    m0, m1 = fit(tr, ["meld"]), fit(tr, ["meld"] + PANEL)
    c0, c1 = cidx(m0, te, ["meld"]), cidx(m1, te, ["meld"] + PANEL)
    # bootstrap within subgroup (models fixed)
    lp0 = -np.asarray(m0.predict_partial_hazard(te[["time", "event", "center"] + ["meld"]]))
    lp1 = -np.asarray(m1.predict_partial_hazard(te[["time", "event", "center"] + ["meld"] + PANEL]))
    tt, ee = te["time"].to_numpy(), te["event"].to_numpy()
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(te), len(te))
        if ee[idx].sum() < 5:
            continue
        try:
            boot.append(concordance_index(tt[idx], lp1[idx], ee[idx])
                        - concordance_index(tt[idx], lp0[idx], ee[idx]))
        except ZeroDivisionError:
            continue
    boot = np.array(boot)
    rows.append(dict(subgroup=label, n=len(te), events=int(te["event"].sum()),
                     c_meld=c0, c_prot=c1, dC=c1 - c0,
                     lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                     p_boot=float(min(1.0, (boot <= 0).mean() * 2)) if len(boot) else np.nan))
sub = pd.DataFrame(rows)
sub.to_csv(OUT / "matrices" / "subgroup_results.csv", index=False)
print(sub.to_string(index=False))

# ---------------- C: organ dysfunction x protein ----------------
ORGAN = {
    "Liver\n(TB)": np.log(pd.to_numeric(clin["TB(umol/L)"], errors="coerce").clip(lower=1)),
    "Coagulation\n(INR)": np.log(inr.clip(lower=0.8)),
    "Kidney\n(Cr)": np.log(cr.clip(lower=0.2)),
    "Circulation\n(MAP, -)": -pd.to_numeric(clin["MAP"], errors="coerce"),
    "Brain\n(HE any)": pd.to_numeric(clin["AD_HE(0.no1.yes)"], errors="coerce").gt(0).astype(float),
}
SHOW = PANEL + ["ANG.2", "HCC.1", "CRP"]
SHOW = [p for p in SHOW if p in Z.columns]
cen_dum = pd.get_dummies(pd.Series(center), drop_first=True).astype(float).to_numpy()

def adj_rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return np.nan, np.nan
    rx = stats.rankdata(x[ok]); ry = stats.rankdata(y[ok])
    X = np.column_stack([np.ones(ok.sum()), cen_dum[ok]])
    bx = np.linalg.lstsq(X, rx, rcond=None)[0]
    by = np.linalg.lstsq(X, ry, rcond=None)[0]
    ex, ey = rx - X @ bx, ry - X @ by
    r, p = stats.pearsonr(ex, ey)
    return r, p

M, Pv = np.zeros((len(ORGAN), len(SHOW))), np.ones((len(ORGAN), len(SHOW)))
for i, (oname, oser) in enumerate(ORGAN.items()):
    for j, pname in enumerate(SHOW):
        M[i, j], Pv[i, j] = adj_rho(Z[pname].to_numpy(), np.asarray(oser, dtype=float))
# BH-FDR over all cells with finite p
flat = Pv.ravel()
order = np.argsort(flat)
m = len(flat)
q = np.empty(m)
prev = 1.0
for k in range(m - 1, -1, -1):
    idx = order[k]
    prev = min(prev, flat[idx] * m / (k + 1))
    q[idx] = prev
Q = q.reshape(Pv.shape)
pd.DataFrame(M, index=list(ORGAN), columns=SHOW).to_csv(OUT / "matrices" / "organ_correlation.csv")
pd.DataFrame(Q, index=list(ORGAN), columns=SHOW).to_csv(OUT / "matrices" / "organ_correlation_qval.csv")
print("rho:\n", pd.DataFrame(M, index=list(ORGAN), columns=SHOW).round(2).to_string())

# ---------------- figure ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig = plt.figure(figsize=(10.2, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78], width_ratios=[1.12, 0.88],
                      hspace=0.52, wspace=0.30)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[1, :])

# A grouped bars per center
axA.text(-0.14, 1.06, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")
xp = np.arange(len(loco))
axA.bar(xp - 0.20, loco["c_meld"], width=0.38, color="#9E9E9E", label="MELD")
axA.bar(xp + 0.20, loco["c_prot"], width=0.38, color="#C0504D", label="MELD + 5-protein panel")
for i, r in loco.iterrows():
    if np.isfinite(r["dC"]):
        axA.text(i, max(r["c_meld"], r["c_prot"]) + 0.012, f"{r['dC']:+.2f}",
                 ha="center", fontsize=5.4, color="#7A2E24")
axA.set_xticks(xp, [f"{r['center_label']}\n{int(r['n'])}" for _, r in loco.iterrows()],
               fontsize=6.2, rotation=0, ha="center")
axA.text(1.0, -0.22, "lower row: n patients", transform=axA.transAxes,
         fontsize=5.8, ha="right", color="#555555")
axA.set_ylim(0.55, 1.10); axA.set_ylabel("C-index (held-out center)", fontsize=7)
axA.axhline(0.5, color="#CCCCCC", lw=0.6, ls=":")
axA.legend(fontsize=5.8, frameon=False, loc="upper left", ncol=1, bbox_to_anchor=(0.005, 0.99))
axA.set_title("Leave-one-centre-out validation (13 folds)", fontsize=8.5, loc="left")
axA.text(0.0, -0.22, "Centres anonymised as C1–C13, ordered by sample size",
         transform=axA.transAxes, fontsize=5.8, color="#555555")
axA.tick_params(labelsize=6)

# B forest
axB.text(-0.30, 1.06, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
yy = np.arange(len(sub))[::-1]
axB.axvline(0, color="#888888", ls="--", lw=0.8)
for k, (_, r) in enumerate(sub.iterrows()):
    col = "#C0504D" if r["dC"] > 0 else "#4E79A7"
    axB.plot([r["lo"], r["hi"]], [yy[k], yy[k]], color=col, lw=1.2)
    axB.plot(r["dC"], yy[k], "o", color=col, ms=4.5)
    axB.text(1.02, yy[k], f"{r['c_meld']:.2f}→{r['c_prot']:.2f}",
             transform=axB.get_yaxis_transform(), fontsize=5.4, va="center", color="#333333")
axB.set_yticks(yy, [f"{r['subgroup']} ({r['n']}/{r['events']})" for _, r in sub.iterrows()],
               fontsize=6.2)
axB.set_xlabel("ΔC-index (MELD + panel  −  MELD), 95% bootstrap CI", fontsize=7)
axB.set_title("Subgroup robustness (models trained on complement)", fontsize=8.5, loc="left")
axB.text(0.98, -0.20, "right: C-index MELD → MELD+panel", transform=axB.transAxes,
         fontsize=5.2, ha="right", color="#666666")
axB.tick_params(labelsize=6)
axB.set_xlim(min(-0.06, sub["lo"].min() - 0.01), max(0.20, sub["hi"].max() + 0.02))
axB.set_ylim(-0.55, len(sub) - 0.35)

# C heatmap
axC.text(-0.055, 1.10, "C", transform=axC.transAxes, fontsize=12, fontweight="bold")
im = axC.imshow(M, cmap="RdBu_r", vmin=-0.45, vmax=0.45, aspect="auto")
axC.set_xticks(range(len(SHOW)), [p.replace(".", "-") for p in SHOW], fontsize=7)
axC.set_yticks(range(len(ORGAN)), list(ORGAN.keys()), fontsize=6.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        star = "***" if Q[i, j] < 0.001 else "**" if Q[i, j] < 0.01 else "*" if Q[i, j] < 0.05 else ""
        axC.text(j, i, f"{M[i, j]:.2f}{star}", ha="center", va="center", fontsize=5.8,
                 color="white" if abs(M[i, j]) > 0.28 else "#222222")
cb = fig.colorbar(im, ax=axC, fraction=0.018, pad=0.012)
cb.set_label("center-adjusted partial Spearman ρ", fontsize=6.5)
cb.ax.tick_params(labelsize=5.5)
axC.set_title("Protein–organ dysfunction association (n=1390; * FDR<0.05, ** <0.01, *** <0.001)",
              fontsize=8.5, loc="left")
axC.tick_params(length=0)

fig.suptitle("Figure 5. The 5-protein panel improves risk stratification across centres and aetiologies, "
             "with gain concentrated in patients outside established ACLF",
             fontsize=10, fontweight="bold", y=0.985)
fig.savefig(OUT / "output" / "fig5.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "output" / "fig5.png", dpi=300, bbox_inches="tight")
print("DONE")
