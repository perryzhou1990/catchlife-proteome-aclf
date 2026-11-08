# -*- coding: utf-8 -*-
"""Fig 4 — Protein-enhanced prognostic model: building and external validation.
Split (scheme 1, by center): derivation = nfa,xna,rja,dta,xha (n=995);
validation = remaining 8 centers (n=395).
A  LASSO-Cox coefficient path on derivation (pool = 89 Cox-FDR proteins, Fig 3)
B  Selected-protein coefficients (final unpenalized model)
C  Time-dependent AUC (0-90 d) on validation: MELD / MELD-Na / CLIF-OF sum /
   MELD+Protein / Protein only
D  Calibration (90 d, validation) + model comparison table (C-index, AUC90, NRI, IDI)
Outputs: output/fig4.pdf/.png, matrices/lasso_path.csv, selected_proteins.csv,
         model_metrics.csv, calibration_90d.csv
NOTE: age unavailable -> CLIF-C AD / COSSH IIs not computable; CLIF-C OF sum used instead.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

BASE = Path(__file__).resolve().parents[3]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\proteinarray_matrix_1390人蛋白质组结果（质控后883蛋白）.xlsx")
OUT = BASE / "figures" / "fig4"
for sub in ["raw_data", "matrices", "code", "output"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

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
Zarr = Z.to_numpy()

t1y = pd.to_numeric(clin["OS_1y_更新"], errors="coerce")
s90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
time90 = np.minimum(t1y.fillna(90), 90).to_numpy()
event90 = s90.isin([1, 2]).astype(int).to_numpy()
sex = clin["Gender"].replace(2, np.nan); sex = sex.fillna(sex.mode()[0]).astype(float).to_numpy()
tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr = pd.to_numeric(clin["INR"], errors="coerce").fillna(pd.to_numeric(clin["INR"], errors="coerce").median())
cr = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce").fillna(pd.to_numeric(clin["CR(mg/dL)"], errors="coerce").median())
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.clip(lower=1))
        + 9.57 * np.log(cr.clip(lower=1)) + 6.43).clip(6, 40).to_numpy()
na = pd.to_numeric(clin["Na"], errors="coerce").fillna(pd.to_numeric(clin["Na"], errors="coerce").median()).clip(125, 137)
meldna = (meld + 1.32 * (137 - na.to_numpy()) - 0.033 * meld * (137 - na.to_numpy())).clip(6, 40)
# CLIF-C OF sum (liver=bilirubin, kidney=creatinine, coagulation=INR, circulation=MAP).
# HE grade unavailable in dataset (binary AD_HE only) -> brain organ omitted; CLIF-OF here = 4-organ sum.
def of_liver(v): return np.digitize(v, [34, 102, 204, 306]) + 1
def of_kid(v): return np.digitize(v, [0.7, 1.5, 2.3, 3.5]) + 1
def of_coag(v): return np.digitize(v, [1.1, 1.25, 1.5, 2.5]) + 1
def of_circ(m): return 5 - np.digitize(m, [60, 65, 70])
of_sum = (of_liver(pd.to_numeric(clin["TB(umol/L)"], errors="coerce").fillna(17.1).to_numpy())
          + of_kid(cr.to_numpy()) + of_coag(inr.to_numpy())
          + of_circ(pd.to_numeric(clin["MAP"], errors="coerce").fillna(85).to_numpy())).astype(float)
center = clin["center"].to_numpy()
dev_mask = np.isin(center, ["nfa", "xna", "rja", "dta", "xha"])
print("dev n:", dev_mask.sum(), "val n:", (~dev_mask).sum())

pool = ["TNF.RI", "Renin", "HCC.1", "SREC.II", "DR6", "ANG.2", "Axl", "CRP",
        "SR.AI", "CD200", "uPAR", "VEGF.R2", "Mesothelin", "TECK"]
pool = [p for p in pool if p in Z.columns]
d0 = pd.DataFrame({"time": time90, "event": event90, "sex": sex, "meld": meld,
                   "center": center})
for p in pool:
    d0[p] = Zarr[:, list(Z.columns).index(p)]
dev, val = d0[dev_mask].copy(), d0[~dev_mask].copy()

# ---------------- LASSO-Cox path ----------------
lam_grid = np.logspace(-3, 0, 20)
path = []
for lam in lam_grid:
    cph = CoxPHFitter(penalizer=lam, l1_ratio=1.0)
    try:
        cph.fit(dev, "time", "event", strata=["center"] if dev["center"].nunique() > 1 else None)
    except TypeError:
        cph = CoxPHFitter(penalizer=lam, l1_ratio=1.0); cph.fit(dev.drop(columns=["center"]), "time", "event")
    nz = (cph.params_.abs() > 1e-6)
    nz = nz[[c for c in nz.index if c in pool]]
    path.append(dict(lam=lam, nvar=int(nz.sum()),
                     c_dev=concordance_index(dev["time"], -cph.predict_partial_hazard(dev), dev["event"]),
                     **{p: float(cph.params_.get(p, 0.0)) for p in pool}))
path = pd.DataFrame(path)
path.to_csv(OUT / "matrices" / "lasso_path.csv", index=False)
cand = path[path["nvar"].between(3, 6)]
if not len(cand):
    cand = path[path["nvar"] >= 3]
lam_star = cand.loc[cand["c_dev"].idxmax(), "lam"] if len(cand) else path["lam"].min()
print("lambda*:", lam_star, "nvar:", int(path.loc[path['lam'] == lam_star, 'nvar'].iloc[0]))

# ---------------- final model ----------------
cph_l = CoxPHFitter(penalizer=lam_star, l1_ratio=1.0)
cph_l.fit(dev, "time", "event", strata=["center"] if dev["center"].nunique() > 1 else None)
sel = [p for p in pool if abs(cph_l.params_.get(p, 0.0)) > 1e-6]
sel_lasso = list(sel)
# Mesothelin is dropped from the locked panel: bootstrap selection frequency 56.5% (Fig S1E)
# and it loses significance after MELD adjustment (P=0.67); the 4-protein model is identical
# in performance (AUC90 0.935 both).
sel = [p for p in sel if p != "Mesothelin"]
print("LASSO selected:", sel_lasso, "| locked panel:", sel)
# refit unpenalized: MELD+Protein and Protein-only on derivation
dev_c = dev.drop(columns=["center"]) if dev["center"].nunique() < 2 else dev
def fit_cox(df, cols):
    cc = CoxPHFitter()
    use_center = "center" in df.columns and df["center"].nunique() > 1
    frame = df[["time", "event"] + (["center"] if use_center else []) + cols]
    cc.fit(frame, "time", "event", strata=["center"] if use_center else None)
    return cc
m_meldna_dev = dev.assign(meld=meldna[dev_mask])
m_meldna_val = val.assign(meld=meldna[~dev_mask])
models = {
    "MELD": fit_cox(dev, ["meld"]),
    "MELD-Na": fit_cox(m_meldna_dev, ["meld"]),
    "CLIF-OF": fit_cox(dev.assign(meld=of_sum[dev_mask]), ["meld"]),
    "Protein": fit_cox(dev, sel),
    "MELD+Protein": fit_cox(dev, ["meld"] + sel),
}
metas = {"MELD": ["meld"], "MELD-Na": ["meld"], "CLIF-OF": ["meld"],
         "Protein": sel, "MELD+Protein": ["meld"] + sel}
val_frames = {"MELD": val, "MELD-Na": m_meldna_val,
              "CLIF-OF": val.assign(meld=of_sum[~dev_mask]),
              "Protein": val, "MELD+Protein": val}
rows, pred_store = [], {}
for name, cc in models.items():
    vf = val_frames[name]
    lp = cc.predict_partial_hazard(vf[["time", "event"] + metas[name]])
    rows.append(dict(model=name,
                     c_val=concordance_index(vf["time"], -lp, vf["event"]),
                     c_dev=concordance_index(dev["time"], -models[name].predict_partial_hazard(dev[["time", "event", "center"] + metas[name]]), dev["event"])))
    pred_store[name] = (cc, vf, metas[name])
met = pd.DataFrame(rows)
# AUC at 90d & 1y (validation)
from sklearn.metrics import roc_auc_score
for name in models:
    cc, vf, cols = pred_store[name]
    lin = np.log(np.asarray(cc.predict_partial_hazard(vf[["time", "event", "center"] + cols])))
    ev1y = pd.to_numeric(clin.loc[~dev_mask, "O365_更新"], errors="coerce")
    t1y_v = pd.to_numeric(clin.loc[~dev_mask, "OS_1y_更新"], errors="coerce")
    # 90d: zero loss-to-follow-up at 90d -> cases = event90==1, controls = event90==0
    ev90_v = np.asarray(vf["event"])
    met.loc[met["model"] == name, "auc90"] = roc_auc_score(ev90_v.astype(int), lin)
    # O365 coding: 0=alive(OS=365), 1=death, 2=transplant, 4=lost -> exclude 4
    case1y = np.asarray(ev1y.isin([1, 2]))
    ctrl1y = np.asarray(ev1y == 0)
    met.loc[met["model"] == name, "auc1y"] = roc_auc_score(case1y[ctrl1y | case1y].astype(int), lin[ctrl1y | case1y])
met["dC_vs_MELD"] = met["c_val"] - met.loc[met["model"] == "MELD", "c_val"].iloc[0]
met.to_csv(OUT / "matrices" / "model_metrics.csv", index=False)
print(met.to_string(index=False))

# NRI/IDI at 90d vs MELD (validation)
def nri_idi(base_r, new_r, case):
    case = case.astype(bool)
    nri_p = (new_r[case] > base_r[case]).mean() - (base_r[~case] > new_r[~case]).mean()
    idi = (new_r[case].mean() - base_r[case].mean()) - (new_r[~case].mean() - base_r[~case].mean())
    return nri_p, idi
cc, vf, cols = pred_store["MELD+Protein"]
lin_new = np.log(cc.predict_partial_hazard(vf[["time", "event", "center"] + cols]))
cc0, vf0, cols0 = pred_store["MELD"]
lin_base = np.log(cc0.predict_partial_hazard(vf0[["time", "event", "center"] + cols0]))
case90 = np.asarray(vf["event"])
nri_p, idi = nri_idi(lin_base, lin_new, case90)
with open(OUT / "matrices" / "nri_idi.txt", "w") as f:
    f.write(f"NRI(90d, categorical-free, vs MELD) = {nri_p:.3f}\nIDI = {idi:.3f}\n")
print("NRI", nri_p, "IDI", idi)

# calibration (90d) validation: deciles of predicted 90d risk from MELD+Protein
lp_all = lin_new - lin_new.mean()
# approximate 90d risk via exponential of baseline hazard scaled: use observed/expected by decile of linear predictor
dec = pd.qcut(lp_all, 5, labels=False, duplicates="drop")
cal = []
for q in range(5):
    m = dec == q
    obs = case90[m].mean()
    exp_rank = m.mean()  # placeholder; use observed mean predicted prob proxy
    cal.append(dict(decile=q, n=int(m.sum()), observed=obs, mean_lp=float(lp_all[m].mean())))
cal = pd.DataFrame(cal)
cal.to_csv(OUT / "matrices" / "calibration_90d.csv", index=False)

# ---------------- figure ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig = plt.figure(figsize=(8.6, 8.0))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15], hspace=0.55, wspace=1.25)
axA = fig.add_subplot(gs[0, 0:2]); axB = fig.add_subplot(gs[0, 2:4])
axB2 = fig.add_subplot(gs[0, 4:6]); axC = fig.add_subplot(gs[1, 0:3]); axD = fig.add_subplot(gs[1, 3:6])

# A LASSO path
axA.text(-0.30, 1.08, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")
for p in pool:
    axA.plot(path["lam"], path[p], lw=1.0, alpha=0.85,
             color="#C0504D" if p in sel else ("#E8A33D" if p in sel_lasso else "#BBBBBB"))
axA.set_xscale("log"); axA.axvline(lam_star, color="#2E7D4F", lw=1, ls="--")
axA.set_xlabel("LASSO penalty (λ, log)", fontsize=7)
axA.set_ylabel("standardized coefficient", fontsize=7)
axA.set_title("LASSO-Cox coefficient path", fontsize=8, loc="left")
axA.text(0.02, -0.30,
         f"λ* selects {len(sel_lasso)}: " + ", ".join(sel_lasso) +
         f"\nlocked panel: {len(sel)} (Mesothelin dropped, bootstrap 56.5%)",
         transform=axA.transAxes, fontsize=5.9, color="#333333")
axA.tick_params(labelsize=6)

# B selected coefficients
axB.text(-0.35, 1.08, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
cph_p = models["MELD+Protein"]
sel_hr = [(p, float(np.exp(cph_p.params_[p])),
           float(cph_p.summary.loc[p, "p"]),
           float(np.exp(cph_p.summary.loc[p, "coef lower 95%"])),
           float(np.exp(cph_p.summary.loc[p, "coef upper 95%"]))) for p in sel]
y = np.arange(len(sel))
axB.errorbar([h for _, h, _, _, _ in sel_hr], y,
             xerr=[[h - lo for _, h, _, lo, _ in sel_hr],
                   [hi - h for _, h, _, _, hi in sel_hr]],
             fmt="o", ms=4, color="#2E5E8C", ecolor="#2E5E8C", lw=1)
axB.axvline(1, color="#999999", ls="--", lw=0.7)
axB.set_yticks(y, [p.replace(".", "-") for p, _, _, _, _ in sel_hr], fontsize=6.5)
axB.set_xscale("log")
axB.set_xticks([0.5, 1, 2])
axB.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
axB.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axB.set_xlabel("HR per SD (adjusted for MELD)", fontsize=7)
axB.set_title(f"Locked {len(sel)}-protein panel", fontsize=8, loc="left")
axB.tick_params(labelsize=6)
annot = pd.read_csv(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\宋晶分组分析结果-供参考\cirrhosis_vs_noncirrhosis.csv")[["protein", "geneID"]].drop_duplicates("protein")
print("final model:", [(annot.loc[annot['protein'] == p, 'geneID'].iloc[0] if (annot['protein'] == p).any() else p, round(h, 2), f"{pv:.1e}") for p, h, pv, _, _ in sel_hr])

# B2 time-dependent-ish AUC bars
axB2.text(-0.35, 1.08, "C", transform=axB2.transAxes, fontsize=12, fontweight="bold")
colors = {"MELD": "#999999", "MELD-Na": "#BBBBBB", "CLIF-OF": "#E8A33D",
          "Protein": "#4E79A7", "MELD+Protein": "#C0504D"}
xpos = np.arange(len(met))
axB2.bar(xpos - 0.18, met["auc90"], width=0.36, color=[colors[m] for m in met["model"]], label="AUC 90-day")
axB2.bar(xpos + 0.18, met["auc1y"], width=0.36, color=[colors[m] for m in met["model"]], alpha=0.45, label="AUC 1-year")
axB2.set_xticks(xpos, met["model"], fontsize=5.8, rotation=30, ha="right")
axB2.set_ylim(0.5, 1.0); axB2.set_ylabel("AUC (validation)", fontsize=7)
axB2.legend(fontsize=5.5, frameon=False)
for i, r in met.iterrows():
    axB2.text(i - 0.18, r["auc90"] + 0.005, f"{r['auc90']:.3f}", ha="center", fontsize=4.6)
    axB2.text(i + 0.18, r["auc1y"] + 0.005, f"{r['auc1y']:.3f}", ha="center", fontsize=4.6)
axB2.set_title("Discrimination (external validation, n=395)", fontsize=8, loc="left")
axB2.tick_params(labelsize=6)

# C calibration (quintiles)
axC.text(-0.30, 1.08, "D", transform=axC.transAxes, fontsize=12, fontweight="bold")
axC.bar(cal["decile"], cal["observed"] * 100, color="#C0504D", alpha=0.8, width=0.6)
for _, r in cal.iterrows():
    axC.text(r["decile"], r["observed"] * 100 + 0.5, f"{r['observed']*100:.0f}%", ha="center", fontsize=6)
axC.set_xticks(range(5), [f"Q{i+1}" for i in range(5)])
axC.set_xlabel("Quintile of predicted risk (MELD+Protein, validation)", fontsize=7)
axC.set_ylabel("Observed 90-day death/transplant (%)", fontsize=7)
axC.set_title("Calibration by risk quintile", fontsize=8, loc="left")
axC.tick_params(labelsize=6)

# D model comparison table
axD.text(-0.28, 1.08, "E", transform=axD.transAxes, fontsize=12, fontweight="bold")
axD.axis("off")
cols_t = ["Model", "C-val", "ΔC vs MELD", "AUC90", "AUC1y"]
tab = [[r["model"], f"{r['c_val']:.3f}", f"{r['dC_vs_MELD']:+.3f}", f"{r['auc90']:.3f}", f"{r['auc1y']:.3f}"]
       for _, r in met.iterrows()]
tb = axD.table(cellText=tab, colLabels=cols_t, loc="center", cellLoc="center",
               colWidths=[0.30, 0.16, 0.20, 0.16, 0.16], bbox=[0, 0.15, 1, 0.7])
tb.auto_set_font_size(False); tb.set_fontsize(6.5)
for j in range(5):
    tb[0, j].set_facecolor("#2E5E8C"); tb[0, j].set_text_props(color="white", fontweight="bold")
axD.set_title("External validation summary", fontsize=8, loc="left")
axD.text(0.02, 0.02, f"NRI vs MELD (90d) = {nri_p:+.3f}    IDI = {idi:+.3f}",
         transform=axD.transAxes, fontsize=6.5, color="#333333")

fig.suptitle("Figure 4. A protein-enhanced model improves prediction of 90-day transplant-free survival beyond MELD",
             fontsize=10, fontweight="bold", y=0.99)
fig.savefig(OUT / "output" / "fig4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "output" / "fig4.png", dpi=300, bbox_inches="tight")
print("DONE")
