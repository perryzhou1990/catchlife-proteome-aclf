# -*- coding: utf-8 -*-
"""Fig 4 — Protein-enhanced prognostic model (protocol v1.2, manuscript v0.3).

Screening set: all 883 quantified proteins, NO detection filter (low-detection
proteins are flagged and reported separately, never discarded).
Split by centre: derivation = nfa,xna,rja,dta,xha (n=995, 116 events);
validation = the remaining 8 centres (n=395, 46 events).
Candidate pool: 91 proteins at BH-FDR<0.05 in the derivation set only.
Locked panel: union of the two pre-specified primary lockings
    primary883 -> TNF-RI, DR6, Renin
    filter665  -> TNF-RI, DR6, Renin, SREC-II, GH
    => 5 proteins (TNF-RI, DR6, Renin, SREC-II, GH).

A  LASSO-Cox coefficient paths over the 91-protein pool, lambda_1se marked
B  AUC at 90 days in validation (8 held-out centres) across nine models, including MELD 3.0 and
   COSSH-ACLF II (scores read from analysis/output2/scores_strict_1390.csv; every model
   evaluated on its own complete cases, as in Table 2)
C  Adjusted HR of the five panel proteins, derivation vs refit on validation
D  Calibration across risk quintiles + model comparison metrics

Outputs: output/fig4.pdf/.png; matrices/lasso_path_v03.csv, model_metrics.csv,
         panel_hr_derivation_validation.csv, calibration_90d.csv
"""
from pathlib import Path
import json
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
OUT = BASE / "figures" / "fig4"
POOL = BASE / "analysis" / "output2"
for sub in ["raw_data", "matrices", "code", "output"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20261106)
BLUE, GREEN, PURPLE, AMBER, RED = "#2E5E8C", "#2E7D4F", "#6A4C93", "#B07A2A", "#A03A3A"
GREY = "#777777"

# ======================= data =======================
clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
clin.columns = [str(c).strip() for c in clin.columns]
clin["center"] = clin["ID"].astype(str).str.split("-").str[0]
prot = pd.read_excel(PROT, sheet_name="1390已测蛋白质组")
prot.columns = [str(c).strip() for c in prot.columns]
prot = prot.set_index(prot.columns[0]).loc[clin["ID"]].astype(float)
DET = (prot == prot.min()).mean()          # detection frequency (annotation only)
NPROT = prot.shape[1]
log2 = np.log2(prot.clip(lower=1))
Z = (log2 - log2.mean()) / log2.std(ddof=1)

t1y = pd.to_numeric(clin["OS_1y_更新"], errors="coerce")
s90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
time90 = np.minimum(t1y.fillna(90), 90).to_numpy()
event90 = s90.isin([1, 2]).astype(int).to_numpy()
sex = clin["Gender"].replace(2, np.nan)
sex = sex.fillna(sex.mode()[0]).astype(float).to_numpy()
tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr = pd.to_numeric(clin["INR"], errors="coerce").fillna(pd.to_numeric(clin["INR"], errors="coerce").median())
cr = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce").fillna(pd.to_numeric(clin["CR(mg/dL)"], errors="coerce").median())
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.clip(lower=1))
        + 9.57 * np.log(cr.clip(lower=1)) + 6.43).clip(6, 40).to_numpy()
na = pd.to_numeric(clin["Na"], errors="coerce").fillna(pd.to_numeric(clin["Na"], errors="coerce").median()).clip(125, 137)
meldna = (meld + 1.32 * (137 - na.to_numpy()) - 0.033 * meld * (137 - na.to_numpy())).clip(6, 40)

# ---- 另四个评分不在此重算，直接取自 analysis/output2/scores_strict_1390.csv ----
# 该表由 analysis/score_engine.py 产出：每个评分按原始文献公式实现，并有 self_test() 用
# 文献算例反算通过。此处按 ID 对齐，避免同一评分出现两套实现。
SCORES = pd.read_csv(BASE / "analysis" / "output2" / "scores_strict_1390.csv")
SCORES = SCORES.set_index("id").reindex(clin["ID"].astype(str).to_numpy())
# Row alignment is the invariant; missing values are handled per model below (the raw clinical
# table leaves 35 MELD, 55 MELD 3.0 and 65 COSSH-II values incomplete, as in Table 2).
assert len(SCORES) == len(clin) and SCORES.index.notna().all(), "score table does not align with the clinical table"
meld3 = SCORES["MELD3"].to_numpy()
cossh2 = SCORES["COSSH_II"].to_numpy()
print("score join: MELD3 %.2f-%.2f | COSSH-II %.2f-%.2f"
      % (np.nanmin(meld3), np.nanmax(meld3), np.nanmin(cossh2), np.nanmax(cossh2)))


# ---- CLIF-C OF（Jalan 2014, J Hepatol; EF CLIF 官方定义）：六器官各 1–3 分，总分 6–18 ----
# 数据限制（见 Methods）：无升压药字段 → 循环最高 2 分；无 RRT/透析字段 → 肾 3 分仅按
# Cr≥3.5 判定。因此本实现系统性偏低最多 1 分——该偏差对 comparator 不利，须在正文声明。
def of_liver(v): return np.select([v < 6.0, v < 12.0], [1, 2], default=3)     # bilirubin, mg/dL
def of_kid(v): return np.select([v < 2.0, v < 3.5], [1, 2], default=3)        # creatinine, mg/dL
def of_brain(v): return np.select([v == 0, v <= 2], [1, 2], default=3)        # HE grade (West Haven)
def of_coag(v): return np.select([v < 2.0, v < 2.5], [1, 2], default=3)       # INR
def of_circ(m): return np.where(m >= 70, 1, 2)                                # MAP, mmHg
def of_resp(pf): return np.select([pf > 400, pf > 300], [1, 2], default=3)    # PaO2/FiO2


of_sum = (of_liver(pd.to_numeric(clin["TB(umol/L)"], errors="coerce").fillna(17.1).to_numpy() / 17.1)
          + of_kid(cr.to_numpy())
          + of_brain(pd.to_numeric(clin["病程有脑病及最大等级(0.no1.yes)_部分核实"],
                                   errors="coerce").fillna(0).to_numpy())
          + of_coag(inr.to_numpy())
          + of_circ(pd.to_numeric(clin["MAP"], errors="coerce").fillna(70).to_numpy())
          + of_resp(pd.to_numeric(clin["FIO_2"], errors="coerce").fillna(476.19).to_numpy()))

base = pd.DataFrame({"time": time90, "event": event90, "sex": sex,
                     "meld": meld, "meldna": meldna, "meld3": meld3,
                     "cossh2": cossh2, "of_sum": of_sum,
                     "center": clin["center"].to_numpy()})
for c in Z.columns:
    base[c] = Z[c].to_numpy()

DEV_C = ["nfa", "xna", "rja", "dta", "xha"]
dev_mask = base["center"].isin(DEV_C).to_numpy()
dev = base[dev_mask].reset_index(drop=True)
val = base[~dev_mask].reset_index(drop=True)
print("derivation n=%d events=%d | validation n=%d events=%d"
      % (len(dev), dev.event.sum(), len(val), val.event.sum()))


def fit(df, cols, lam=0.0, strata=True):
    frame = df[["time", "event"] + (["center"] if strata else []) + cols]
    kw = dict(penalizer=lam, l1_ratio=1.0) if lam > 0 else dict(penalizer=0.0)
    if strata:
        try:
            c = CoxPHFitter(**kw)
            c.fit(frame, "time", "event", strata=["center"])
            return c
        except Exception:
            pass
    c = CoxPHFitter(**kw)
    c.fit(frame[["time", "event"] + cols], "time", "event")
    return c


def baseline_h90(cph):
    b = cph.baseline_cumulative_hazard_
    rows = b.index[b.index <= 90]
    if not len(rows):
        return 0.0
    h = {s: float(b[s].loc[rows[-1]]) for s in b.columns}
    w = pd.Series(dev["center"]).value_counts(normalize=True)
    return float(sum(h.get(s, 0.0) * w.get(s, 0.0) for s in w.index))


# ======================= candidate pool and panel =======================
P5 = ["TNF.RI", "DR6", "Renin", "SREC.II", "GH"]
P3 = ["TNF.RI", "DR6", "Renin"]
scr = pd.read_csv(POOL / "screen_dev_primary883.csv")
pool = scr.loc[scr.FDR < 0.05, "protein"].tolist()
print("candidate pool FDR<0.05 in derivation:", len(pool))
assert all(p in base.columns for p in P5)

path_df = pd.read_csv(POOL / "lasso_path_primary883.csv")
res = json.load(open(POOL / "result_primary883.json", encoding="utf-8"))
LAM_1SE, LAM_CV = float(res["lam_1se"]), float(res["lam_cv"])
print("lambda_cv=%.4f  lambda_1se=%.4f" % (LAM_CV, LAM_1SE))

# ======================= models =======================
# MELD 3.0 and COSSH-ACLF II carry missing values (albumin / urea / neutrophil count), so every
# model is evaluated on its own complete cases in the validation set, exactly as in Table 2.
MODELS = {
    "MELD": (["meld"], "score"),
    "MELD-Na": (["meldna"], "score"),
    "MELD 3.0": (["meld3"], "score"),
    "COSSH-ACLF II": (["cossh2"], "score"),
    "CLIF-C OF": (["of_sum"], "score"),
    "MELD+3P": (["meld"] + P3, "p3"),
    "MELD+5P": (["meld"] + P5, "p5"),
    "3P only": (P3, "p3"),
    "5P only": (P5, "p5"),
}
rows, store = [], {}
for name, (cols, kind) in MODELS.items():
    dsub = dev.dropna(subset=cols).reset_index(drop=True)   # complete cases, as in Table 2
    m = fit(dsub, cols)
    ok = val[cols].notna().all(axis=1).to_numpy()      # complete cases in the validation set
    vsub = val[ok].reset_index(drop=True)
    ph_val = m.predict_partial_hazard(val.loc[ok, cols]).to_numpy()
    lp_full = np.full(len(val), np.nan)               # keep full length so later subsets align
    lp_full[ok] = np.log(ph_val)
    pr = 1.0 - np.exp(-baseline_h90(m) * ph_val)
    rows.append(dict(model=name, kind=kind, n_param=len(cols),
                     n_dev=int(len(dsub)), n_val=int(ok.sum()), ev_val=int(vsub.event.sum()),
                     c_derivation=concordance_index(dsub.time, -m.predict_partial_hazard(dsub[cols]).to_numpy(), dsub.event),
                     c_validation=concordance_index(vsub.time, -ph_val, vsub.event),
                     auc90=roc_auc_score(vsub.event, np.log(ph_val)),
                     brier90=float(np.mean((pr - vsub.event.to_numpy()) ** 2))))
    store[name] = dict(fit=m, cols=cols, ph=ph_val, lp=lp_full, risk=pr, ok=ok)

met = pd.DataFrame(rows)
# 1-year AUC (validation); O365: 0 alive, 1 death, 2 transplant, 4 lost -> drop 4
ev1y = pd.to_numeric(clin.loc[~dev_mask, "O365_更新"], errors="coerce").reset_index(drop=True)
case1y, ctrl1y = ev1y.isin([1, 2]).to_numpy(), (ev1y == 0).to_numpy()
keep0 = case1y | ctrl1y
met["auc1y"] = [roc_auc_score(case1y[k].astype(int), store[n]["lp"][k])
                for n in met.model
                for k in [keep0 & np.isfinite(store[n]["lp"])]]
meld_auc = met.loc[met.model == "MELD", "auc90"].iloc[0]
meld_c = met.loc[met.model == "MELD", "c_validation"].iloc[0]
met["dAUC_vs_MELD"] = met.auc90 - meld_auc
met["dC_vs_MELD"] = met.c_validation - meld_c

# bootstrap: 3P and 5P vs MELD, plus 5P vs 3P (paired on the same patients)
y = val.event.to_numpy()
ref = {k: store[k]["risk"] for k in ("MELD", "MELD+3P", "MELD+5P")}
boot = {k: [] for k in ("3vM", "5vM", "5v3")}
for _ in range(2000):
    s = RNG.choice(len(y), len(y), replace=True)
    if not (5 < y[s].sum() < len(s) - 5):
        continue
    a_m, a3, a5 = (roc_auc_score(y[s], ref[k][s]) for k in ("MELD", "MELD+3P", "MELD+5P"))
    boot["3vM"].append(a3 - a_m); boot["5vM"].append(a5 - a_m); boot["5v3"].append(a5 - a3)
ci = {}
for k, v in boot.items():
    v = np.array(v)
    ci[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)),
             2 * min((v <= 0).mean(), (v >= 0).mean()))
print(met.to_string(index=False))
print("dAUC 3P vs MELD %+.4f (%.4f, %.4f) P=%.3f"
      % (met.loc[met.model == "MELD+3P", "dAUC_vs_MELD"].iloc[0], *ci["3vM"]))
print("dAUC 5P vs MELD %+.4f (%.4f, %.4f) P=%.3f"
      % (met.loc[met.model == "MELD+5P", "dAUC_vs_MELD"].iloc[0], *ci["5vM"]))
print("dAUC 5P vs 3P   %+.4f (%.4f, %.4f) P=%.3f"
      % (met.loc[met.model == "MELD+5P", "auc90"].iloc[0] - met.loc[met.model == "MELD+3P", "auc90"].iloc[0], *ci["5v3"]))

# The numbers printed on the figure are taken from the primary comparison table
# (analysis/output2/table2_v2.csv) and from panel_5v3.json rather than from the loop above,
# so that the figure and Table 2 cannot disagree; the loop's own bootstrap stays as a check.
T2 = pd.read_csv(POOL / "table2_v2.csv").set_index("row")
P53 = json.load(open(POOL / "panel_5v3.json", encoding="utf-8"))


def t2v(row, col):
    return float(T2.loc[row, col])


for row, key in (("MELD + 3-protein panel", "3P"), ("MELD + 5-protein panel", "5P")):
    print("%s vs MELD (table2): %+.3f (%+.3f, %+.3f) P=%.3f"
          % (key, t2v(row, "dAUC"), t2v(row, "dAUC_lo"), t2v(row, "dAUC_hi"), t2v(row, "dAUC_p")))
print("5P vs 3P (panel_5v3): %+.4f (%+.4f, %+.4f) P=%.3f"
      % (P53["dAUC"]["delta"], P53["dAUC"]["lo"], P53["dAUC"]["hi"], P53["dAUC"]["p"]))


# NRI / IDI (MELD+5P vs MELD, and 5P vs 3P) on 90-day predicted risk
def nri_idi(old, new, yy):
    up, dn = new > old, new < old
    c, k = yy == 1, yy == 0
    return ((up[c].mean() - dn[c].mean()) + (dn[k].mean() - up[k].mean()),
            (new[c].mean() - old[c].mean()) - (new[k].mean() - old[k].mean()))


nri5, idi5 = nri_idi(ref["MELD"], ref["MELD+5P"], y)
nri53, idi53 = nri_idi(ref["MELD+3P"], ref["MELD+5P"], y)
print("NRI/IDI MELD+5P vs MELD: %+.3f / %+.3f" % (nri5, idi5))
print("NRI/IDI MELD+5P vs MELD+3P: %+.3f / %+.4f" % (nri53, idi53))

# ======================= panel HR: derivation vs validation =======================
m_dev = fit(dev, ["meld"] + P5)
m_val = fit(val, ["meld"] + P5, strata=True)
hr_rows = []
for v in ["meld"] + P5:
    s = m_dev.summary.loc[v]
    sv = m_val.summary.loc[v]
    hr_rows.append(dict(term="MELD" if v == "meld" else v,
                        hr_dev=float(s["exp(coef)"]), lo_dev=float(np.exp(s["coef"] - 1.96 * s["se(coef)"])),
                        hi_dev=float(np.exp(s["coef"] + 1.96 * s["se(coef)"])), p_dev=float(s["p"]),
                        hr_val=float(sv["exp(coef)"]), lo_val=float(np.exp(sv["coef"] - 1.96 * sv["se(coef)"])),
                        hi_val=float(np.exp(sv["coef"] + 1.96 * sv["se(coef)"])), p_val=float(sv["p"]),
                        floor_rate=float(DET.get(v, np.nan))))
hr = pd.DataFrame(hr_rows)
print(hr.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# ======================= calibration =======================
pr5 = ref["MELD+5P"]
q = pd.qcut(pr5, 5, labels=False, duplicates="drop")
cal = pd.DataFrame({"q": q, "pred": pr5, "obs": y}).groupby("q").agg(
    n=("obs", "size"), pred=("pred", "mean"), obs=("obs", "sum"))
cal["obs_rate"] = cal.obs / cal.n
print(cal.to_string())

met.to_csv(OUT / "matrices" / "model_metrics.csv", index=False, encoding="utf-8-sig")
hr.to_csv(OUT / "matrices" / "panel_hr_derivation_validation.csv", index=False, encoding="utf-8-sig")
cal.to_csv(OUT / "matrices" / "calibration_90d.csv", encoding="utf-8-sig")
path_df.to_csv(OUT / "matrices" / "lasso_path_v03.csv", index=False, encoding="utf-8-sig")

# ---------------- figure ----------------
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7})
fig = plt.figure(figsize=(9.4, 6.9))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.02], hspace=0.62, wspace=1.30)
axA = fig.add_subplot(gs[0, 0:3])
axB = fig.add_subplot(gs[0, 3:6])
axC = fig.add_subplot(gs[1, 0:3])
axD = fig.add_subplot(gs[1, 3:6])

# --- A: LASSO path ---
lam_all = path_df["lam"].to_numpy()
show = lam_all <= 0.0756   # the 0.0866 grid point did not converge (nvar jumps 6->16, C drops 0.82->0.78) and is omitted
lam = lam_all[show]
for p in pool:
    if p in path_df.columns:
        axA.plot(lam, path_df.loc[show, p], color="#C9CFD6", lw=0.7, zorder=1)
hi_cols = {"TNF.RI": BLUE, "DR6": PURPLE, "Renin": GREEN, "SREC.II": AMBER, "GH": RED}
for p, c in hi_cols.items():
    if p in path_df.columns:
        axA.plot(lam, path_df.loc[show, p], color=c, lw=1.5, zorder=3,
                 label=p.replace(".", "-"))
axA.axvline(LAM_1SE, color="#111111", ls="--", lw=0.9, zorder=4)
axA.axhline(0, color="#999999", lw=0.6)
axA.set_xscale("log")
axA.set_xlabel("LASSO penalty $\\lambda$ (log scale)", fontsize=7)
axA.set_ylabel("Standardised coefficient", fontsize=7)
axA.set_title("LASSO-Cox paths, %d-protein pool" % len(pool), fontsize=8, loc="left")
axA.text(LAM_1SE * 1.07, axA.get_ylim()[1] * 0.95, "$\\lambda_{1\\mathrm{SE}}$",
         fontsize=6.8, color="#111111", ha="left", va="top")
axA.legend(fontsize=5.8, frameon=False, loc="upper right", ncol=2, handlelength=1.4)
axA.tick_params(labelsize=6)
axA.text(0.02, -0.30, "Penalty range shown 0.010-0.076 (the 0.087 grid point did not converge and is omitted).\nGrey: the remaining %d candidates; five proteins are non-zero at $\\lambda_{1\\mathrm{SE}}$."
         % (len(pool) - 5), transform=axA.transAxes, fontsize=5.9, color="#333333")
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
axA.text(-0.14, 1.06, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")

# --- B: AUC90 across models ---
order = ["MELD", "MELD-Na", "MELD 3.0", "COSSH-ACLF II", "CLIF-C OF",
         "3P only", "5P only", "MELD+3P", "MELD+5P"]
cols_b = []
for m in order:
    if m == "MELD+5P":
        cols_b.append(BLUE)
    elif m == "MELD+3P":
        cols_b.append("#7FA6C8")
    elif m in ("3P only", "5P only"):
        cols_b.append("#B9C6D2")
    elif m in ("MELD 3.0", "COSSH-ACLF II"):
        cols_b.append("#CDD3DA")      # current scores, shaded one step darker than MELD
    else:
        cols_b.append("#DCDCDC")
vals = [met.loc[met.model == m, "auc90"].iloc[0] for m in order]
xb = np.arange(len(order))
axB.bar(xb, vals, color=cols_b, edgecolor="#555555", linewidth=0.5, width=0.70)
for i, v in enumerate(vals):
    axB.text(i, v + 0.0035, f"{v:.3f}", ha="center", fontsize=5.2)
axB.axhline(meld_auc, color=GREY, ls=":", lw=0.8)
axB.set_xticks(xb, ["MELD", "MELD-\nNa", "MELD\n3.0", "COSSH-\nII", "CLIF-C\nOF",
                    "3P\nonly", "5P\nonly", "MELD\n+3P", "MELD\n+5P"], fontsize=5.0)
axB.set_ylim(0.80, 0.988)
axB.set_ylabel("AUC at 90 days (validation)", fontsize=7)
axB.set_title("Discrimination in validation (8 held-out centres)", fontsize=8, loc="left")
i3, i5 = order.index("MELD+3P"), order.index("MELD+5P")
axB.annotate("", xy=(i5 - 0.34, 0.957), xytext=(i3 + 0.34, 0.957),
             arrowprops=dict(arrowstyle="<->", lw=0.7, color="#333333"))
_p53 = P53["dAUC"]["p"]
_p53 = int(_p53 * 100 + 0.5) / 100.0        # round-half-up so the figure matches the text (0.215 -> 0.22)
axB.text((i3 + i5) / 2, 0.962, "$\\Delta$ %+.3f (P=%.2f)" % (P53["dAUC"]["delta"], _p53),
         ha="center", va="bottom", fontsize=5.6, color="#333333")
axB.text(0.015, 0.985, "vs MELD:  +%.3f (P=%.3f) with 3P\n                +%.3f (P=%.3f) with 5P" %
         (t2v("MELD + 3-protein panel", "dAUC"), t2v("MELD + 3-protein panel", "dAUC_p"),
          t2v("MELD + 5-protein panel", "dAUC"), t2v("MELD + 5-protein panel", "dAUC_p")),
         transform=axB.transAxes, fontsize=5.5, color="#333333", va="top", ha="left")
axB.tick_params(labelsize=6)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.text(-0.14, 1.06, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")

# --- C: HR derivation vs validation ---
terms = ["MELD"] + P5
ypos = np.arange(len(terms))[::-1]
for i, t in zip(ypos, terms):
    r = hr.loc[hr.term == t].iloc[0]
    axC.plot([r.lo_dev, r.hi_dev], [i + 0.17, i + 0.17], color=BLUE, lw=1.1, solid_capstyle="butt")
    axC.plot([r.hr_dev], [i + 0.17], "o", ms=4.0, color=BLUE,
             label="Derivation (n=995, 116 events)" if i == ypos[0] else None)
    axC.plot([r.lo_val, r.hi_val], [i - 0.17, i - 0.17], color=AMBER, lw=1.1, solid_capstyle="butt")
    axC.plot([r.hr_val], [i - 0.17], "s", ms=3.6, color=AMBER, mfc="white", mew=1.0,
             label="Validation, refit (n=395, 46 events)" if i == ypos[0] else None)
    star = "*" if r.p_val < 0.05 else ""
    axC.text(r.hi_val * 1.06, i - 0.17, f"{r.hr_val:.2f}{star}", fontsize=5.6,
             va="center", color=AMBER if r.p_val < 0.05 else "#8A8A8A")
axC.axvline(1, color="#999999", ls="--", lw=0.7)
axC.set_yticks(ypos, [t.replace(".", "-") for t in terms], fontsize=6.5)
axC.set_xscale("log")
axC.set_xticks([0.5, 0.7, 1, 1.5, 2, 3])
axC.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
axC.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axC.set_xlim(0.45, 3.4)
axC.set_xlabel("HR per SD, adjusted for MELD", fontsize=7)
axC.set_title("Panel proteins: derivation vs validation", fontsize=8, loc="left")
axC.legend(fontsize=5.4, frameon=False, loc="lower right")
axC.tick_params(labelsize=6)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)
axC.text(-0.30, 1.06, "C", transform=axC.transAxes, fontsize=12, fontweight="bold")

# --- D: calibration + metrics ---
axD.plot([0, 0.55], [0, 0.55], color="#999999", ls="--", lw=0.7)
axD.plot(cal.pred, cal.obs_rate, "o-", color=BLUE, ms=4.2, lw=1.0)
for _, r in cal.iterrows():
    axD.annotate(f"{int(r.obs)}/{int(r.n)}", (r.pred, r.obs_rate), textcoords="offset points",
                 xytext=(5, -7), fontsize=5.4, color="#555555")
axD.set_xlim(0, 0.55); axD.set_ylim(0, 0.55)
axD.set_xlabel("Predicted 90-day event probability", fontsize=7)
axD.set_ylabel("Observed 90-day event rate", fontsize=7)
axD.set_title("Calibration of MELD+5P (validation)", fontsize=8, loc="left")
axD.tick_params(labelsize=6)
for s in ("top", "right"):
    axD.spines[s].set_visible(False)
r5 = met.loc[met.model == "MELD+5P"].iloc[0]
r3 = met.loc[met.model == "MELD+3P"].iloc[0]
txt = ("C-index  %.3f  (MELD %.3f)\n"
       "AUC 90 d  %.3f   |   1 y  %.3f\n"
       "$\\Delta$AUC vs MELD   %+.3f  (%+.3f to %+.3f)\n"
       "$\\Delta$AUC 5P vs 3P  %+.3f  (%+.3f to %+.3f)\n"
       "NRI  %+.3f   |   IDI  %+.3f\n"
       "Brier  %.3f") % (
    r5.c_validation, meld_c, r5.auc90, r5.auc1y,
    t2v("MELD + 5-protein panel", "dAUC"), t2v("MELD + 5-protein panel", "dAUC_lo"),
    t2v("MELD + 5-protein panel", "dAUC_hi"),
    P53["dAUC"]["delta"], P53["dAUC"]["lo"], P53["dAUC"]["hi"], nri5, idi5, r5.brier90)
axD.text(0.03, 0.97, txt, transform=axD.transAxes, fontsize=5.7, va="top", ha="left",
         color="#222222", linespacing=1.65)
axD.text(-0.30, 1.06, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")

fig.savefig(OUT / "output" / "fig4.pdf", bbox_inches="tight")
fig.savefig(OUT / "output" / "fig4.png", dpi=300, bbox_inches="tight")
print("saved fig4")
