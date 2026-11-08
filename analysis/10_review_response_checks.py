# -*- coding: utf-8 -*-
"""Checks requested by the simulated peer review of manuscript v0.3.

Every item below is answered with computation on the same analytic dataset
that produced the manuscript, so the response letter can quote numbers rather
than intentions.

Outputs -> analysis/output4/
  1 ph_test.csv                 Schoenfeld residual test of the final model
  2 calibration_validation.csv  calibration slope / intercept / CITL / deciles
  3 loco_clean.csv              leave-one-centre-out, flagged by whether the
                                held-out centre contributed to selection
  4 bootstrap_se.csv            SE of each C-index vs SE of the paired difference
  5 aclf_interaction.csv        formal panel x ACLF interaction test
  6 crp_missing.csv             CRP availability and complete-case sample
  7 sex_audit.csv               raw sex-code audit
  8 clif_of_truncation.csv      ceiling effects in the CLIF-C OF components
  9 stability_table_S6.csv      full stability frequencies, both protocols
 10 split_clean.csv             200-split analysis flagged for selection overlap

Run: <managed python> analysis/10_review_response_checks.py
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "analysis" / "output4"
OUT.mkdir(parents=True, exist_ok=True)

CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\proteinarray_matrix_1390人蛋白质组结果（质控后883蛋白）.xlsx")

BIG5 = ["nfa", "xna", "rja", "dta", "xha"]          # derivation (5dev scheme)
SMALL8 = ["jda", "tha", "sda", "zxa", "hna", "xya", "xyb", "zya"]  # validation
PANEL = ["TNF-RI", "DR6", "Renin", "SREC-II", "GH"]
PANEL_RAW = {"TNF-RI": "TNF.RI", "DR6": "DR6", "Renin": "Renin",
             "SREC-II": "SREC.II", "GH": "GH"}
RNG = np.random.default_rng(20261107)

# --------------------------------------------------------------------------- #
# data — identical preparation to reselect_panel.py (5dev scheme)
# --------------------------------------------------------------------------- #
clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
clin.columns = [str(c).strip() for c in clin.columns]
clin["center"] = clin["ID"].astype(str).str.split("-").str[0]

prot = pd.read_excel(PROT, sheet_name="1390已测蛋白质组")
prot.columns = [str(c).strip() for c in prot.columns]
prot = prot.set_index(prot.columns[0]).loc[clin["ID"]].astype(float)

log2 = np.log2(prot.clip(lower=1))
Z = (log2 - log2.mean()) / log2.std(ddof=1)
Z = Z.loc[:, Z.std(ddof=1) > 0]

t1y = pd.to_numeric(clin["OS_1y_更新"], errors="coerce")
s90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
time90 = np.minimum(t1y.fillna(90), 90).to_numpy()
event90 = s90.isin([1, 2]).astype(int).to_numpy()

raw_sex = pd.to_numeric(clin["Gender"], errors="coerce")
sex = raw_sex.replace(2, np.nan)
sex = sex.fillna(sex.mode()[0]).astype(float).to_numpy()

tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr = pd.to_numeric(clin["INR"], errors="coerce")
cr = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce")
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.fillna(inr.median()).clip(lower=1))
        + 9.57 * np.log(cr.fillna(cr.median()).clip(lower=1)) + 6.43).clip(6, 40).to_numpy()
center = clin["center"].to_numpy()

base = pd.DataFrame({"time": time90, "event": event90, "sex": sex, "meld": meld,
                     "center": center})
for g in PANEL:
    raw = PANEL_RAW[g]
    assert raw in Z.columns, (raw, "missing from protein matrix")
    base[g] = Z[raw].to_numpy()

dev = base["center"].isin(BIG5).to_numpy()
tr, va = base[dev].copy(), base[~dev].copy()
print(f"n={len(base)} events={event90.sum()} | dev {len(tr)} ({tr.event.sum()} ev) | "
      f"val {len(va)} ({va.event.sum()} ev)")

COLS = ["meld", "sex"] + PANEL


def fit(df, cols=COLS, strata=True):
    frame = df[["time", "event"] + (["center"] if strata else []) + cols]
    if strata:
        try:
            c = CoxPHFitter(penalizer=0.0)
            c.fit(frame, "time", "event", strata=["center"])
            return c
        except Exception:
            pass
    c = CoxPHFitter(penalizer=0.0)
    c.fit(frame[["time", "event"] + cols], "time", "event")
    return c


def h0_90(cph, train_center):
    b = cph.baseline_cumulative_hazard_
    rows = b.index[b.index <= 90]
    h = {s: float(b[s].loc[rows[-1]]) for s in b.columns}
    w = pd.Series(train_center).value_counts(normalize=True)
    return float(sum(h.get(s, 0.0) * w.get(s, 0.0) for s in w.index))


def risk90(cph, df, cols, h0):
    return 1.0 - np.exp(-h0 * cph.predict_partial_hazard(df[cols]).to_numpy())


model = fit(tr)
h0 = h0_90(model, tr["center"])

# --------------------------------------------------------------------------- #
# 1. proportional hazards (Schoenfeld residuals)
# --------------------------------------------------------------------------- #
rows = []
for label, dfit in [("derivation (5 centres, n=995, 116 events)", tr),
                    ("validation (8 centres, refit, n=395, 46 events)", va)]:
    m = fit(dfit)
    try:
        r = proportional_hazard_test(m, dfit, time_transform="rank")
        sm = r.summary
        for var in sm.index:
            rows.append(dict(dataset=label, variable=var,
                             statistic=float(sm.loc[var, "test_statistic"]),
                             p=float(sm.loc[var, "p"])))
    except Exception as exc:
        rows.append(dict(dataset=label, variable="__error__", statistic=np.nan, p=np.nan,
                         note=str(exc)[:140]))
ph = pd.DataFrame(rows)
if "p" in ph:
    ok = ph["p"].notna()
    ph.loc[ok, "p_holm"] = np.minimum(1, ph.loc[ok, "p"] * ok.sum())
ph.to_csv(OUT / "ph_test.csv", index=False, encoding="utf-8-sig")
print("\n[1] Schoenfeld\n", ph.to_string(index=False))
min_p = float(ph["p"].min()) if ph["p"].notna().any() else np.nan
n_sig = int((ph["p"] < 0.05).sum()) if ph["p"].notna().any() else 0

# --------------------------------------------------------------------------- #
# 2. calibration in the validation set
# --------------------------------------------------------------------------- #
p = risk90(model, va, COLS, h0)
y = va["event"].to_numpy()
p = np.clip(p, 1e-8, 1 - 1e-8)
lp = np.log(p / (1 - p))


def nll(b):
    eta = b[0] + b[1] * lp
    return -np.sum(y * eta - np.log1p(np.exp(eta)))


res = optimize.minimize(nll, x0=[0.0, 1.0], method="Nelder-Mead",
                        options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000))
slope, intercept = float(res.x[1]), float(res.x[0])
# standard errors from the numerical Hessian
h = np.zeros((2, 2))
eps = 1e-4
for i in range(2):
    for j in range(2):
        e = np.zeros(2)
        e[i] = eps
        f = np.zeros(2)
        f[j] = eps
        h[i, j] = (nll(res.x + e + f) - nll(res.x + e - f) - nll(res.x - e + f) + nll(res.x - e - f)) / (4 * eps * eps)
se = np.sqrt(np.diag(np.linalg.inv(h)))
citl = float(np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean())))

q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
dec = pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q").agg(
    n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
dec.to_csv(OUT / "calibration_validation.csv", index=False, encoding="utf-8-sig")
cal = dict(calibration_slope=slope, calibration_slope_se=float(se[1]),
           calibration_intercept=intercept, calibration_intercept_se=float(se[0]),
           citl=citl, mean_predicted=float(p.mean()), observed_rate=float(y.mean()),
           brier=float(np.mean((p - y) ** 2)))
pd.Series(cal).to_csv(OUT / "calibration_summary.csv", encoding="utf-8-sig", header=["value"])
print("\n[2] calibration:", json.dumps(cal, indent=1))
print(dec.to_string())

# --------------------------------------------------------------------------- #
# 3. leave-one-centre-out, flagged by selection overlap
# --------------------------------------------------------------------------- #
rows = []
for c in sorted(set(center)):
    mask = base["center"] == c
    if base.loc[mask, "event"].sum() < 5:
        continue
    fit_df, test_df = base[~mask], base[mask]
    m = fit(fit_df)
    h = h0_90(m, fit_df["center"])
    c_panel = concordance_index(test_df["time"], -m.predict_partial_hazard(test_df[COLS]),
                                test_df["event"])
    mp = fit(fit_df, cols=["meld", "sex"])
    c_meld = concordance_index(test_df["time"], -mp.predict_partial_hazard(test_df[["meld", "sex"]]),
                               test_df["event"])
    rows.append(dict(centre=c, n=int(mask.sum()), events=int(base.loc[mask, "event"].sum()),
                     selection_role=("derivation — held-out centre contributed to panel selection"
                                     if c in BIG5 else "validation — held-out centre never used for selection"),
                     clean=(c not in BIG5),
                     c_meld=c_meld, c_panel=c_panel, d_c=c_panel - c_meld))
loco = pd.DataFrame(rows).sort_values(["clean", "centre"])
loco.to_csv(OUT / "loco_clean.csv", index=False, encoding="utf-8-sig")
print("\n[3] LOCO\n", loco.to_string(index=False))

# --------------------------------------------------------------------------- #
# 4. paired bootstrap: SE of each C-index vs SE of the difference
# --------------------------------------------------------------------------- #
dm = model.predict_partial_hazard(va[COLS]).to_numpy()
dp = None
mp = fit(tr, cols=["meld", "sex"])
meld_lp = mp.predict_partial_hazard(va[["meld", "sex"]]).to_numpy()
c_m = concordance_index(va["time"], -meld_lp, va["event"])
c_p = concordance_index(va["time"], -dm, va["event"])
bs = []
idx = np.arange(len(va))
for _ in range(2000):
    b = RNG.choice(idx, len(va), replace=True)
    if va["event"].to_numpy()[b].sum() < 5:
        continue
    bs.append((concordance_index(va["time"].to_numpy()[b], -meld_lp[b], va["event"].to_numpy()[b]),
               concordance_index(va["time"].to_numpy()[b], -dm[b], va["event"].to_numpy()[b])))
bs = np.array(bs)
boot = dict(c_meld=c_m, c_panel=c_p, d_c=c_p - c_m,
            se_c_meld=float(bs[:, 0].std(ddof=1)), se_c_panel=float(bs[:, 1].std(ddof=1)),
            se_d_c=float((bs[:, 1] - bs[:, 0]).std(ddof=1)),
            corr=float(np.corrcoef(bs[:, 0], bs[:, 1])[0, 1]),
            n_boot=len(bs))
pd.Series(boot).to_csv(OUT / "bootstrap_se.csv", encoding="utf-8-sig", header=["value"])
print("\n[4] bootstrap:", json.dumps({k: round(v, 4) for k, v in boot.items()}, indent=1))

# --------------------------------------------------------------------------- #
# 5. formal panel x ACLF interaction (supports or refutes "before ACLF")
# --------------------------------------------------------------------------- #
aclf = pd.to_numeric(clin["COSSH-ACLF"], errors="coerce") if "COSSH-ACLF" in clin.columns else None
if aclf is None:
    cand = [c for c in clin.columns if "ACLF" in c.upper() and "COSSH" in c.upper()]
    aclf = pd.to_numeric(clin[cand[0]], errors="coerce") if cand else None
if aclf is not None:
    b2 = base.copy()
    b2["aclf"] = (pd.to_numeric(aclf, errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    b2["pscore"] = (base[PANEL].to_numpy() * model.params_[PANEL].to_numpy()).sum(1)
    b2["inter"] = b2["pscore"] * b2["aclf"]
    f0 = fit(b2, cols=["meld", "sex", "pscore", "aclf"])
    f1 = fit(b2, cols=["meld", "sex", "pscore", "aclf", "inter"])
    lr = 2 * (f1.log_likelihood_ - f0.log_likelihood_)
    pint = float(stats.chi2.sf(max(lr, 0), 1))
    out = dict(n_aclf=int(b2["aclf"].sum()), lr=float(lr), p_interaction=pint,
               hr_interaction=float(np.exp(f1.params_["inter"])),
               lo=float(np.exp(f1.confidence_intervals_.loc["inter"].iloc[0])),
               hi=float(np.exp(f1.confidence_intervals_.loc["inter"].iloc[1])))
    pd.Series(out).to_csv(OUT / "aclf_interaction.csv", encoding="utf-8-sig", header=["value"])
    print("\n[5] ACLF interaction:", json.dumps({k: round(v, 4) for k, v in out.items()}, indent=1))

# --------------------------------------------------------------------------- #
# 6. CRP availability
# --------------------------------------------------------------------------- #
if "CRP" in prot.columns:
    crp = pd.to_numeric(prot["CRP"], errors="coerce")
    f = (prot["CRP"] == prot["CRP"].min())
    rows = [dict(group="all", n=len(crp), at_floor=int(f.sum()),
                 pct_floor=100 * float(f.mean())),
            dict(group="derivation", n=int(dev.sum()), at_floor=int(f[dev].sum()),
                 pct_floor=100 * float(f[dev].mean())),
            dict(group="validation", n=int((~dev).sum()), at_floor=int(f[~dev].sum()),
                 pct_floor=100 * float(f[~dev].mean()))]
    pd.DataFrame(rows).to_csv(OUT / "crp_missing.csv", index=False, encoding="utf-8-sig")
    print("\n[6] CRP\n", pd.DataFrame(rows).to_string(index=False))

# --------------------------------------------------------------------------- #
# 7. sex audit
# --------------------------------------------------------------------------- #
aud = raw_sex.value_counts(dropna=False).rename_axis("raw_code").reset_index(name="n")
aud.to_csv(OUT / "sex_audit.csv", index=False, encoding="utf-8-sig")
print("\n[7] sex\n", aud.to_string(index=False))

# --------------------------------------------------------------------------- #
# 8. CLIF-C OF component ceilings (from the derived score file)
# --------------------------------------------------------------------------- #
sc = pd.read_csv(BASE / "analysis" / "output2" / "scores_by_patient.csv")
rows = []
for comp, col in [("liver", "OF_liver"), ("kidney", "OF_kidney"), ("brain", "OF_brain"),
                  ("coagulation", "OF_coag"), ("circulation", "OF_circ"), ("respiration", "OF_resp")]:
    v = pd.to_numeric(sc[col], errors="coerce")
    rows.append(dict(component=comp, n=int(v.notna().sum()), min=float(v.min()),
                     median=float(v.median()), max=float(v.max()),
                     pct_at_min=100 * float((v == v.min()).mean()),
                     pct_at_max=100 * float((v == v.max()).mean())))
pd.DataFrame(rows).to_csv(OUT / "clif_of_truncation.csv", index=False, encoding="utf-8-sig")
print("\n[8] CLIF-C OF\n", pd.DataFrame(rows).to_string(index=False))

# --------------------------------------------------------------------------- #
# 9. stability frequencies, both protocols
# --------------------------------------------------------------------------- #
rows = []
for tag, f in [("primary (883, no filter)", "stability_primary883.csv"),
               (">=80% quantified (665)", "stability_filter665.csv")]:
    d = pd.read_csv(BASE / "analysis" / "output2" / f)
    d = d.sort_values("freq", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    for _, r in d.head(15).iterrows():
        rows.append(dict(protocol=tag, rank=int(r["rank"]), protein=r["protein"],
                         freq=float(r["freq"]), n_boot=int(r["n_boot"]),
                         panel_member=r["protein"] in PANEL or r["protein"].replace(".", "-") in PANEL))
    for name in ["SREC.II", "GH"]:
        sub = d[d["protein"] == name]
        if len(sub):
            r = sub.iloc[0]
            rows.append(dict(protocol=tag, rank=int(r["rank"]), protein=r["protein"],
                             freq=float(r["freq"]), n_boot=int(r["n_boot"]), panel_member=True))
pd.DataFrame(rows).drop_duplicates(["protocol", "protein"]).to_csv(
    OUT / "stability_table_S6.csv", index=False, encoding="utf-8-sig")
print("\n[9] stability table written")

# --------------------------------------------------------------------------- #
# 10. 200-split analysis: how many splits put a selection centre in validation?
# --------------------------------------------------------------------------- #
sp = pd.read_csv(BASE / "analysis" / "output2" / "repeated_centre_split.csv")
sp["n_selection_centres_in_validation"] = sp["dev_centres"].apply(
    lambda s: len(set(BIG5) - set(str(s).split("|"))))
sp["clean"] = sp["n_selection_centres_in_validation"] == 0
sp.to_csv(OUT / "split_clean.csv", index=False, encoding="utf-8-sig")
summ = dict(n_splits=len(sp), n_clean=int(sp["clean"].sum()),
            median_d_auc_all=float(sp["d_auc"].median()),
            median_d_auc_clean=float(sp.loc[sp["clean"], "d_auc"].median()) if sp["clean"].any() else np.nan,
            frac_positive=float((sp["d_auc"] > 0).mean()),
            median_n_overlap=int(sp["n_selection_centres_in_validation"].median()))
pd.Series(summ).to_csv(OUT / "split_clean_summary.csv", encoding="utf-8-sig", header=["value"])
print("\n[10] splits:", json.dumps(summ, indent=1))

json.dump(dict(ph_min_p=min_p, ph_n_violations=n_sig,
               calibration=cal, bootstrap=boot, splits=summ),
          open(OUT / "review_checks_summary.json", "w"), indent=1)
print("\ndone ->", OUT)
