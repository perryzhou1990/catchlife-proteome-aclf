# -*- coding: utf-8 -*-
"""Supplementary robustness analyses (round 2) for manuscript v0.3.

Protocol v1.2: ALL 883 quantified proteins enter screening (no detection
filter); locked panel = union of the two pre-specified primary lockings
(primary883: TNF-RI/DR6/Renin; filter665: TNF-RI/DR6/Renin/SREC-II/GH)
=> FIVE proteins. The v0.2 4-protein results are archived in output_v02_legacy_4panel.

Motivated by the two metabolomics papers of the same CATCH-LIFE cohort
(literature/00_两篇代谢组学文献深度解读.md), this script adds four analyses
that reviewers of this field routinely ask for and that the v0.2 manuscript
did not yet contain:

  A  Locked panel score stratified by MELD tertile (does the panel work
     where MELD is uninformative? interaction test on the continuous scale)
  B  Five-protein panel versus five proteins + CRP (does an inexpensive
     acute-phase protein replace the panel?)
  C  Win ratio for the 90-day composite endpoint, stratified by MELD decile
     (Pocock hierarchical-composite framing)
  D  Decision-curve analysis in the external validation set
  E  Selection funnel 883 -> 128 -> 91 -> 6 -> 3/5 -> 5 (for Fig S2A)

Design mirrors figures/fig4/code/fig4_make.py exactly (same cleaning, same
centre split, same endpoint coding) so that every number here is directly
comparable with the manuscript.

Outputs (analysis/output/):
  meld_tertile.csv, meld_tertile_note.json
  crp_model.csv, crp_reclass.csv
  win_ratio.csv
  dca.csv
  funnel.csv
  robustness2_summary.json
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)

CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\proteinarray_matrix_1390人蛋白质组结果（质控后883蛋白）.xlsx")
RNG = np.random.default_rng(20261105)

# --------------------------------------------------------------------------- #
# 0. data (identical to fig4_make.py)
# --------------------------------------------------------------------------- #
clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
clin.columns = [str(c).strip() for c in clin.columns]
clin["center"] = clin["ID"].astype(str).str.split("-").str[0]

prot = pd.read_excel(PROT, sheet_name="1390已测蛋白质组")
prot.columns = [str(c).strip() for c in prot.columns]
prot = prot.set_index(prot.columns[0]).loc[clin["ID"]].astype(float)
# Protocol v1.2: no detection filter - all 883 quantified proteins enter.
ff = (prot == prot.min()).mean()
log2 = np.log2(prot.clip(lower=1))
Z = (log2 - log2.mean()) / log2.std(ddof=1)
Zarr = Z.to_numpy()
zcol = {c: i for i, c in enumerate(Z.columns)}

t1y = pd.to_numeric(clin["OS_1y_更新"], errors="coerce")
s90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
time90 = np.minimum(t1y.fillna(90), 90).to_numpy()
event90 = s90.isin([1, 2]).astype(int).to_numpy()

sex = clin["Gender"].replace(2, np.nan)
sex = sex.fillna(sex.mode()[0]).astype(float).to_numpy()
tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr_raw = pd.to_numeric(clin["INR"], errors="coerce")
cr_raw = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce")
inr = inr_raw.fillna(inr_raw.median())
cr = cr_raw.fillna(cr_raw.median())
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.clip(lower=1))
        + 9.57 * np.log(cr.clip(lower=1)) + 6.43).clip(6, 40).to_numpy()
center = clin["center"].to_numpy()
dev_mask = np.isin(center, ["nfa", "xna", "rja", "dta", "xha"])

PANEL = ["TNF.RI", "Renin", "DR6", "SREC.II", "GH"]
assert all(p in Z.columns for p in PANEL), [p for p in PANEL if p not in Z.columns]
print("cohort n=%d  events90=%d  dev n=%d (%d ev)  val n=%d (%d ev)"
      % (len(clin), event90.sum(), dev_mask.sum(), event90[dev_mask].sum(),
         (~dev_mask).sum(), event90[~dev_mask].sum()))

base = pd.DataFrame({"time": time90, "event": event90, "sex": sex, "meld": meld,
                     "center": center})
for p in PANEL + ["CRP"]:
    base[p] = Zarr[:, zcol[p]]

dev = base[dev_mask].copy()
val = base[~dev_mask].copy()


def fit(df, cols, strata=True):
    c = CoxPHFitter()
    frame = df[["time", "event"] + (["center"] if strata else []) + cols]
    c.fit(frame, "time", "event", strata=["center"] if strata else None)
    return c


def baseline_h90(cph, train_center):
    """Marginal 90-day baseline cumulative hazard of a centre-stratified Cox.

    predict_survival_function() cannot be used here because the validation
    centres are by construction absent from the training strata.  We therefore
    average the stratum-specific baseline cumulative hazard at t=90 over the
    derivation centres, weighted by their size, and apply the patient-specific
    partial hazard to it -- a standard marginal prediction from a stratified
    Cox model.  Only the baseline level is marginalised; within-centre ranking
    (and hence AUC and the win ratio) is untouched.
    """
    b = cph.baseline_cumulative_hazard_
    rows = b.index[b.index <= 90]
    if not len(rows):
        return 0.0
    h = {s: float(b[s].loc[rows[-1]]) for s in b.columns}
    w = pd.Series(train_center).value_counts(normalize=True)
    return float(sum(h.get(s, 0.0) * w.get(s, 0.0) for s in w.index))


def surv90(cph, df, cols, h0):
    ph = cph.predict_partial_hazard(df[cols]).to_numpy()
    return 1.0 - np.exp(-h0 * ph)


def boot_delta_auc(risk_a, risk_b, event, n=1000):
    """Bootstrap CI for AUC(b) - AUC(a); a = reference model."""
    d = []
    idx = np.arange(len(event))
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        if event[s].sum() < 5 or event[s].sum() > len(s) - 5:
            continue
        d.append(roc_auc_score(event[s], risk_b[s]) - roc_auc_score(event[s], risk_a[s]))
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(2 * min((d <= 0).mean(), (d >= 0).mean()))


# --------------------------------------------------------------------------- #
# 1. reference models: MELD and MELD + 5-protein panel (locked, from dev)
# --------------------------------------------------------------------------- #
m_meld = fit(dev, ["meld"])
m_panel = fit(dev, ["meld"] + PANEL)
m_crp = fit(dev, ["meld"] + PANEL + ["CRP"])

# locked panel score: panel-only linear predictor (no MELD), fitted on dev
m_prot_only = fit(dev, PANEL)
score_dev_beta = {p: float(m_prot_only.params_[p]) for p in PANEL}
print("panel-only coefficients:", {k: round(v, 4) for k, v in score_dev_beta.items()})
score = sum(score_dev_beta[p] * base[p] for p in PANEL).to_numpy()
score_z = (score - score[dev_mask].mean()) / score[dev_mask].std(ddof=1)
base["panel_score_z"] = score_z

risk_meld = surv90(m_meld, val, ["meld"], baseline_h90(m_meld, dev["center"]))
risk_panel = surv90(m_panel, val, ["meld"] + PANEL, baseline_h90(m_panel, dev["center"]))
risk_crp = surv90(m_crp, val, ["meld"] + PANEL + ["CRP"], baseline_h90(m_crp, dev["center"]))
ev_val = val["event"].to_numpy()

# --------------------------------------------------------------------------- #
# 2. A — panel score within MELD tertiles
# --------------------------------------------------------------------------- #
tert, edges = pd.qcut(meld, 3, labels=["T1", "T2", "T3"], retbins=True)
rows = []
for lab in ["T1", "T2", "T3"]:
    m = np.asarray(tert == lab)
    lo, hi = meld[m].min(), meld[m].max()
    df = base[m].copy()
    cph = fit(df, ["panel_score_z"])
    s = cph.summary.loc["panel_score_z"]
    rows.append(dict(tertile=lab, meld_low=round(float(lo), 1), meld_high=round(float(hi), 1),
                     n=int(m.sum()), events=int(event90[m].sum()),
                     HR=float(np.exp(s["coef"])),
                     CI_low=float(np.exp(s["coef lower 95%"])),
                     CI_high=float(np.exp(s["coef upper 95%"])),
                     p=float(s["p"])))
# interaction on the continuous scale
di = base.copy()
di["meld_c"] = (di["meld"] - di["meld"].mean()) / di["meld"].std(ddof=1)
m0 = fit(di, ["panel_score_z", "meld_c"])
di["panel_x_meld"] = di["panel_score_z"] * di["meld_c"]
m1 = fit(di, ["panel_score_z", "meld_c", "panel_x_meld"])
lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
p_int = float(stats.chi2.sf(lr, 1))
p_int_coef = float(m1.summary.loc["panel_x_meld", "p"])
tert_df = pd.DataFrame(rows)
tert_df.to_csv(OUT / "meld_tertile.csv", index=False, encoding="utf-8-sig")
print(tert_df.to_string(index=False))
print("interaction LRT p=%.3f (coef p=%.3f)" % (p_int, p_int_coef))

# --------------------------------------------------------------------------- #
# 3. B — 5-protein panel versus 5 proteins + CRP
# --------------------------------------------------------------------------- #
auc_meld = roc_auc_score(ev_val, risk_meld)
auc_panel = roc_auc_score(ev_val, risk_panel)
auc_crp = roc_auc_score(ev_val, risk_crp)
lo_p, hi_p, p_p = boot_delta_auc(risk_meld, risk_panel, ev_val)
lo_c, hi_c, p_c = boot_delta_auc(risk_panel, risk_crp, ev_val)

# continuous NRI and IDI, CRP-augmented vs 5-protein (both vs MELD too)
def nri_idi(r_old, r_new, y):
    up = r_new > r_old
    down = r_new < r_old
    c, k = y == 1, y == 0
    nri = (up[c].mean() - down[c].mean()) + (down[k].mean() - up[k].mean())
    idi = (r_new[c].mean() - r_old[c].mean()) - (r_new[k].mean() - r_old[k].mean())
    return float(nri), float(idi)


nri_crp, idi_crp = nri_idi(risk_panel, risk_crp, ev_val)
nri_panel, idi_panel = nri_idi(risk_meld, risk_panel, ev_val)
lr_crp = 2 * (m_crp.log_likelihood_ - m_panel.log_likelihood_)
p_lr_crp = float(stats.chi2.sf(lr_crp, 1))
crp_hr = float(np.exp(m_crp.params_["CRP"]))
crp_p = float(m_crp.summary.loc["CRP", "p"])

# also: CRP alone instead of the panel
m_crponly = fit(dev, ["meld", "CRP"])
risk_crponly = surv90(m_crponly, val, ["meld", "CRP"], baseline_h90(m_crponly, dev["center"]))
auc_crponly = roc_auc_score(ev_val, risk_crponly)

crp_df = pd.DataFrame([
    dict(model="MELD", n_param=1, AUC90=auc_meld, dAUC_vs_MELD=0.0,
         dAUC_CI_low=np.nan, dAUC_CI_high=np.nan, NRI_vs_MELD=np.nan, IDI_vs_MELD=np.nan),
    dict(model="MELD + CRP", n_param=2, AUC90=auc_crponly,
         dAUC_vs_MELD=auc_crponly - auc_meld, dAUC_CI_low=np.nan, dAUC_CI_high=np.nan,
         NRI_vs_MELD=np.nan, IDI_vs_MELD=np.nan),
    dict(model="MELD + 5-protein", n_param=5, AUC90=auc_panel,
         dAUC_vs_MELD=auc_panel - auc_meld, dAUC_CI_low=lo_p, dAUC_CI_high=hi_p,
         NRI_vs_MELD=nri_panel, IDI_vs_MELD=idi_panel),
    dict(model="MELD + 5-protein + CRP", n_param=6, AUC90=auc_crp,
         dAUC_vs_MELD=auc_crp - auc_meld, dAUC_CI_low=lo_c, dAUC_CI_high=hi_c,
         NRI_vs_MELD=nri_crp, IDI_vs_MELD=idi_crp),
])
crp_df.to_csv(OUT / "crp_model.csv", index=False, encoding="utf-8-sig")
print(crp_df.to_string(index=False))
print("CRP in augmented model: HR %.2f, p=%.3f, LRT p=%.3f" % (crp_hr, crp_p, p_lr_crp))

# --------------------------------------------------------------------------- #
# 4. C — win ratio, 90-day composite, stratified by MELD decile
# --------------------------------------------------------------------------- #
def win_ratio(score_v, eve, meld_v, group_cut=0.0, nstr=10):
    """Unmatched win ratio (Finkelstein-Schoenfeld) within MELD strata.

    Orientation: group A = panel score <= cut (lower risk, the "exposed to a
    better profile"), group B = score > cut.  For a pair (a, b) in the same
    stratum with a single composite endpoint, a wins when a is event-free at
    90 days and b had an event, and loses in the reverse case; otherwise the
    pair is tied.  A win ratio > 1 therefore means the low-score group fares
    better, which is the clinically readable direction.
    """
    grp = score_v > group_cut
    strata = pd.qcut(meld_v, nstr, labels=False, duplicates="drop")
    win = loss = tie = 0
    for s in np.unique(strata):
        m = strata == s
        hi, lo = m & grp, m & (~grp)
        if hi.sum() == 0 or lo.sum() == 0:
            continue
        eh, el = eve[hi], eve[lo]
        n_ev_h, n_ev_l = int(eh.sum()), int(el.sum())
        win += int((el == 0).sum() * n_ev_h)      # low survives, high dies
        loss += int(n_ev_l * (eh == 0).sum())
        tie += int(n_ev_h * n_ev_l + (el == 0).sum() * (eh == 0).sum())
    return win, loss, tie


w, l, t = win_ratio(score_z, event90, meld)
wr = w / l if l else np.inf
bs = []
idx = np.arange(len(clin))
for _ in range(1000):
    s = RNG.choice(idx, len(idx), replace=True)
    ww, ll, _ = win_ratio(score_z[s], event90[s], meld[s])
    if ll:
        bs.append(ww / ll)
bs = np.array(bs)
wr_rows = [dict(comparison="All patients (MELD decile stratified)",
                wins=w, losses=l, ties=t, win_ratio=wr,
                CI_low=float(np.percentile(bs, 2.5)), CI_high=float(np.percentile(bs, 97.5)),
                n_boot=len(bs))]
# by MELD tertile (stratified within tertile, no cross-tertile pairing)
for lab in ["T1", "T2", "T3"]:
    m = np.asarray(tert == lab)
    ww, ll, tt = win_ratio(score_z[m], event90[m], meld[m], nstr=4)
    sub = np.where(m)[0]
    bs_t = []
    for _ in range(500):
        s = RNG.choice(sub, len(sub), replace=True)
        a, b, _ = win_ratio(score_z[s], event90[s], meld[s], nstr=4)
        if b:
            bs_t.append(a / b)
    bs_t = np.array(bs_t) if bs_t else np.array([np.nan])
    wr_rows.append(dict(comparison=f"{lab} (MELD quartile stratified)",
                        wins=ww, losses=ll, ties=tt,
                        win_ratio=(ww / ll if ll else np.nan),
                        CI_low=float(np.nanpercentile(bs_t, 2.5)),
                        CI_high=float(np.nanpercentile(bs_t, 97.5)),
                        n_boot=len(bs_t)))
    print("  %s: wins=%d losses=%d WR=%s" % (lab, ww, ll, (ww / ll if ll else "inf (0 losses)")))
wr_df = pd.DataFrame(wr_rows)
wr_df.to_csv(OUT / "win_ratio.csv", index=False, encoding="utf-8-sig")
print(wr_df.to_string(index=False))

# --------------------------------------------------------------------------- #
# 5. D — decision-curve analysis (external validation, 90 days)
# --------------------------------------------------------------------------- #
thr = np.arange(0.05, 0.601, 0.01)
n_val = len(ev_val)
prev = ev_val.mean()
dca = []
for pt in thr:
    for name, r in [("MELD", risk_meld), ("MELD + 5-protein", risk_panel)]:
        pos = r >= pt
        tp = int((pos & (ev_val == 1)).sum())
        fp = int((pos & (ev_val == 0)).sum())
        nb = tp / n_val - fp / n_val * pt / (1 - pt)
        dca.append(dict(threshold=round(float(pt), 3), strategy=name, net_benefit=nb,
                        n_treat=int(pos.sum())))
    dca.append(dict(threshold=round(float(pt), 3), strategy="treat all",
                    net_benefit=prev - (1 - prev) * pt / (1 - pt), n_treat=n_val))
    dca.append(dict(threshold=round(float(pt), 3), strategy="treat none",
                    net_benefit=0.0, n_treat=0))
dca_df = pd.DataFrame(dca)
dca_df.to_csv(OUT / "dca.csv", index=False, encoding="utf-8-sig")
# net reduction in "treat for no reason" at the clinically relevant thresholds
hi_t = dca_df[(dca_df.strategy == "treat all")].set_index("threshold")["net_benefit"]
gain = []
for pt in (0.15, 0.25, 0.35):
    a = dca_df[(dca_df.strategy == "MELD + 5-protein") & (np.isclose(dca_df.threshold, pt))]["net_benefit"].iloc[0]
    b = dca_df[(dca_df.strategy == "MELD") & (np.isclose(dca_df.threshold, pt))]["net_benefit"].iloc[0]
    gain.append(dict(threshold=pt, nb_panel=float(a), nb_meld=float(b), nb_diff=float(a - b)))
print(pd.DataFrame(gain).to_string(index=False))

# --------------------------------------------------------------------------- #
# 6. E — selection funnel
# --------------------------------------------------------------------------- #
coxall = pd.read_csv(BASE / "figures" / "fig3" / "matrices" / "cox90_all.csv")
n_tested = len(coxall)
n_sig = int((coxall["FDR"] < 0.05).sum())          # full-cohort descriptive count
scrd = pd.read_csv(BASE / "analysis" / "output2" / "screen_dev_primary883.csv")
n_pool = int((scrd["FDR"] < 0.05).sum())           # derivation-only candidate pool
lock = pd.read_csv(BASE / "analysis" / "output2" / "locking_rules_primary883.csv")
n_stab = int(lock.loc[lock.rule.str.startswith("(i) stability"), "n"].iloc[0])
funnel = pd.DataFrame([
    dict(step=1, label="Proteins assayed", n=883,
         detail="targeted plasma proteomic panel; no detection filter (protocol v1.2)"),
    dict(step=2, label="Associated with 90-day outcome (full cohort)", n=n_sig,
         detail="descriptive: Cox per SD, sex + MELD adjusted, centre-stratified, BH-FDR<0.05 (%.1f%% of %d)" % (100 * n_sig / n_tested, n_tested)),
    dict(step=3, label="Candidate pool in the derivation set", n=n_pool,
         detail="same model within the 5 derivation centres only (prevents validation leakage)"),
    dict(step=4, label="Stability-selected at lambda_1se", n=n_stab,
         detail="300 derivation bootstrap resamples; largest-gap frequency threshold 0.795"),
    dict(step=5, label="Primary locking (883 protocol)", n=3,
         detail="assayability >=80% drops FGF-5 and Bcl-w; joint-model backward drop of ANG-2 (P=0.63) -> TNF-RI, DR6, Renin"),
    dict(step=6, label="Sensitivity locking (665 protocol)", n=5,
         detail="TNF-RI, DR6, Renin, SREC-II, GH"),
    dict(step=7, label="Locked panel (union of the two pre-specified lockings)", n=5,
         detail="TNF-RI, DR6, Renin, SREC-II, GH"),
])
funnel.to_csv(OUT / "funnel.csv", index=False, encoding="utf-8-sig")
print(funnel.to_string(index=False))

# --------------------------------------------------------------------------- #
summary = dict(
    n=len(clin), events90=int(event90.sum()),
    dev=dict(n=int(dev_mask.sum()), events=int(event90[dev_mask].sum())),
    val=dict(n=int((~dev_mask).sum()), events=int(event90[~dev_mask].sum())),
    panel_beta=score_dev_beta,
    meld_tertile_interaction_p=p_int,
    auc=dict(MELD=auc_meld, MELD_CRP=auc_crponly, MELD_panel=auc_panel,
             MELD_panel_CRP=auc_crp),
    dAUC_panel=dict(est=auc_panel - auc_meld, CI=[lo_p, hi_p], p=p_p),
    dAUC_crp=dict(est=auc_crp - auc_panel, CI=[lo_c, hi_c], p=p_c),
    nri_idi_panel=dict(NRI=nri_panel, IDI=idi_panel),
    nri_idi_crp=dict(NRI=nri_crp, IDI=idi_crp),
    crp=dict(HR=crp_hr, p=crp_p, lrt_p=p_lr_crp),
    win_ratio=dict(WR=wr, wins=int(w), losses=int(l), ties=int(t),
                   CI=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]),
    funnel=dict(assayed=883, prognostic_full=int(n_sig), pool_dev=int(n_pool),
                stability=int(n_stab), lock_primary=3, lock_filter=5, locked=5),
    dca_gain=gain,
)
(OUT / "robustness2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
print("\nDONE ->", OUT)
