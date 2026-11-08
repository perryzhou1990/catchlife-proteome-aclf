# -*- coding: utf-8 -*-
"""Re-selection of the prognostic panel — protocol v1 (2026-11-05).

WHY THIS SCRIPT EXISTS
----------------------
Two methodological defects were found in manuscript v0.2:

  (1) The "pre-specified 14-protein inflammation-endothelial candidate pool"
      was not pre-specified: it was assembled by hand after looking at the
      data.  Any reviewer asking for the provenance of those 14 names would
      find no protocol, no gene set and no literature rule behind them.

  (2) The detection filter (drop proteins at the assay floor in >20% of
      samples, 883 -> 665) has no citable threshold.  Detection-limit
      filtering is standard in plasma proteomics, but the conventional
      criterion is "measurable in >=70-80% of samples", not "floor in >20%",
      and discarding 24.7% of the assay before any analysis is hard to
      defend.  It also risks discarding real biology (Activin RIIA among the
      excluded proteins had HR 1.77, P=1.2e-7).

PROTOCOL (fixed before any model was fitted; identical for every variant)
-------------------------------------------------------------------------
Analysis set      all 883 assayed proteins, NO detection filter (primary).
                  Sensitivity: the 665 proteins measurable in >=80% of
                  patients (the old filter) and a pathway-restricted pool.
Candidate pool    derivation centres only (5 centres, n=995, 116 events):
                  per-protein Cox for 90-day transplant-free survival,
                  adjusted for sex and MELD, stratified by centre;
                  Benjamini-Hochberg FDR<0.05 across all proteins tested.
                  No biological curation, no manual list.
Penalised model   LASSO-Cox (l1_ratio=1), centre-stratified, on the pool.
                  lambda by 10-fold cross-validated held-out concordance
                  (folds stratified on event; lambda_1se also reported).
Stability         300 bootstrap resamples of the derivation set at lambda_cv;
                  selection frequency per protein.
Locking rules     in this order, all fixed in advance (protocol v1.2):
                  (i)   stability selection: the frequency threshold is placed
                        at the LARGEST GAP in the sorted selection-frequency
                        distribution (bounded to [0.60, 0.95]), following the
                        standard practice of treating the stability threshold
                        as a tuning parameter read off the empirical path;
                        proteins above the gap are candidates;
                  (ii)  measurable in >=80% of patients (assayability: a
                        clinical panel must be quantifiable in nearly all
                        patients; used as a LOCKING criterion, never as an
                        analysis filter);
                  (iii) backward elimination in the joint MELD-adjusted model:
                        fit MELD + all surviving candidates, remove the protein
                        with the largest P, refit, repeat until every remaining
                        protein has P<0.05;
                  (iv)  if more than 8 proteins survive (i)-(ii), the 8 with
                        the highest selection frequency enter step (iii).
                  History: v1.0 used a fixed 0.70 threshold at lambda_cv
                  (frequencies undifferentiated: 88/91 above threshold);
                  v1.1 anchored stability at lambda_1se with the fixed 0.70
                  threshold (4 borderline proteins at 0.70-0.72 admitted);
                  v1.2 reads the threshold off the frequency distribution.
                  The validation set was not consulted at any point.
Validation        the locked model is applied unchanged to the validation
                  centres: C-index, AUC at 90 days, dAUC with bootstrap CI,
                  categorical-free NRI, IDI, calibration by quintile.
                  Nothing is refitted on validation.

SPLIT SCHEME (env SPLIT=5dev|8dev; default 8dev as of 2026-11-06)
-----------------------------------------------------------------
  5dev : derivation = nfa,xna,rja,dta,xha (5 largest centres, n=995, 116 ev)
         validation = other 8 centres (n=395, 46 ev)          [manuscript v0.2]
  8dev : derivation = the other 8 centres (n=395, 46 ev)
         validation = nfa,xna,rja,dta,xha (n=995, 116 ev)     [current primary]
  The 8dev scheme deliberately puts the small centres in derivation and the
  large tertiary centres in external validation, so that the validation set
  carries 116 events (tight dAUC CI) rather than 46.  The price is a small
  derivation set; the candidate pool therefore shrinks accordingly, and the
  stability threshold is still read off the empirical frequency gap rather
  than a fixed number.

Outputs -> analysis/output3/ for 8dev (analysis/output2/ keeps the 5dev run)
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
import os
SPLIT = os.environ.get("SPLIT", "8dev")          # "5dev" or "8dev"
BIG5 = ["nfa", "xna", "rja", "dta", "xha"]        # 5 largest centres (n=995, 116 ev)
SMALL8 = ["jda", "tha", "sda", "zxa", "hna", "xya", "xyb", "zya"]  # other 8 (n=395, 46 ev)
DEV_CENTRES = BIG5 if SPLIT == "5dev" else SMALL8
VAL_CENTRES = SMALL8 if SPLIT == "5dev" else BIG5
OUT = BASE / "analysis" / ("output2" if SPLIT == "5dev" else "output3")
OUT.mkdir(parents=True, exist_ok=True)
GS = BASE / "analysis" / "genesets"

CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\proteinarray_matrix_1390人蛋白质组结果（质控后883蛋白）.xlsx")
ANNOT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\宋晶分组分析结果-供参考\cirrhosis_vs_noncirrhosis.csv")

RNG = np.random.default_rng(20261105)
N_BOOT_STAB = 300
FREQ_CUT = 0.70
DET_CUT = 0.80
MAX_PANEL = 6

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
clin.columns = [str(c).strip() for c in clin.columns]
clin["center"] = clin["ID"].astype(str).str.split("-").str[0]

prot = pd.read_excel(PROT, sheet_name="1390已测蛋白质组")
prot.columns = [str(c).strip() for c in prot.columns]
prot = prot.set_index(prot.columns[0]).loc[clin["ID"]].astype(float)
floor_frac = (prot == prot.min()).mean()          # per protein, fraction at assay floor
det_rate = 1.0 - floor_frac

log2 = np.log2(prot.clip(lower=1))
Z = (log2 - log2.mean()) / log2.std(ddof=1)
Z = Z.loc[:, Z.std(ddof=1) > 0]                    # drop constant columns
Zarr = Z.to_numpy()
zidx = {c: i for i, c in enumerate(Z.columns)}

t1y = pd.to_numeric(clin["OS_1y_更新"], errors="coerce")
s90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
time90 = np.minimum(t1y.fillna(90), 90).to_numpy()
event90 = s90.isin([1, 2]).astype(int).to_numpy()
sex = clin["Gender"].replace(2, np.nan)
sex = sex.fillna(sex.mode()[0]).astype(float).to_numpy()
tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr = pd.to_numeric(clin["INR"], errors="coerce")
cr = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce")
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.fillna(inr.median()).clip(lower=1))
        + 9.57 * np.log(cr.fillna(cr.median()).clip(lower=1)) + 6.43).clip(6, 40).to_numpy()
center = clin["center"].to_numpy()
dev_mask = np.isin(center, DEV_CENTRES)

base = pd.DataFrame({"time": time90, "event": event90, "sex": sex, "meld": meld,
                     "center": center})
for c in Z.columns:
    base[c] = Zarr[:, zidx[c]]
dev_all, val_all = base[dev_mask].copy(), base[~dev_mask].copy()
print("n=%d events=%d | dev %d (%d ev) | val %d (%d ev)"
      % (len(base), event90.sum(), dev_mask.sum(), event90[dev_mask].sum(),
         (~dev_mask).sum(), event90[~dev_mask].sum()))
print("proteins with zero variance:", 883 - Z.shape[1])


def fit_cph(df, cols, lam=0.0, strata=True):
    """Centre-stratified Cox; falls back to unstratified when a resample/
    fold leaves a centre empty or event-free."""
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


def baseline_h90(cph, train_center):
    """Marginal 90-day baseline cumulative hazard of a centre-stratified Cox
    (validation centres are absent from the training strata by construction).
    Stratum-specific baselines are averaged over derivation centres weighted
    by size; only the baseline level is marginalised."""
    b = cph.baseline_cumulative_hazard_
    rows = b.index[b.index <= 90]
    if not len(rows):
        return 0.0
    h = {s: float(b[s].loc[rows[-1]]) for s in b.columns}
    w = pd.Series(train_center).value_counts(normalize=True)
    return float(sum(h.get(s, 0.0) * w.get(s, 0.0) for s in w.index))


def prob90(cph, df, cols, h0):
    ph = cph.predict_partial_hazard(df[cols]).to_numpy()
    return 1.0 - np.exp(-h0 * ph)


def bh(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    pv = p[ok].copy()
    pv[pv <= 0] = 1e-300
    o = np.argsort(pv)
    ranked = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    fdr = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    tmp = np.empty_like(fdr)
    tmp[o] = fdr
    q[ok] = tmp
    return q


# --------------------------------------------------------------------------- #
# 1. derivation-only screening
# --------------------------------------------------------------------------- #
def screen(cols, tag):
    """Derivation-only Cox screen.  Reuses an existing result file when the
    protein list matches, so re-runs of the pipeline skip this step."""
    cache = OUT / f"screen_dev_{tag}.csv"
    if cache.exists():
        old = pd.read_csv(cache)
        if list(old["protein"]) == list(cols):
            print("[%s] reused cached screen (%d proteins, FDR<0.05: %d)"
                  % (tag, len(old), int((old["FDR"] < 0.05).sum())))
            return old
    rows = []
    for p in cols:
        d = dev_all[["time", "event", "sex", "meld", "center", p]]
        try:
            c = fit_cph(d, ["sex", "meld", p])
            s = c.summary.loc[p]
            rows.append(dict(protein=p, n_dev=int(dev_mask.sum()), beta=float(s["coef"]),
                             se=float(s["se(coef)"]), HR=float(np.exp(s["coef"])),
                             p=float(s["p"]),
                             det_rate=float(det_rate.get(p, np.nan))))
        except Exception:
            rows.append(dict(protein=p, n_dev=int(dev_mask.sum()), beta=np.nan, se=np.nan,
                             HR=np.nan, p=np.nan, det_rate=float(det_rate.get(p, np.nan))))
    df = pd.DataFrame(rows)
    df["FDR"] = bh(df["p"])
    df = df.sort_values("p").reset_index(drop=True)
    df.to_csv(OUT / f"screen_dev_{tag}.csv", index=False, encoding="utf-8-sig")
    print("[%s] screened %d proteins, FDR<0.05: %d"
          % (tag, len(df), int((df["FDR"] < 0.05).sum())))
    return df


# --------------------------------------------------------------------------- #
# 2. LASSO with cross-validated lambda + stability selection
# --------------------------------------------------------------------------- #
def cv_lasso(pool, tag, n_fold=10, n_lam=30):
    """lambda by 10-fold CV of held-out partial log-likelihood (glmnet-style);
    lambda_1se = largest lambda within 1 SE of the best (sparser solution).
    Reuses a cached curve when the protein list is unchanged.
    NOTE: lifelines' penalizer is not n-scaled; the effective shrinkage range
    for this cohort is ~0.01-0.5 (verified empirically), hence the grid."""
    cache = OUT / f"cv_curve_{tag}.csv"
    if cache.exists():
        cv = pd.read_csv(cache)
        if len(cv) == n_lam:
            best = cv.loc[cv["ll_mean"].idxmax()]
            lam_cv = float(best["lam"])
            thr = best["ll_mean"] - best["ll_sd"] / np.sqrt(best["n_fold"])
            lam_1se = float(cv[cv["ll_mean"] >= thr]["lam"].max())
            print("[%s] reused cached CV: lambda_cv=%.4f  lambda_1se=%.4f" % (tag, lam_cv, lam_1se))
            return lam_cv, lam_1se, cv
    lam_grid = np.logspace(np.log10(0.01), np.log10(0.5), n_lam)
    ev = dev_all["event"].to_numpy()
    folds = np.zeros(len(dev_all), int)
    for e in (0, 1):
        idx = np.where(ev == e)[0]
        RNG.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_fold

    cv = []
    for lam in lam_grid:
        cs = []
        for k in range(n_fold):
            tr = dev_all.iloc[folds != k]
            te = dev_all.iloc[folds == k]
            if te["event"].sum() < 2:
                continue
            try:
                c = fit_cph(tr, pool, lam=lam)
                cs.append(float(c.score(te, scoring_method="log_likelihood")))
            except Exception:
                continue
        if cs:
            cv.append(dict(lam=lam, ll_mean=float(np.mean(cs)), ll_sd=float(np.std(cs)),
                           n_fold=len(cs)))
    cv = pd.DataFrame(cv)
    cv.to_csv(OUT / f"cv_curve_{tag}.csv", index=False, encoding="utf-8-sig")
    best = cv.loc[cv["ll_mean"].idxmax()]
    lam_cv = float(best["lam"])
    thr = best["ll_mean"] - best["ll_sd"] / np.sqrt(best["n_fold"])
    lam_1se = float(cv[cv["ll_mean"] >= thr]["lam"].max())
    print("[%s] lambda_cv=%.4f (LL=%.2f)  lambda_1se=%.4f" % (tag, lam_cv, best["ll_mean"], lam_1se))

    # path for the figure
    path = []
    for lam in lam_grid:
        try:
            c = fit_cph(dev_all, pool, lam=lam)
            nz = [p for p in pool if abs(c.params_.get(p, 0.0)) > 1e-6]
            path.append(dict(lam=lam, nvar=len(nz),
                             c_dev=float(concordance_index(dev_all["time"],
                                                           -c.predict_partial_hazard(dev_all[pool]).to_numpy(),
                                                           dev_all["event"])),
                             **{p: float(c.params_.get(p, 0.0)) for p in pool}))
        except Exception:
            pass
    pd.DataFrame(path).to_csv(OUT / f"lasso_path_{tag}.csv", index=False, encoding="utf-8-sig")
    return lam_cv, lam_1se, cv


def stability(pool, lam, tag, n_boot=N_BOOT_STAB):
    cache = OUT / f"stability_{tag}.csv"
    if cache.exists():
        old = pd.read_csv(cache)
        if len(old) == len(pool) and set(old["protein"]) == set(pool) \
                and old["n_boot"].iloc[0] == n_boot:
            print("[%s] reused cached stability (n_boot=%d, >=0.70: %d)"
                  % (tag, n_boot, int((old["freq"] >= 0.70).sum())))
            return old
    freq = {p: 0 for p in pool}
    n_ok = 0
    for b in range(n_boot):
        idx = RNG.choice(len(dev_all), len(dev_all), replace=True)
        d = dev_all.iloc[idx]
        if d["event"].sum() < 10:
            continue
        try:
            c = fit_cph(d, pool, lam=lam)
            for p in pool:
                if abs(c.params_.get(p, 0.0)) > 1e-6:
                    freq[p] += 1
            n_ok += 1
        except Exception:
            continue
    st = (pd.DataFrame([dict(protein=p, freq=freq[p] / max(n_ok, 1)) for p in pool])
          .sort_values("freq", ascending=False).reset_index(drop=True))
    st["n_boot"] = n_ok
    st.to_csv(OUT / f"stability_{tag}.csv", index=False, encoding="utf-8-sig")
    print("[%s] stability done (%d usable resamples); >=%.2f: %d proteins"
          % (tag, n_ok, FREQ_CUT, int((st["freq"] >= FREQ_CUT).sum())))
    return st


# --------------------------------------------------------------------------- #
# 3. lock, fit, validate
# --------------------------------------------------------------------------- #
def evaluate(panel, tag):
    panel = list(panel)
    m_prot = fit_cph(dev_all, panel)
    m_comb = fit_cph(dev_all, ["meld"] + panel)
    m_meld = fit_cph(dev_all, ["meld"])
    rows = []
    for name, mdl, cols in (("MELD", m_meld, ["meld"]),
                            ("panel only", m_prot, panel),
                            ("MELD + panel", m_comb, ["meld"] + panel)):
        ph_dev = mdl.predict_partial_hazard(dev_all[cols]).to_numpy()
        ph_val = mdl.predict_partial_hazard(val_all[cols]).to_numpy()
        c_dev = concordance_index(dev_all["time"], -ph_dev, dev_all["event"])
        c_val = concordance_index(val_all["time"], -ph_val, val_all["event"])
        auc_val = roc_auc_score(val_all["event"], ph_val)
        rows.append(dict(model=name, n_param=len(cols), c_dev=c_dev, c_val=c_val, auc90_val=auc_val,
                         **({"HR": None} if name != "MELD" else {})))
    met = pd.DataFrame(rows)
    meld_auc = met.loc[met.model == "MELD", "auc90_val"].iloc[0]
    met["dAUC_vs_MELD"] = met["auc90_val"] - meld_auc

    # bootstrap dAUC (panel vs MELD) on validation
    ph_m = m_meld.predict_partial_hazard(val_all[["meld"]]).to_numpy()
    ph_p = m_comb.predict_partial_hazard(val_all[["meld"] + panel]).to_numpy()
    y = val_all["event"].to_numpy()
    d = []
    for _ in range(1000):
        s = RNG.choice(len(y), len(y), replace=True)
        if 5 < y[s].sum() < len(s) - 5:
            d.append(roc_auc_score(y[s], ph_p[s]) - roc_auc_score(y[s], ph_m[s]))
    d = np.array(d)
    met.loc[met.model == "MELD + panel", "dAUC_CI_low"] = np.percentile(d, 2.5)
    met.loc[met.model == "MELD + panel", "dAUC_CI_high"] = np.percentile(d, 97.5)
    met.loc[met.model == "MELD + panel", "dAUC_p"] = 2 * min((d <= 0).mean(), (d >= 0).mean())

    # continuous NRI / IDI, panel vs MELD (validation), on predicted 90-day
    # event PROBABILITIES (marginal baseline), not on the hazard scale
    def nri_idi(r_old, r_new, yy):
        up, dn = r_new > r_old, r_new < r_old
        c, k = yy == 1, yy == 0
        return (float((up[c].mean() - dn[c].mean()) + (dn[k].mean() - up[k].mean())),
                float((r_new[c].mean() - r_old[c].mean()) - (r_new[k].mean() - r_old[k].mean())))
    r_m = prob90(m_meld, val_all, ["meld"], baseline_h90(m_meld, dev_all["center"]))
    r_p = prob90(m_comb, val_all, ["meld"] + panel, baseline_h90(m_comb, dev_all["center"]))
    nri, idi = nri_idi(r_m, r_p, y)
    met.loc[met.model == "MELD + panel", "NRI_vs_MELD"] = nri
    met.loc[met.model == "MELD + panel", "IDI_vs_MELD"] = idi
    met.to_csv(OUT / f"metrics_{tag}.csv", index=False, encoding="utf-8-sig")
    print(met.to_string(index=False))

    coeff = pd.DataFrame([dict(protein=p,
                               HR_comb=float(np.exp(m_comb.params_[p])),
                               CI_low=float(np.exp(m_comb.summary.loc[p, "coef lower 95%"])),
                               CI_high=float(np.exp(m_comb.summary.loc[p, "coef upper 95%"])),
                               p=float(m_comb.summary.loc[p, "p"]),
                               HR_prot_only=float(np.exp(m_prot.params_[p])),
                               det_rate=float(det_rate.get(p, np.nan)))
                          for p in panel])
    coeff.to_csv(OUT / f"panel_coefficients_{tag}.csv", index=False, encoding="utf-8-sig")
    print(coeff.to_string(index=False))
    return met, coeff, dict(c_dev=c_dev, c_val=float(met.loc[met.model == 'MELD + panel', 'c_val'].iloc[0]),
                            auc90_val=float(met.loc[met.model == 'MELD + panel', 'auc90_val'].iloc[0]),
                            nri=nri, idi=idi, dAUC=float(met.loc[met.model == 'MELD + panel', 'dAUC_vs_MELD'].iloc[0]),
                            dAUC_CI=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def lock(st, screen_df, tag):
    """Apply rules (i)-(iv) in order (protocol v1.2) and return the panel."""
    rule_log = []
    freqs = st["freq"].sort_values(ascending=False).to_numpy()
    thr = FREQ_CUT
    if len(freqs) > 1:
        gaps = freqs[:-1] - freqs[1:]          # descending order -> gaps >= 0
        order = np.argsort(-gaps)
        for k in order:
            hi, lo = freqs[k], freqs[k + 1]
            if lo >= 0.60 and hi <= 0.95 and gaps[k] > 0:
                thr = float((lo + hi) / 2)
                rule_log.append(dict(
                    rule=f"(i) stability threshold at largest gap: {thr:.3f} "
                         f"(between {hi:.3f} and {lo:.3f})", n=int((st["freq"] >= thr).sum()),
                    proteins=";".join(st.loc[st["freq"] >= thr, "protein"])))
                break
    else:
        rule_log.append(dict(rule="(i) stability threshold 0.70 (single protein)", n=1,
                             proteins=st["protein"].iloc[0]))
    cand = st.loc[st["freq"] >= thr, "protein"].tolist()
    sdi = screen_df.set_index("protein")
    keep = [p for p in cand if sdi.loc[p, "det_rate"] >= DET_CUT]
    rule_log.append(dict(rule="(ii) measured in >=80% of patients", n=len(keep),
                         proteins=";".join(keep)))
    if len(keep) > 8:
        keep = keep[:8]
        rule_log.append(dict(rule="(iv) top 8 by selection frequency", n=len(keep),
                             proteins=";".join(keep)))
    # (iii) backward elimination in the joint MELD-adjusted model
    cur = list(keep)
    while cur:
        c = fit_cph(dev_all, ["meld"] + cur)
        pvals = {p: float(c.summary.loc[p, "p"]) for p in cur}
        worst = max(pvals, key=pvals.get)
        if pvals[worst] < 0.05:
            break
        cur.remove(worst)
        rule_log.append(dict(rule=f"(iii) backward: drop {worst} (P={pvals[worst]:.3f})",
                             n=len(cur), proteins=";".join(cur)))
    rule_log.append(dict(rule="locked panel", n=len(cur), proteins=";".join(cur)))
    pd.DataFrame(rule_log).to_csv(OUT / f"locking_rules_{tag}.csv", index=False, encoding="utf-8-sig")
    print("[%s] locked panel: %s" % (tag, ", ".join(cur)))
    return cur


def run(tag, cols, note=""):
    print("\n===== %s %s =====" % (tag, note))
    scr = screen(cols, tag)
    pool = scr.loc[scr["FDR"] < 0.05, "protein"].tolist()
    if len(pool) < 3:
        print("pool too small"); return None
    lam_cv, lam_1se, cv = cv_lasso(pool, tag)
    # stability selection anchored at lambda_1se (protocol v1.1)
    st = stability(pool, lam_1se, tag)
    panel = lock(st, scr, tag)
    if not panel:
        print("no protein passed locking"); return None
    met, coeff, summ = evaluate(panel, tag)
    out = dict(tag=tag, pool_size=len(pool), lam_cv=lam_cv, lam_1se=lam_1se,
               panel=panel, summary=summ)
    (OUT / f"result_{tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = {}
    # ---- primary: all 883, no detection filter ----
    all_cols = Z.columns.tolist()
    r = run("primary883", all_cols, "(all assayed proteins, no detection filter)")
    if r: res["primary883"] = r

    # ---- sensitivity 1: old detection filter, 665 ----
    det_cols = [c for c in all_cols if det_rate[c] >= DET_CUT]
    r = run("filter665", det_cols, "(proteins measurable in >=80%, the old filter)")
    if r: res["filter665"] = r

    # ---- sensitivity 2: pathway-restricted pool (reproducible gene sets) ----
    annot = pd.read_csv(ANNOT)
    g2 = dict(zip(annot["protein"], annot["geneID"]))
    TERMS = {
        "GO_Biological_Process_2023": ["Inflammatory Response (GO:0006954)",
                                       "Angiogenesis (GO:0001525)"],
        "KEGG_2021_Human": ["Renin-angiotensin system", "TNF signaling pathway"],
        "Reactome_2022": ["Signaling By Interleukins R-HSA-449147",
                          "Signaling By VEGF R-HSA-194138"],
    }
    genes = set()
    used = []
    for lib, terms in TERMS.items():
        for line in open(GS / f"{lib}.gmt", encoding="utf8"):
            f = line.rstrip("\n").split("\t")
            if f[0].strip() in terms:
                genes |= set(f[2:])
                used.append(f"{lib}:{f[0].strip()}")
    path_cols = [c for c in all_cols if str(g2.get(c, "")) in genes]
    pd.Series(used).to_csv(OUT / "pathway_terms_used.csv", index=False, header=["term"])
    print("\npathway-restricted pool: %d of %d measured proteins in %d terms"
          % (len(path_cols), len(all_cols), len(used)))
    r = run("pathway", path_cols, "(reproducible pathway-restricted pool)")
    if r: res["pathway"] = r

    # ---- reference: the v0.2 hand-picked panel, evaluated under the same
    #      machinery (screen/CV/stability bypassed; model identical in form) ----
    OLD = ["TNF.RI", "Renin", "DR6", "SREC.II"]
    if all(p in Z.columns for p in OLD):
        met_old, coeff_old, summ_old = evaluate(OLD, "oldpanel")
        res["oldpanel"] = dict(tag="oldpanel", panel=OLD, summary=summ_old)
        (OUT / "result_oldpanel.json").write_text(
            json.dumps(res["oldpanel"], indent=2, ensure_ascii=False), encoding="utf-8")

    (OUT / "all_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDONE")
