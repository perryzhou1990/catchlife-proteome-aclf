# -*- coding: utf-8 -*-
"""Fig 6 A/B: cis-pQTL Mendelian randomisation + colocalisation (cirrhosis, GCST90319878).

Revision history
----------------
v1 (2026-11-05)  initial run.
v2 (2026-11-06)  two method-level fixes, see analysis/06_MR方法学审计_v1.md:

  1.1  Weighted-median standard error.  v1 used `se = sqrt(1/sum(w))`, which is the
       IVW standard error, not the standard error of the weighted median.  v2 uses a
       parametric bootstrap of the two GWAS (b* ~ N(b, se) on both sides) and reports
       the bootstrap SD as the standard error, plus a percentile CI and a
       non-parametric bootstrap as a cross-check.  This is the estimator implemented
       by TwoSampleMR::mr_weighted_median().

  1.2  Colocalisation.  v1 summed log-ABFs (`s1 = l1.sum()`), i.e. it computed
       log(prod BF) instead of log(sum BF).  Summing ~2,000 null variants at
       log-ABF ~ -1.4 each drove the regional evidence to -2,500, so a locus with a
       cis-pQTL at P = 3.8e-36 was returned as PP.H0 = 1.000.  v2 aggregates with
       log-sum-exp and implements H3 as S1*S2 - S12 (the standard coloc
       decomposition), so a locus with signal on one side only resolves to PP.H1.
       Prior sensitivity (W in {0.05, 0.15, 0.30} x 0.15^2-as-SD, p12 in 1e-6..1e-4)
       and variant-density sensitivity (no thinning / 10 kb / 5 kb) are written to
       matrices/coloc_sensitivity.csv.

v3 (2026-11-06)  instruments now come from true LD clumping (analysis/mr_ldclump):
       1000 Genomes phase 3 EUR, r2 < 0.001, 10 Mb window, PLINK --clump semantics.
       v2 pruned by a flat 100-kb distance because no LD reference was available, which
       collapsed every locus to a single instrument (TNFRSF1A: 12 significant variants
       spanning 23 kb -> 1; SCARF2: 227 spanning 78 kb -> 1) and made MR-Egger, the
       weighted median and Cochran's Q undefined. Distance is not a valid proxy for LD:
       recombination rate varies more than 100-fold along a chromosome, so variants
       5 kb apart can be independent and variants 300 kb apart can be r2 > 0.8.
       If matrices/ld_clump_instruments.csv is absent the code falls back to distance
       pruning so the pipeline still runs offline.

Outputs: mr_results.csv, mr_instruments.csv, coloc_results.csv, coloc_sensitivity.csv,
         mr_diagnostics.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

BASE = Path(__file__).resolve().parents[3]
FIG6 = BASE / "figures" / "fig6"
RAW, MAT = FIG6 / "raw_data", FIG6 / "matrices"
MAT.mkdir(parents=True, exist_ok=True)
# True-LD instrument sets from analysis/mr_ldclump (1000G EUR, r2 < 0.001).
# Present => used as the primary specification; absent => distance-pruning fallback.
LD_INSTR = MAT / "ld_clump_instruments.csv"

GENES = {"TNFRSF1A": "TNF-RI", "REN": "Renin", "TNFRSF21": "DR6", "SCARF2": "SREC-II",
         "ANGPT2": "ANG-2", "CCL14": "HCC-1", "CRP": "CRP"}
CORE = {"TNFRSF1A", "REN", "TNFRSF21", "SCARF2"}
P_THR = 5e-8
CLUMP_KB = 100          # fallback distance pruning only -- see clump() docstring
W_PRIOR = 0.15 ** 2
NBOOT_WMED = 10000
SEED = 20261106


def prepare_outcome(out: pd.DataFrame) -> pd.DataFrame:
    """Normalise either harmonised GWAS-Catalog layout (beta) or log-OR layout."""
    o = out.copy()
    if "odds_ratio" in o.columns:          # sparse array GWAS (log OR + SE)
        o["b_out"] = np.log(pd.to_numeric(o["odds_ratio"], errors="coerce"))
        o["se_out"] = pd.to_numeric(o["standard_error"], errors="coerce")
        o["p_out"] = pd.to_numeric(o["p_value"], errors="coerce")
        o["eaf_out"] = pd.to_numeric(o["eaf"], errors="coerce")
    else:                                   # dense GWAS (beta + SE)
        o["b_out"] = pd.to_numeric(o["beta"], errors="coerce")
        o["se_out"] = pd.to_numeric(o["se"], errors="coerce")
        o["p_out"] = pd.to_numeric(o["p"], errors="coerce")
        o["eaf_out"] = pd.to_numeric(o["eaf"], errors="coerce")
    o["ea_out"] = o["effect_allele"]
    o["oa_out"] = o["other_allele"]
    return o[["chromosome", "pos", "b_out", "se_out", "ea_out", "oa_out", "eaf_out", "p_out"]]


def harmonise(exp: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Match on (chromosome, position), align effect alleles (palindromic SNPs with
    intermediate allele frequency dropped)."""
    e = exp.rename(columns={"beta": "b_exp", "se": "se_exp", "effect_allele": "ea_exp",
                            "other_allele": "oa_exp", "eaf": "eaf_exp", "p": "p_exp"})
    o = prepare_outcome(out)
    m = e.merge(o, on=["chromosome", "pos"], how="inner", suffixes=("", "_o"))
    m = m.dropna(subset=["b_exp", "se_exp", "b_out", "se_out"])
    same = (m["ea_exp"].str.upper() == m["ea_out"].str.upper()) & \
           (m["oa_exp"].str.upper() == m["oa_out"].str.upper())
    flip = (m["ea_exp"].str.upper() == m["oa_out"].str.upper()) & \
           (m["oa_exp"].str.upper() == m["ea_out"].str.upper())
    m = m[same | flip].copy()
    m["b_out"] = np.where(flip[m.index], -m["b_out"], m["b_out"])
    pal = m.apply(lambda r: {r["ea_exp"].upper(), r["oa_exp"].upper()} in ({"A", "T"}, {"C", "G"}), axis=1)
    m = m[~(pal & m["eaf_exp"].between(0.42, 0.58))]
    return m.reset_index(drop=True)


def clump(m: pd.DataFrame, kb=CLUMP_KB, gene=None) -> pd.DataFrame:
    """Instrument selection.

    Primary path: use the true-LD clumping result in matrices/ld_clump_instruments.csv,
    produced by analysis/mr_ldclump/run_ld_clump.py from 1000 Genomes EUR genotypes
    (r2 < 0.001, 10 Mb window, PLINK --clump semantics). Physical distance is not a
    valid stand-in for LD, so this is the specification reported in the manuscript.

    Fallback: greedy distance pruning, strongest variant first. Retained only so the
    pipeline still runs on a machine without the LD lookup table. Ranked by |z| rather
    than by reported P because several cis-pQTLs underflow to P=0 and the resulting
    ties made the retained variant order-dependent.
    """
    m = m.assign(_z=np.abs(m["b_exp"] / m["se_exp"])).sort_values(
        ["_z", "pos"], ascending=[False, True]).drop(columns="_z")
    if gene is not None and LD_INSTR.exists():
        tab = pd.read_csv(LD_INSTR)
        sub = tab[tab["protein"] == gene]
        if len(sub):
            keep = set(sub["pos"].astype(int))
            sel = m[m["pos"].astype(int).isin(keep)]
            if len(sel):
                return sel.reset_index(drop=True)
    keep, used = [], []
    for i, r in m.iterrows():
        p = r["pos"]
        if all(abs(p - u) > kb * 1000 for u in used):
            keep.append(i); used.append(p)
    return m.loc[keep]


def thin_by_distance(m: pd.DataFrame, kb: int) -> pd.DataFrame:
    """Variant-density sensitivity for the colocalisation posterior: keep the strongest
    (largest |z|) variant in each `kb`-wide window.  Selecting within a window rather
    than taking the first variant by position guarantees that a lone strong signal is
    never discarded by the thinning step itself."""
    if kb <= 0:
        return m
    bin_id = (m["pos"] // (kb * 1000)).astype(int)
    z = np.abs(m["b_exp"] / m["se_exp"])
    keep = z.groupby(bin_id).idxmax()
    return m.loc[keep.sort_values()]


# --------------------------------------------------------------------------- MR
def abf(beta, se, w=W_PRIOR):
    """Wakefield log approximate Bayes factor (per variant)."""
    v = np.asarray(se, float) ** 2
    z = np.asarray(beta, float) / np.asarray(se, float)
    return 0.5 * (np.log(v / (v + w)) + z ** 2 * w / (v + w))


def coloc(m: pd.DataFrame, w1=W_PRIOR, w2=W_PRIOR, p1=1e-4, p2=1e-4, p12=1e-5) -> dict:
    """Single-causal-variant colocalisation.

    Aggregation is log-sum-exp (sum of Bayes factors), NOT the sum of log Bayes
    factors.  H3 uses the standard decomposition S1*S2 - S12: two distinct causal
    variants means any pair of trait-1 and trait-2 candidates except the same one.
    w1 applies to the protein (SD units), w2 to cirrhosis (log-OR); w2 is varied in
    the sensitivity analysis.
    """
    l1 = abf(m["b_exp"].to_numpy(), m["se_exp"].to_numpy(), w1)
    l2 = abf(m["b_out"].to_numpy(), m["se_out"].to_numpy(), w2)
    a = logsumexp(l1)          # log S1
    b = logsumexp(l2)          # log S2
    c = logsumexp(l1 + l2)     # log S12
    # log(S1*S2 - S12); S1*S2 >= S12 holds because all BF > 0
    tot = a + b
    diff = tot + np.log1p(-np.exp(c - tot)) if c < tot else -np.inf
    lh = np.array([0.0,
                   np.log(p1) + a,
                   np.log(p2) + b,
                   np.log(p1) + np.log(p2) + diff,
                   np.log(p12) + c])
    lh -= lh.max()
    pp = np.exp(lh); pp /= pp.sum()
    return dict(PPH0=pp[0], PPH1=pp[1], PPH2=pp[2], PPH3=pp[3], PPH4=pp[4],
                n_shared=len(m), logS1=a, logS2=b,
                max_z_exp=float(np.abs(m["b_exp"] / m["se_exp"]).max()),
                max_z_out=float(np.abs(m["b_out"] / m["se_out"]).max()),
                min_p_out=float(m["p_out"].min()))


def _sim_ratio(b_exp, b_out, se_exp, se_out):
    ratio = b_out / b_exp
    se_ratio = np.sqrt(se_out ** 2 / b_exp ** 2 + b_out ** 2 * se_exp ** 2 / b_exp ** 4)
    return ratio, 1.0 / se_ratio ** 2


def _wmed_point(ratio, w):
    order = np.argsort(ratio)
    r, ww = ratio[order], w[order]
    cw = np.cumsum(ww) / ww.sum()
    return float(r[np.searchsorted(cw, 0.5)])


def weighted_median(m: pd.DataFrame, nboot=NBOOT_WMED, seed=SEED) -> dict:
    """Bowden 2016 weighted median with a parametric-bootstrap standard error."""
    b_exp, se_exp = m["b_exp"].to_numpy(), m["se_exp"].to_numpy()
    b_out, se_out = m["b_out"].to_numpy(), m["se_out"].to_numpy()
    ratio, w = _sim_ratio(b_exp, b_out, se_exp, se_out)
    beta = _wmed_point(ratio, w)
    rng = np.random.default_rng(seed)
    boot = np.empty(nboot)
    for i in range(nboot):
        be = rng.normal(b_exp, se_exp)
        bo = rng.normal(b_out, se_out)
        r, ww = _sim_ratio(be, bo, se_exp, se_out)
        boot[i] = _wmed_point(r, ww)
    se_boot = float(np.std(boot, ddof=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # non-parametric cross-check (resample variants with replacement)
    np_boot = np.empty(nboot)
    n = len(ratio)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        np_boot[i] = _wmed_point(ratio[idx], w[idx])
    return dict(b_wmed=beta, se_wmed=se_boot,
                p_wmed=float(2 * stats.norm.sf(abs(beta / se_boot))),
                wmed_lo=float(lo), wmed_hi=float(hi),
                p_wmed_nparam=float(2 * stats.norm.sf(abs(beta / np.std(np_boot, ddof=1)))),
                nboot=nboot)


def mr(m: pd.DataFrame) -> dict:
    b_exp, b_out = m["b_exp"].to_numpy(), m["b_out"].to_numpy()
    se_exp, se_out = m["se_exp"].to_numpy(), m["se_out"].to_numpy()
    ratio = b_out / b_exp
    se_ratio = np.sqrt(se_out ** 2 / b_exp ** 2 + b_out ** 2 * se_exp ** 2 / b_exp ** 4)
    w = 1 / se_ratio ** 2
    b_ivw = (ratio * w).sum() / w.sum()
    se_ivw = np.sqrt(1 / w.sum())
    res = dict(n_snp=len(m), b_ivw=b_ivw, se_ivw=se_ivw,
               p_ivw=2 * stats.norm.sf(abs(b_ivw / se_ivw)))
    res["or_ivw"] = np.exp(b_ivw)
    res["or_lo"] = np.exp(b_ivw - 1.96 * se_ivw)
    res["or_hi"] = np.exp(b_ivw + 1.96 * se_ivw)
    # heterogeneity / random effects
    q = (w * (ratio - b_ivw) ** 2).sum()
    dfree = max(len(m) - 1, 1)
    res["Q"] = q
    res["Q_p"] = stats.chi2.sf(q, dfree) if len(m) > 1 else np.nan
    se_re = se_ivw * np.sqrt(q / dfree) if (q > dfree and len(m) > 2) else se_ivw
    res["b_ivw_re"] = b_ivw; res["se_ivw_re"] = se_re
    res["p_ivw_re"] = 2 * stats.norm.sf(abs(b_ivw / se_re))
    # instrument strength: F = (b/se)^2 per variant, R2 = sum(F)/N
    f_stat = (b_exp / se_exp) ** 2
    res["sumF"] = float(f_stat.sum())
    n_exp = float(m["n"].iloc[0]) if "n" in m.columns and pd.notna(m["n"].iloc[0]) else np.nan
    res["r2_pct"] = 100 * res["sumF"] / n_exp if np.isfinite(n_exp) else np.nan
    # 80%-power minimum detectable OR for the observed IVW standard error
    res["mde_or80"] = float(np.exp((stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se_ivw))
    # MR-Egger + weighted median need >= 3 variants
    if len(m) >= 3:
        X = np.column_stack([np.ones(len(b_exp)), b_exp])
        Wm = np.diag(1 / se_out ** 2)
        beta_eg = np.linalg.solve(X.T @ Wm @ X, X.T @ Wm @ b_out)
        resid = b_out - X @ beta_eg
        s2 = (resid ** 2 * np.diag(Wm)).sum() / (len(m) - 2)
        cov = s2 * np.linalg.inv(X.T @ Wm @ X)
        res["b_egger"] = beta_eg[1]; res["se_egger"] = np.sqrt(cov[1, 1])
        res["p_egger"] = 2 * stats.norm.sf(abs(beta_eg[1] / np.sqrt(cov[1, 1])))
        res["egger_intercept"] = beta_eg[0]
        res["p_egger_intercept"] = 2 * stats.t.sf(abs(beta_eg[0] / np.sqrt(cov[0, 0])), len(m) - 2)
        res.update(weighted_median(m))
        # leave-one-out
        loo = []
        for i in range(len(m)):
            k = np.ones(len(m), bool); k[i] = False
            wi = 1 / se_ratio[k] ** 2
            loo.append((ratio[k] * wi).sum() / wi.sum())
        res["loo_min"] = float(min(loo)); res["loo_max"] = float(max(loo))
        res["loo_sign_stable"] = bool(all(np.sign(np.array(loo)) == np.sign(b_ivw)))
    return res


# ---------------------------------------------------------------------- driver
rows_mr, rows_inst, rows_col, rows_sens = [], [], [], []
for g, alias in GENES.items():
    fe, fo = RAW / f"cis_{g}.tsv.gz", RAW / f"out2_{g}.tsv.gz"
    if not fo.exists():
        fo = RAW / f"out_cirrhosis_{g}.tsv.gz"
    if not fe.exists() or not fo.exists():
        print("missing", g); continue
    exp = pd.read_csv(fe, sep="\t")
    out = pd.read_csv(fo, sep="\t")
    both = harmonise(exp, out)
    if len(both) == 0:
        rows_mr.append(dict(gene=g, protein=alias, n_snp=0, note="no shared SNPs"))
        rows_col.append(dict(gene=g, protein=alias, PPH4=np.nan, n_shared=0))
        continue
    rows_col.append(dict(gene=g, protein=alias, core=g in CORE, **coloc(both)))
    # prior / density sensitivity around the primary specification
    for kb in (0, 10, 5):
        sub = thin_by_distance(both, kb)
        for w2 in (0.05 ** 2, W_PRIOR, 0.30 ** 2):
            for p12 in (1e-6, 1e-5, 1e-4):
                r = coloc(sub, w2=w2, p12=p12)
                rows_sens.append(dict(gene=g, protein=alias, thin_kb=kb, w_out=np.sqrt(w2),
                                      p12=p12, PPH1=r["PPH1"], PPH4=r["PPH4"],
                                      n_shared=r["n_shared"]))
    inst = clump(both[both["p_exp"] < P_THR], gene=g)
    if len(inst) == 0:
        inst = clump(both[both["p_exp"] < 1e-5], gene=g)
        note = "relaxed 1e-5"
    else:
        note = "cis p<5e-8"
    if len(inst) == 0:
        rows_mr.append(dict(gene=g, protein=alias, core=g in CORE, n_snp=0, note="no instrument"))
        continue
    r = mr(inst)
    r.update(dict(gene=g, protein=alias, core=g in CORE, note=note))
    rows_mr.append(r)
    for _, x in inst.iterrows():
        rows_inst.append(dict(gene=g, protein=alias, rsid=x["rsid"], pos=int(x["pos"]),
                              ea=x["ea_exp"], oa=x["oa_exp"], eaf=x["eaf_exp"],
                              beta_exp=x["b_exp"], se_exp=x["se_exp"], p_exp=x["p_exp"],
                              beta_out=x["b_out"], se_out=x["se_out"], p_out=x["p_out"]))
    print(g, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

mrd = pd.DataFrame(rows_mr)
mrd.to_csv(MAT / "mr_results.csv", index=False)
pd.DataFrame(rows_inst).to_csv(MAT / "mr_instruments.csv", index=False)
pd.DataFrame(rows_col).to_csv(MAT / "coloc_results.csv", index=False)
pd.DataFrame(rows_sens).to_csv(MAT / "coloc_sensitivity.csv", index=False)

cols = ["protein", "n_snp", "sumF", "r2_pct", "or_ivw", "or_lo", "or_hi", "p_ivw",
        "mde_or80", "Q", "Q_p", "p_egger", "p_egger_intercept",
        "b_wmed", "p_wmed", "p_wmed_nparam", "loo_min", "loo_max", "loo_sign_stable"]
diag = mrd.reindex(columns=cols)
diag.to_csv(MAT / "mr_diagnostics.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== MR ===")
print(mrd[cols].round(4).to_string(index=False))
print("\n=== COLOC ===")
print(pd.DataFrame(rows_col).round(3).to_string(index=False))
