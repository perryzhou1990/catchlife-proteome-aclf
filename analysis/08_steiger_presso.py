# -*- coding: utf-8 -*-
"""Steiger directionality test + MR-PRESSO global/outlier test (v4 addendum to audit 1.x).

Closes the two remaining "cannot compute" items in 06_MR方法学审计_v1.md where possible:

1. Steiger directionality (TwoSampleMR `directionality_test` equivalent, approximate):
   - Exposure side: per-SNP N is available in the cis summary stats (UKB-PPP, n=47,745;
     CRP n=1,909), r_exp = sign(b) * sqrt(chi2 / (chi2 + N - 2)).
   - Outcome side: the dense cirrhosis summary stats carry NO per-SNP N, so we use the
     study-level effective sample size for a case-control GWAS,
         N_eff = 4 * n0 * n1 / (n0 + n1)  (n1=18,265, n0=1,782,047 -> 72,333),
     and r_out = sign(b) * sqrt(chi2 / (chi2 + N_eff)).  Constant-N approximation,
     disclosed in the manuscript.  Logistic betas also make this approximate.
   - Direction: forward if sum(r2_exp) > sum(r2_out) across instruments.

2. MR-PRESSO (Verbanck et al. 2018), our implementation of the published algorithm:
   - Global test: J(theta) = sum_i (b_out_i - theta * b_exp_i)^2 / se_out_i^2;
     p-value from a parametric bootstrap under the IVW null (10,000 sims):
     b_exp* ~ N(b_exp, se_exp), b_out* ~ N(theta_ivw * b_exp*, se_out).
   - Outlier test only if global P < 0.05: leave-one-instrument-out, corrected
     estimate = IVW on the retained set.
   - Needs >= 3 instruments; TNF-RI, SREC-II and CRP have 1-2 and are not tested.

Input : figures/fig6/matrices/mr_instruments.csv (18 instruments, harmonised)
Output: figures/fig6/matrices/steiger_presso.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG6 = ROOT / "figures" / "fig6"
RAW, MAT = FIG6 / "raw_data", FIG6 / "matrices"

N_CASE, N_CTRL = 18_265, 1_782_047          # Ghouse et al., Nat Genet 2024 (cross-ancestry)
N_EFF = 4.0 * N_CASE * N_CTRL / (N_CASE + N_CTRL)
RNG = np.random.default_rng(20261107)
N_SIM = 10_000

GENE_OF = {"TNF-RI": "TNFRSF1A", "Renin": "REN", "DR6": "TNFRSF21", "SREC-II": "SCARF2",
           "ANG-2": "ANGPT2", "HCC-1": "CCL14", "CRP": "CRP"}


def r_from_pn(b, se, n):
    """Approximate correlation between SNP and trait (quantitative side)."""
    z2 = (np.asarray(b) / np.asarray(se)) ** 2
    r = np.sign(b) * np.sqrt(z2 / (z2 + np.asarray(n, dtype=float) - 2.0))
    return r


def steiger(rows):
    """TwoSampleMR-style directionality for one protein. rows: exposure n per SNP."""
    r2_exp = r_from_pn(rows.beta_exp, rows.se_exp, rows.n_exp) ** 2
    r2_out = r_from_pn(rows.beta_out, rows.se_out, N_EFF) ** 2
    f_exp = r2_exp * (rows.n_exp - 2) / (1 - r2_exp)
    f_out = r2_out * (N_EFF - 2) / (1 - r2_out)
    r2e, r2o = r2_exp.sum(), r2_out.sum()
    return dict(n_snp=len(rows),
                r2_exp_sum=r2e, r2_out_sum=r2o,
                F_exp_sum=f_exp.sum(), F_out_sum=f_out.sum(),
                direction="forward" if r2e > r2o else "reverse",
                correct_direction=r2e > r2o)


def presso(rows):
    """MR-PRESSO global (+ outlier if significant) for one protein."""
    b_x = rows.beta_exp.to_numpy(float)
    s_x = rows.se_exp.to_numpy(float)
    b_y = rows.beta_out.to_numpy(float)
    s_y = rows.se_out.to_numpy(float)
    k = len(b_x)
    if k < 3:
        return dict(n_snp=k, presso_global_J=np.nan, presso_global_p=np.nan,
                    presso_note="needs >=3 instruments")
    # observed statistic at the IVW estimate
    theta = float((b_x * b_y / s_y**2).sum() / (b_x**2 / s_y**2).sum())
    J_obs = float(((b_y - theta * b_x) ** 2 / s_y**2).sum())
    # parametric bootstrap under the IVW null
    J_null = np.empty(N_SIM)
    for i in range(N_SIM):
        bx = RNG.normal(b_x, s_x)
        by = RNG.normal(theta * bx, s_y)
        w = 1.0 / s_y**2
        th = float((bx * by * w).sum() / (bx**2 * w).sum())
        J_null[i] = float(((by - th * bx) ** 2 * w).sum())
    p_global = float((J_null >= J_obs).mean())
    out = dict(n_snp=k, presso_global_J=J_obs, presso_global_p=p_global,
               presso_note="")
    if p_global < 0.05:                      # outlier search
        best = None
        for j in range(k):
            keep = np.ones(k, bool); keep[j] = False
            bx, by, sy2 = b_x[keep], b_y[keep], s_y[keep] ** 2
            th = float((bx * by / sy2).sum() / (bx**2 / sy2).sum())
            rss = float(((by - th * bx) ** 2 / sy2).sum())
            if best is None or rss < best[1]:
                best = (j, rss, th)
        j, _, th_c = best
        keep = np.ones(k, bool); keep[j] = False
        bx, by, sy2 = b_x[keep], b_y[keep], s_y[keep] ** 2
        w = 1.0 / sy2
        out.update(presso_outlier=int(rows.index[j]),
                   presso_theta_corrected=th_c,
                   presso_se_corrected=float(1.0 / np.sqrt((b_x[keep]**2 * w).sum())),
                   presso_note=f"global P<0.05; instrument #{j} dropped")
    return out


def main():
    inst = pd.read_csv(MAT / "mr_instruments.csv")
    res = []
    for prot, grp in inst.groupby("protein", sort=False):
        gene = GENE_OF[prot]
        cis = pd.read_csv(RAW / f"cis_{gene}.tsv.gz", sep="\t")
        m = grp.merge(cis[["pos", "n"]], on="pos", how="left", suffixes=("", "_cis"))
        assert m.n.notna().all(), f"exposure N join failed for {prot}"
        m = m.rename(columns={"n": "n_exp"})
        row = {"protein": prot, "gene": gene}
        row.update(steiger(m))
        row.update(presso(m))
        res.append(row)
        print(f"{prot:8s} k={row['n_snp']:2d}  R2exp={row['r2_exp_sum']:.4f} "
              f"R2out={row['r2_out_sum']:.6f}  Fexp={row['F_exp_sum']:9.1f} "
              f"Fout={row['F_out_sum']:7.1f}  -> {row['direction']}"
              + (f"  | PRESSO p={row['presso_global_p']:.3f}" if not np.isnan(row.get('presso_global_p', np.nan)) else f"  | {row['presso_note']}"))

    df = pd.DataFrame(res)
    out = MAT / "steiger_presso.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
