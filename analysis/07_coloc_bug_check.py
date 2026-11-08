# -*- coding: utf-8 -*-
"""Quick numeric check: does the ABF aggregation bug explain PP.H0=1 for TNFRSF1A?"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.special import logsumexp

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "figures" / "fig6" / "raw_data"

sys.path.insert(0, str(BASE / "figures" / "fig6" / "code"))
import importlib.util
spec = importlib.util.spec_from_file_location("m", BASE / "figures" / "fig6" / "code" / "mr_coloc.py")

# reproduce harmonise() without running the whole script
GENES = ["TNFRSF1A", "REN", "TNFRSF21", "SCARF2", "ANGPT2", "CCL14", "CRP"]
W_PRIOR = 0.15 ** 2

def prepare_outcome(out):
    o = out.copy()
    if "odds_ratio" in o.columns:
        o["b_out"] = np.log(pd.to_numeric(o["odds_ratio"], errors="coerce"))
        o["se_out"] = pd.to_numeric(o["standard_error"], errors="coerce")
        o["p_out"] = pd.to_numeric(o["p_value"], errors="coerce")
        o["eaf_out"] = pd.to_numeric(o["eaf"], errors="coerce")
    else:
        o["b_out"] = pd.to_numeric(o["beta"], errors="coerce")
        o["se_out"] = pd.to_numeric(o["se"], errors="coerce")
        o["p_out"] = pd.to_numeric(o["p"], errors="coerce")
        o["eaf_out"] = pd.to_numeric(o["eaf"], errors="coerce")
    o["ea_out"] = o["effect_allele"]; o["oa_out"] = o["other_allele"]
    return o[["rsid", "b_out", "se_out", "ea_out", "oa_out", "eaf_out", "p_out", "pos"]]

def harmonise(exp, out):
    e = exp.rename(columns={"beta": "b_exp", "se": "se_exp", "effect_allele": "ea_exp",
                            "other_allele": "oa_exp", "eaf": "eaf_exp", "p": "p_exp"})
    o = prepare_outcome(out)
    m = e.merge(o.drop(columns=["rsid"]), on="pos", how="inner", suffixes=("", "_o"))
    m = m.dropna(subset=["b_exp", "se_exp", "b_out", "se_out"])
    same = (m["ea_exp"].str.upper() == m["ea_out"].str.upper()) & \
           (m["oa_exp"].str.upper() == m["oa_out"].str.upper())
    flip = (m["ea_exp"].str.upper() == m["oa_out"].str.upper()) & \
           (m["oa_exp"].str.upper() == m["ea_out"].str.upper())
    m = m[same | flip].copy()
    m["b_out"] = np.where(flip[m.index], -m["b_out"], m["b_out"])
    pal = m.apply(lambda r: {r["ea_exp"].upper(), r["oa_exp"].upper()} in ({"A", "T"}, {"C", "G"}), axis=1)
    return m[~(pal & m["eaf_exp"].between(0.42, 0.58))]

def abf(b, se, w=W_PRIOR):
    v = se ** 2
    return 0.5 * (np.log(v / (v + w)) + (b / se) ** 2 * w / (v + w))

for g in GENES:
    exp = pd.read_csv(RAW / f"cis_{g}.tsv.gz", sep="\t")
    out = pd.read_csv(RAW / f"out2_{g}.tsv.gz", sep="\t")
    m = harmonise(exp, out)
    l1 = abf(m["b_exp"].to_numpy(), m["se_exp"].to_numpy())
    l2 = abf(m["b_out"].to_numpy(), m["se_out"].to_numpy())
    print(f"\n{g}: n={len(m)}  max|z_exp|={np.abs(m.b_exp/m.se_exp).max():.1f}  "
          f"max|z_out|={np.abs(m.b_out/m.se_out).max():.2f}")
    print(f"  OLD (sum of logs):  s1={l1.sum():9.1f}   s2={l2.sum():9.1f}")
    print(f"  NEW (log-sum-exp):  s1={logsumexp(l1):9.1f}   s2={logsumexp(l2):9.1f}")
    print(f"  per-SNP logABF exp side: median={np.median(l1):.2f}  "
          f">0: {(l1>0).sum()}  outcome side median={np.median(l2):.2f}")
