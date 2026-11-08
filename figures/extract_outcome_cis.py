# -*- coding: utf-8 -*-
"""Slice the same cis windows (+/-250 kb) from the local cirrhosis GWAS harmonised sumstats.
Writes fig6/raw_data/out_cirrhosis_<GENE>.tsv.gz
"""
from pathlib import Path
import gzip
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
RAW = BASE / "figures" / "fig6" / "raw_data"
CIRR = RAW / "cirrhosis_GCST90081520.h.tsv.gz"

GENES = {
    "TNFRSF1A": dict(alias="TNF-RI", chrom="12", start=6_328_757, end=6_342_114),
    "REN":      dict(alias="Renin",  chrom="1",  start=204_154_819, end=204_169_907),
    "TNFRSF21": dict(alias="DR6",    chrom="6",  start=47_233_230, end=47_310_575),
    "SCARF2":   dict(alias="SREC-II", chrom="22", start=20_766_631, end=20_783_807),
    "ANGPT2":   dict(alias="ANG-2",  chrom="8",  start=6_494_000, end=6_540_000),
    "CCL14":    dict(alias="HCC-1",  chrom="17", start=35_983_000, end=36_005_000),
    "CRP":      dict(alias="CRP",    chrom="1",  start=159_712_000, end=159_715_000),
}
FLANK = 250_000
win = {g: (str(v["chrom"]), v["start"] - FLANK, v["end"] + FLANK) for g, v in GENES.items()}

USECOLS = ["hm_rsid", "hm_chrom", "hm_pos", "hm_effect_allele", "hm_other_allele",
           "hm_odds_ratio", "hm_ci_lower", "hm_ci_upper", "hm_effect_allele_frequency",
           "p_value", "standard_error"]

buf = {g: [] for g in GENES}
n_read = 0
with gzip.open(CIRR, "rt", encoding="utf-8") as f:
    head = f.readline().rstrip("\n").split("\t")
    idx = {c: head.index(c) for c in USECOLS if c in head}
    for line in f:
        n_read += 1
        p = line.rstrip("\n").split("\t")
        c = p[idx["hm_chrom"]]
        for g, (gc, lo, hi) in win.items():
            if c != gc:
                continue
            try:
                pos = int(p[idx["hm_pos"]])
            except (ValueError, KeyError):
                continue
            if lo <= pos <= hi:
                buf[g].append([p[idx[k]] if k in idx else "NA" for k in USECOLS])
print("lines read", n_read)
rows_summary = []
for g, rows in buf.items():
    df = pd.DataFrame(rows, columns=USECOLS)
    for c in ["hm_pos", "hm_odds_ratio", "hm_ci_lower", "hm_ci_upper",
              "hm_effect_allele_frequency", "p_value", "standard_error"]:
        df[c] = pd.to_numeric(df[c].replace("NA", "nan"), errors="coerce")
    df = df.rename(columns={"hm_rsid": "rsid", "hm_chrom": "chrom", "hm_pos": "pos",
                            "hm_effect_allele": "effect_allele", "hm_other_allele": "other_allele",
                            "hm_odds_ratio": "odds_ratio", "hm_effect_allele_frequency": "eaf"})
    df.to_csv(RAW / f"out_cirrhosis_{g}.tsv.gz", sep="\t", index=False, compression="gzip")
    rows_summary.append(dict(gene=g, n_var=len(df),
                             n_sig=int((df["p_value"] < 5e-8).sum())))
    print(g, len(df), "sig", int((df["p_value"] < 5e-8).sum()))
pd.DataFrame(rows_summary).to_csv(BASE / "figures" / "fig6" / "matrices" / "outcome_cis_summary.csv", index=False)
print("DONE")
