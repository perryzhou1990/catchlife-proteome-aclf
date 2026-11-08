# -*- coding: utf-8 -*-
"""Extract the same cis windows from the dense cirrhosis GWAS (GCST90319878, 22.7M SNPs)
via remote tabix. Output: raw_data/out2_<GENE>.tsv.gz
"""
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import bgzf_tabix as bt

BASE = Path(__file__).resolve().parents[3]
FIG6 = BASE / "figures" / "fig6"
RAW, MAT = FIG6 / "raw_data", FIG6 / "matrices"
CACHE = RAW / "tbi_cache"
URL = ("https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
       "GCST90319001-GCST90320000/GCST90319878/harmonised/GCST90319878.h.tsv.gz")

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

def run(gene):
    g = GENES[gene]
    t0 = time.time()
    try:
        tb = bt.RemoteTabix(URL, cache_dir=str(CACHE))
        rows = tb.fetch(g["chrom"], g["start"] - FLANK, g["end"] + FLANK)
        if not rows:
            return dict(gene=gene, n_var=0, status="no rows", secs=round(time.time() - t0))
        cols = ["chromosome", "pos", "effect_allele", "other_allele", "beta", "se", "eaf",
                "p", "variant_id", "rsid"]
        df = pd.DataFrame([r[:len(cols)] for r in rows], columns=cols[:len(rows[0])])
        for c in ["pos", "beta", "se", "eaf", "p"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["pos", "beta", "se", "p"])
        df.to_csv(RAW / f"out2_{gene}.tsv.gz", sep="\t", index=False, compression="gzip")
        return dict(gene=gene, n_var=len(df), n_sig=int((df["p"] < 5e-8).sum()),
                    status="ok", secs=round(time.time() - t0))
    except Exception as e:
        return dict(gene=gene, n_var=0, status=f"ERR {e}", secs=round(time.time() - t0))

if __name__ == "__main__":
    res = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run, g): g for g in GENES}
        for f in as_completed(futs):
            r = f.result(); print(r, flush=True); res.append(r)
    pd.DataFrame(res).to_csv(MAT / "outcome2_extraction_summary.csv", index=False)
    print("DONE")
