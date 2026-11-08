# -*- coding: utf-8 -*-
"""Extract cis windows (+/- 250 kb) from UKB-PPP pQTL harmonised sumstats via remote tabix.
Writes fig6/raw_data/cis_<GENE>.tsv.gz  and fig6/matrices/cis_extraction_summary.csv
"""
from pathlib import Path
import gzip, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import bgzf_tabix as bt

BASE = Path(__file__).resolve().parents[3]
FIG6 = BASE / "figures" / "fig6"
RAW, MAT = FIG6 / "raw_data", FIG6 / "matrices"
RAW.mkdir(parents=True, exist_ok=True); MAT.mkdir(parents=True, exist_ok=True)
CACHE = FIG6 / "raw_data" / "tbi_cache"
CACHE.mkdir(parents=True, exist_ok=True)
FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"

# GRCh38 coordinates (gene body) and the pQTL study accession (n=47,745 European, UKB-PPP deposit)
GENES = {
    "TNFRSF1A": dict(alias="TNF-RI", chrom="12", start=6_328_757, end=6_342_114, acc="GCST90470910"),
    "REN":      dict(alias="Renin",  chrom="1",  start=204_154_819, end=204_169_907, acc="GCST90470454"),
    "TNFRSF21": dict(alias="DR6",    chrom="6",  start=47_233_230, end=47_310_575, acc="GCST90470912"),
    "SCARF2":   dict(alias="SREC-II", chrom="22", start=20_766_631, end=20_783_807, acc="GCST90470537"),
    "ANGPT2":   dict(alias="ANG-2",  chrom="8",  start=6_494_000, end=6_540_000, acc="GCST90468301"),
    "CCL14":    dict(alias="HCC-1",  chrom="17", start=35_983_000, end=36_005_000, acc="GCST90468566"),
    "CRP":      dict(alias="CRP",    chrom="1",  start=159_712_000, end=159_715_000, acc="GCST90453081"),
}
FLANK = 250_000

def ftp_url(acc):
    n = int(acc.replace("GCST", ""))
    s = (n - 1) // 1000 * 1000 + 1
    return f"{FTP}/GCST{s}-GCST{s+999}/{acc}/harmonised/{acc}.h.tsv.gz"

def run(gene):
    g = GENES[gene]
    url = ftp_url(g["acc"])
    out = RAW / f"cis_{gene}.tsv.gz"
    t0 = time.time()
    try:
        tb = bt.RemoteTabix(url, cache_dir=str(CACHE))
        rows = tb.fetch(g["chrom"], g["start"] - FLANK, g["end"] + FLANK)
        if not rows:
            return dict(gene=gene, alias=g["alias"], acc=g["acc"], n_var=0,
                        status="no rows", secs=round(time.time() - t0))
        cols = ["chromosome", "pos", "effect_allele", "other_allele", "beta", "se", "eaf",
                "p", "variant_id", "rsid", "rs_id", "n", "hm_conv", "hm_code"]
        df = pd.DataFrame([r[:len(cols)] for r in rows], columns=cols[:len(rows[0])])
        num = ["pos", "beta", "se", "eaf", "p", "n"]
        for c in num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["pos", "beta", "se", "p"])
        df.to_csv(out, sep="\t", index=False, compression="gzip")
        sig = df[df["p"] < 5e-8]
        top = df.loc[df["p"].idxmin()] if len(df) else None
        return dict(gene=gene, alias=g["alias"], acc=g["acc"], n_var=len(df), n_sig=len(sig),
                    top_snp=top["rsid"] if top is not None else "",
                    top_pos=int(top["pos"]) if top is not None else None,
                    top_beta=round(float(top["beta"]), 4) if top is not None else None,
                    top_se=round(float(top["se"]), 4) if top is not None else None,
                    top_p=float(top["p"]) if top is not None else None,
                    status="ok", secs=round(time.time() - t0))
    except Exception as e:
        return dict(gene=gene, alias=g["alias"], acc=g["acc"], n_var=0, status=f"ERR {e}",
                    secs=round(time.time() - t0))

if __name__ == "__main__":
    res = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run, g): g for g in GENES}
        for f in as_completed(futs):
            r = f.result()
            print(r, flush=True)
            res.append(r)
    pd.DataFrame(res).to_csv(MAT / "cis_extraction_summary.csv", index=False)
    print("DONE")
