# -*- coding: utf-8 -*-
"""Find pQTL summary-statistics studies on GWAS Catalog for panel genes.
Writes fig6/matrices/pqtl_candidates.csv
"""
from pathlib import Path
import json, time, urllib.parse, urllib.request
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
OUT = BASE / "figures" / "fig6" / "matrices"
OUT.mkdir(parents=True, exist_ok=True)

GENES = ["TNFRSF1A", "REN", "TNFRSF21", "SCARF2", "ANGPT2", "CCL14", "CRP"]
API = "https://www.ebi.ac.uk/gwas/rest/api/v2/studies"

def query(term, size=100):
    url = f"{API}?reportedTrait={urllib.parse.quote(term)}&size={size}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                d = json.load(r)
            st = d.get("_embedded", {}).get("studies", [])
            return st
        except Exception as e:
            print("retry", term, attempt, e)
            time.sleep(5)
    return []

rows = []
for g in GENES:
    st = query(g)
    print(f"{g}: {len(st)} studies")
    for s in st:
        rows.append(dict(gene=g, accession=s.get("accession_id"),
                         trait=s.get("disease_trait"),
                         efo=";".join(x.get("efo_trait", "") for x in s.get("efo_traits", [])),
                         n=s.get("initial_sample_size"),
                         snps=s.get("snp_count"),
                         sumstats=s.get("full_summary_stats_available"),
                         pmid=s.get("pubmed_id"),
                         ancestry=";".join(s.get("discovery_ancestry", []) or [])))
    time.sleep(2)

df = pd.DataFrame(rows).drop_duplicates(subset=["gene", "accession"])
df.to_csv(OUT / "pqtl_candidates_raw.csv", index=False, encoding="utf-8-sig")
# keep studies that are protein-level measurements with full sumstats
mask = df["trait"].astype(str).str.contains("rotein|pQTL", case=False, na=False) & (df["sumstats"] == True)
sel = df[mask].copy()
sel.to_csv(OUT / "pqtl_candidates.csv", index=False, encoding="utf-8-sig")
print(sel.to_string(index=False))
