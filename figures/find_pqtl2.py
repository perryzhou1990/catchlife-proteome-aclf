# -*- coding: utf-8 -*-
"""Locate pQTL summary-statistics studies (trait naming: '<GENE> protein levels') and
resolve their FTP file paths. Writes fig6/matrices/pqtl_studies.csv
"""
from pathlib import Path
import json, time, urllib.parse, urllib.request, subprocess
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
OUT = BASE / "figures" / "fig6" / "matrices"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://www.ebi.ac.uk/gwas/rest/api/v2/studies"
FTP_ROOT = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"

GENES = {"TNFRSF1A": "TNF-RI", "REN": "Renin", "TNFRSF21": "DR6", "SCARF2": "SREC-II",
         "ANGPT2": "ANG-2", "CCL14": "HCC-1", "CRP": "CRP"}

def q(term, size=25):
    url = f"{API}?diseaseTrait={urllib.parse.quote(term)}&size={size}"
    for a in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r).get("_embedded", {}).get("studies", [])
        except Exception as e:
            print("  retry", term, a, e); time.sleep(6)
    return []

def ftp_dir(accession):
    num = int(accession.replace("GCST", ""))
    start = (num - 1) // 1000 * 1000 + 1
    return f"{FTP_ROOT}/GCST{start}-GCST{start+999}/{accession}"

def ftp_list(path):
    try:
        out = subprocess.run(["curl", "-s", "-m", "90", "--user", "anonymous:a@b.c", path],
                             capture_output=True, timeout=120).stdout.decode("utf-8", "ignore")
        return out
    except Exception:
        return ""

rows = []
for g, alias in GENES.items():
    hits = []
    for term in [f"{g} protein levels", f"{g} levels"]:
        for s in q(term):
            if str(s.get("disease_trait", "")).lower().startswith(g.lower()):
                hits.append(s)
        time.sleep(1)
    seen = set()
    for s in hits:
        if s["accession_id"] in seen:
            continue
        seen.add(s["accession_id"])
        rows.append(dict(gene=g, protein=alias, accession=s["accession_id"],
                         trait=s.get("disease_trait"), n=s.get("initial_sample_size"),
                         snps=s.get("snp_count"), sumstats=s.get("full_summary_stats_available"),
                         pmid=s.get("pubmed_id"), ancestry=";".join(s.get("discovery_ancestry") or []),
                         ftp_dir=ftp_dir(s["accession_id"])))
    print(g, "->", [(r["accession"], r["trait"], str(r["n"])[:30]) for r in rows if r["gene"] == g])

df = pd.DataFrame(rows)
df.to_csv(OUT / "pqtl_studies.csv", index=False, encoding="utf-8-sig")
print(df[["gene", "accession", "trait", "n", "sumstats"]].to_string(index=False))
