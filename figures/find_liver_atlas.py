# -*- coding: utf-8 -*-
"""Find public human liver scRNA-seq datasets on CELLxGENE (assets + cell counts).
Writes fig6/matrices/scrna_candidates.csv
"""
from pathlib import Path
import json, urllib.request
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
OUT = BASE / "figures" / "fig6" / "matrices"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.cellxgene.cziscience.com/curation/v1"

def get(url, timeout=180):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

LIVER_TERMS = {"liver", "hepatic", "right lobe of liver", "left lobe of liver", "caudate lobe"}

def is_liver(ds):
    for t in ds.get("tissue") or []:
        lab = (t.get("label") or "").lower()
        if lab in LIVER_TERMS or "liver" in lab or "hepat" in lab:
            return True
    return False

def human(ds):
    return any((o.get("label") == "Homo sapiens") for o in (ds.get("organism") or []))

# 1) scan all collection summaries for liver datasets
coll_hits = {}
page = 1
while page <= 8:
    try:
        cols = get(f"{API}/collections?page={page}&page_size=100")
    except Exception as e:
        print("page", page, "error", e); break
    if not cols:
        break
    for c in cols:
        for ds in c.get("datasets") or []:
            if human(ds) and is_liver(ds):
                coll_hits.setdefault(c["collection_id"], set()).add(ds["dataset_id"])
    print(f"page {page}: liver collections so far {len(coll_hits)}")
    page += 1

# 2) fetch details for each hit collection
rows = []
for cid, dset_ids in coll_hits.items():
    try:
        c = get(f"{API}/collections/{cid}")
    except Exception as e:
        print("detail failed", cid, e); continue
    for d in c.get("datasets") or []:
        organisms = [o.get("label") for o in (d.get("organism") or [])]
        if "Homo sapiens" not in organisms:
            continue
        h5ad = [a for a in (d.get("assets") or []) if a.get("filetype") == "H5AD"]
        rows.append(dict(collection=(c.get("name") or "")[:60], doi=c.get("doi") or "",
                         dataset_id=d.get("dataset_id"),
                         title=(d.get("title") or "")[:100],
                         cells=d.get("cell_count") or 0,
                         tissue=";".join(t.get("label", "") for t in (d.get("tissue") or [])),
                         disease=";".join(t.get("label", "") for t in (d.get("disease") or [])),
                         assay=";".join(t.get("label", "") for t in (d.get("assay") or [])),
                         h5ad=h5ad[0]["url"] if h5ad else "",
                         h5ad_bytes=h5ad[0].get("file_size") if h5ad else 0))

df = pd.DataFrame(rows).drop_duplicates(subset=["dataset_id"]).sort_values("cells", ascending=False)
df.to_csv(OUT / "scrna_candidates.csv", index=False, encoding="utf-8-sig")
print(f"total datasets: {len(df)}")
print(df.head(20)[["title", "cells", "disease", "h5ad_bytes"]].to_string(index=False))
