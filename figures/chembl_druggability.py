# -*- coding: utf-8 -*-
"""Fig 6 C: druggability annotation of panel proteins via ChEMBL (public REST).
Output: matrices/druggability.csv, matrices/drug_list.csv
"""
from pathlib import Path
import json, time, urllib.parse, urllib.request
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
MAT = BASE / "figures" / "fig6" / "matrices"
MAT.mkdir(parents=True, exist_ok=True)
API = "https://www.ebi.ac.uk/chembl/api/data"

GENES = {"TNFRSF1A": "TNF-RI", "REN": "Renin", "TNFRSF21": "DR6", "SCARF2": "SREC-II",
         "ANGPT2": "ANG-2", "CCL14": "HCC-1", "CRP": "CRP"}
# strict target verification: ChEMBL free-text search is fuzzy, so require the expected protein name
EXPECT = {"TNFRSF1A": "receptor superfamily member 1A", "REN": "renin",
          "TNFRSF21": "receptor superfamily member 21", "SCARF2": "scavenger receptor class F member 2",
          "ANGPT2": "angiopoietin-2", "CCL14": "c-c motif chemokine 14", "CRP": "c-reactive protein"}
SEARCH_TERMS = {"TNFRSF1A": ["TNFRSF1A", "tumor necrosis factor receptor 1A"],
                "REN": ["REN", "renin"], "TNFRSF21": ["TNFRSF21", "DR6", "death receptor 6"],
                "SCARF2": ["SCARF2", "scavenger receptor class F member 2"],
                "ANGPT2": ["ANGPT2", "angiopoietin-2"],
                "CCL14": ["CCL14", "chemokine 14"], "CRP": ["CRP", "C-reactive protein"]}

def get(url, tries=4, timeout=90):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            if t == tries - 1:
                print("fail", url, e)
                return None
            time.sleep(2)

def target_for(gene):
    want = EXPECT[gene].lower()
    for term in SEARCH_TERMS[gene]:
        d = get(f"{API}/target/search.json?q={urllib.parse.quote(term)}&limit=100")
        for t in (d or {}).get("targets", []):
            if t.get("organism") != "Homo sapiens" or t.get("target_type") != "SINGLE PROTEIN":
                continue
            if want in (t.get("pref_name") or "").lower():
                return t
    return None

rows, drugs = [], []
for g, alias in GENES.items():
    t = target_for(g)
    if not t:
        rows.append(dict(gene=g, protein=alias, target="", n_mechanisms=0, max_phase=None,
                         n_approved=0, action_types=""))
        continue
    tid = t["target_chembl_id"]
    d = get(f"{API}/mechanism.json?target_chembl_id={tid}&limit=200")
    mechs = (d or {}).get("mechanisms", [])
    mols = sorted({m["molecule_chembl_id"] for m in mechs})
    mol_info = {}
    if mols:
        for chunk in [mols[i:i + 40] for i in range(0, len(mols), 40)]:
            q = ";".join(chunk)
            md = get(f"{API}/molecule.json?molecule_chembl_id__in={q}&limit=100")
            for mo in (md or {}).get("molecules", []):
                mol_info[mo["molecule_chembl_id"]] = dict(
                    name=mo.get("pref_name") or mo["molecule_chembl_id"],
                    phase=mo.get("max_phase") or 0,
                    first_approval=mo.get("first_approval"),
                    oral=mo.get("oral"),
                    type=mo.get("molecule_type"))
    def _ph(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    phases = [_ph(mol_info.get(m, {}).get("phase", 0)) for m in mols]
    acts = sorted({m.get("action_type") or "" for m in mechs})
    rows.append(dict(gene=g, protein=alias, target=f"{tid} ({t['pref_name']})",
                     n_mechanisms=len(mechs), n_drugs=len(mols),
                     max_phase=float(max(phases)) if phases else 0.0,
                     n_approved=sum(1 for p in phases if p >= 4),
                     action_types=";".join(a for a in acts if a)))
    for m in mols:
        info = mol_info.get(m, {})
        drugs.append(dict(gene=g, protein=alias, molecule=m, name=info.get("name", ""),
                          max_phase=info.get("phase", 0),
                          first_approval=info.get("first_approval"),
                          drug_type=info.get("type", "")))

dr = pd.DataFrame(rows)
dr.to_csv(MAT / "druggability.csv", index=False)
pd.DataFrame(drugs).to_csv(MAT / "drug_list.csv", index=False)
print(dr.to_string(index=False))
print("n drugs total", len(drugs))
