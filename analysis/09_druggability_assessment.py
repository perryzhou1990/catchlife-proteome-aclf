# -*- coding: utf-8 -*-
"""09_druggability_assessment.py

Multi-source, target-level druggability assessment of the 5-protein panel
(plus the three exploratory comparators), replacing the ChEMBL-only count
in figures/fig6/code/chembl_druggability.py.

Sources (all public REST/GraphQL, queried 2026-11-07):
  ChEMBL 35          target-centric mechanisms, named agents, max phase, first approval
  Open Targets 25.x  target class, SM/AB tractability buckets, clinical candidates,
                     safety liabilities, cirrhosis (MONDO_0005155) and ACLF (EFO_0007949)
                     target-disease association scores
  UniProt REST       subcellular localisation and protein family (modality feasibility)
  ClinicalTrials.gov target agents tested in cirrhosis / liver failure / ACLF

Outputs (figures/fig6/matrices/):
  druggability_v2.csv     one row per protein, all annotation axes
  drug_list_v2.csv        one row per named agent
  trials_landscape.csv    trial counts per agent x liver-indication block
  druggability_cache.json raw responses (offline re-runs)
"""
from pathlib import Path
import json
import time
import urllib.parse
import urllib.request
import urllib.error
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MAT = BASE / "figures" / "fig6" / "matrices"
MAT.mkdir(parents=True, exist_ok=True)
CACHE = MAT / "druggability_cache.json"
CACHE_DATA = json.loads(CACHE.read_text("utf-8")) if CACHE.exists() else {}

CH = "https://www.ebi.ac.uk/chembl/api/data"
OT = "https://api.platform.opentargets.org/api/v4/graphql"
PH = "https://pharos-api.ncats.io/graphql"
CT = "https://clinicaltrials.gov/api/v2/studies"

# gene -> (display, panel?, direction implied by the Cox model)
GENES = [
    ("TNFRSF1A", "TNF-RI", "panel", "inhibit"),
    ("TNFRSF21", "DR6", "panel", "inhibit"),
    ("REN", "Renin", "panel", "inhibit"),
    ("SCARF2", "SREC-II", "panel", "augment"),
    ("GH1", "GH", "panel", "augment"),
    ("ANGPT2", "ANG-2", "exploratory", "inhibit"),
    ("CCL14", "HCC-1", "exploratory", "inhibit"),
    ("CRP", "CRP", "exploratory", "inhibit"),
]
UNIPROT = {"TNFRSF1A": "P19438", "TNFRSF21": "O75509", "REN": "P00797",
           "SCARF2": "Q96GP6", "GH1": "P01241", "ANGPT2": "O15123",
           "CCL14": "Q16627", "CRP": "P02741"}
# agents to look up in the trial registry (named ChEMBL molecules checked below)
EXTRA_AGENTS = {"GH1": ["somatropin"]}


def _get(url, key, payload=None, timeout=90, tries=4, raw=False):
    if key in CACHE_DATA:
        return CACHE_DATA[key]
    hdr = {"Content-Type": "application/json"} if payload is not None else {}
    for t in range(tries):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode() if payload is not None else None, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            d = body.decode("utf-8", "ignore") if raw else json.loads(body)
            CACHE_DATA[key] = d
            return d
        except urllib.error.HTTPError as e:
            body = e.read(300).decode("utf-8", "ignore")
            if t == tries - 1:
                print(f"  ! {key} HTTP {e.code}: {body[:120]}")
                CACHE_DATA[key] = {"_error": e.code, "_body": body}
                return CACHE_DATA[key]
            time.sleep(2)
        except Exception as e:
            if t == tries - 1:
                print(f"  ! {key} {type(e).__name__}: {e}")
                CACHE_DATA[key] = {"_error": str(e)}
                return CACHE_DATA[key]
            time.sleep(2)


def gql(url, query, key):
    d = _get(url, key, payload={"query": query})
    if "_error" in d:
        return {}
    return d.get("data", {})


# ---------------------------------------------------------------- Open Targets
OT_TARGET = """
{ target(ensemblId:"%s"){
    id approvedSymbol approvedName biotype
    targetClass { label level }
    proteinIds { id source }
    tractability { modality label value }
    drugAndClinicalCandidates {
      count
      rows { drug { name maximumClinicalStage } } }
    safetyLiabilities { event datasource }
    associatedDiseases(page:{index:0 size:12}){ count rows { disease { id name } score } }
} }"""


def ot_target(ens):
    return gql(OT, OT_TARGET % ens, f"ot:{ens}").get("target") or {}


def ot_search(term):
    q = '{ search(queryString:"%s", entityNames:["target"]){ hits { id name entity } } }' % term
    d = gql(OT, q, f"otsearch:{term}")
    for h in (d.get("search") or {}).get("hits", []):
        if h.get("entity") == "target" and h.get("name", "").upper() == term.upper():
            return h["id"]
    return None


# -------------------------------------------------------------------- ChEMBL
def chembl_target_by_uniprot(acc):
    d = _get(f"{CH}/target.json?target_components__accession={acc}&limit=5", f"chtgt:{acc}")
    ts = (d or {}).get("targets", []) if isinstance(d, dict) else []
    for t in ts:
        if t.get("organism") == "Homo sapiens" and t.get("target_type") == "SINGLE PROTEIN":
            return t
    return ts[0] if ts else None


def chembl_mechanisms(tid):
    d = _get(f"{CH}/mechanism.json?target_chembl_id={tid}&limit=200", f"chmech:{tid}")
    return (d or {}).get("mechanisms", []) if isinstance(d, dict) else []


def chembl_molecules(ids):
    out = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        key = "chmol:" + ",".join(chunk)
        d = _get(f"{CH}/molecule.json?molecule_chembl_id__in={';'.join(chunk)}&limit=100", key)
        for m in (d or {}).get("molecules", []) if isinstance(d, dict) else []:
            out[m["molecule_chembl_id"]] = m
    return out


def chembl_molecule_by_name(name):
    d = _get(f"{CH}/molecule/search.json?q={urllib.parse.quote(name)}&limit=10",
             f"chsearch:{name}")
    for m in (d or {}).get("molecules", []) if isinstance(d, dict) else []:
        if (m.get("pref_name") or "").lower().strip() == name.lower():
            return m
    return None


# --------------------------------------------------- TDL proxy (Pharos-style)
# Pharos/IDG was unreachable from this host on 2026-11-07 (target query returned
# null for every symbol), so the target development level is derived here from
# the Open Targets tractability buckets using the published IDG definitions.
def tdl_proxy(has_approved, has_any_drug, has_ligand):
    if has_approved:
        return "Tclin (approved agent)"
    if has_any_drug:
        return "Tchem (clinical-stage agent)"
    if has_ligand:
        return "Tchem (high-quality ligand only)"
    return "Tbio/Tdark (no ligand evidence)"


# ------------------------------------------------------------------- UniProt
def uniprot_ann(acc):
    url = ("https://rest.uniprot.org/uniprotkb/%s.tsv?fields=accession,protein_name,"
           "cc_subcellular_location,protein_families" % acc)
    txt = _get(url, f"up:{acc}", timeout=60, raw=True)
    if not txt or not isinstance(txt, str):
        return "", ""
    lines = [l for l in txt.splitlines() if l.strip()]
    if len(lines) < 2:
        return "", ""
    f = lines[1].split("\t")
    loc = f[2] if len(f) > 2 else ""
    fam = f[3] if len(f) > 3 else ""
    # condense CC to its first top-level location token
    loc = loc.split("{")[0].replace("SUBCELLULAR LOCATION:", "").strip()
    loc = ";".join([x.strip().split(" Note=")[0] for x in loc.split(";")][:3])
    return loc[:180], fam[:120]


# -------------------------------------------------- disease-level association
def ot_disease_assoc(efo, genes):
    q = ('{ disease(efoId:"%s"){ name associatedTargets(page:{index:0 size:600})'
         '{ count rows{ target{approvedSymbol} score } } } }' % efo)
    d = gql(OT, q, f"otdis:{efo}")
    dis = d.get("disease") or {}
    rows = ((dis.get("associatedTargets") or {}).get("rows") or [])
    sc = {r["target"]["approvedSymbol"]: round(r["score"], 3) for r in rows
          if r["target"]["approvedSymbol"] in genes}
    rank = {r["target"]["approvedSymbol"]: i + 1 for i, r in enumerate(rows)
            if r["target"]["approvedSymbol"] in genes}
    total = (dis.get("associatedTargets") or {}).get("count")
    return sc, rank, total


OT_CIRR = "MONDO_0005155"
OT_ACLF = "EFO_0007949"
GENESET = {g for g, *_ in GENES}
CIRR_SC, CIRR_RANK, CIRR_N = ot_disease_assoc(OT_CIRR, GENESET)
ACLF_SC, ACLF_RANK, ACLF_N = ot_disease_assoc(OT_ACLF, GENESET)
print(f"OT cirrhosis targets {CIRR_N}; ACLF targets {ACLF_N}")


SALTS = {"FUMARATE", "HYDROCHLORIDE", "DIHYDROCHLORIDE", "SODIUM", "MALEATE", "MESYLATE",
         "TARTRATE", "CITRATE", "SUCCINATE", "PHOSPHATE", "ACETATE", "BESYLATE",
         "TOSYLATE", "CALCIUM", "POTASSIUM", "DIPHOSPHATE", "SULFATE", "LACTATE"}


def base_name(n):
    toks = [t for t in n.upper().replace("-", " ").split() if t not in SALTS]
    return " ".join(toks)


# --------------------------------------------------------- ClinicalTrials.gov
def ct_count(intr=None, cond=None, term=None):
    parts = ["pageSize=1", "countTotal=true", "fields=NCTId"]
    if intr:
        parts.append("query.intr=" + urllib.parse.quote(f'"{intr}"'))
    if cond:
        parts.append("query.cond=" + urllib.parse.quote(cond))
    if term:
        parts.append("query.term=" + urllib.parse.quote(term))
    url = CT + "?" + "&".join(parts)
    d = _get(url, "ct:" + "|".join(parts), timeout=60)
    if "_error" in d:
        return None
    try:
        return int(d.get("totalCount"))
    except (TypeError, ValueError):
        return len(d.get("studies") or [])


# ----------------------------------------------------------------- main loop
rows, drugs, trials = [], [], []
for gene, alias, role, direction in GENES:
    print("==", gene, alias)
    ens = ot_search(gene)
    ot = ot_target(ens) if ens else {}
    tract = {(t["modality"], t["label"]): t["value"] for t in (ot.get("tractability") or [])}
    cls = " / ".join(sorted({c["label"] for c in (ot.get("targetClass") or [])})) or ""

    ct = chembl_target_by_uniprot(UNIPROT[gene])
    tid = ct["target_chembl_id"] if ct else None
    mechs = chembl_mechanisms(tid) if tid else []
    mol_ids = sorted({m["molecule_chembl_id"] for m in mechs})
    minfo = chembl_molecules(mol_ids) if mol_ids else {}

    def ph(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    named, phases = [], []
    for m in mol_ids:
        mo = minfo.get(m, {})
        nm = (mo.get("pref_name") or "").strip()
        p = ph(mo.get("max_phase"))
        phases.append(p)
        if nm:
            named.append((nm, p, mo.get("first_approval"), mo.get("molecule_type")))
        drugs.append(dict(gene=gene, protein=alias, role=role, molecule=m, name=nm,
                          max_phase=p, first_approval=mo.get("first_approval"),
                          drug_type=mo.get("molecule_type"),
                          action=";".join(sorted({x.get("action_type") or "" for x in mechs
                                                  if x["molecule_chembl_id"] == m} - {""}))))

    # drug-centric: is the protein itself an approved drug substance?
    self_drug = None
    if gene == "GH1":
        mo = chembl_molecule_by_name("somatropin")
        if mo:
            self_drug = dict(name=mo.get("pref_name"), phase=ph(mo.get("max_phase")),
                             approval=mo.get("first_approval"), type=mo.get("molecule_type"))

    # Open Targets clinical candidates (covers agents ChEMBL may miss)
    cand = ot.get("drugAndClinicalCandidates") or {}
    cand_rows = cand.get("rows") or []
    ot_max_stage = ""
    if cand_rows:
        st = sorted({(r.get("drug") or {}).get("maximumClinicalStage") or "" for r in cand_rows} - {""})
        ot_max_stage = st[-1] if st else ""

    # safety liabilities
    sl = ot.get("safetyLiabilities") or []

    # OT target-disease association for cirrhosis / liver terms
    liver_terms = ("cirrhos", "liver", "hepat", "fibros")
    liver_hits = [(r["disease"]["name"], round(r["score"], 3))
                  for r in ((ot.get("associatedDiseases") or {}).get("rows") or [])
                  if any(k in r["disease"]["name"].lower() for k in liver_terms)]
    n_assoc = (ot.get("associatedDiseases") or {}).get("count")
    tdl = tdl_proxy(bool(tract.get(("SM", "Approved Drug")) or tract.get(("AB", "Approved Drug"))),
                    (cand.get("count") or 0) > 0 or bool(phases and max(phases) >= 1),
                    bool(tract.get(("SM", "High-Quality Ligand")))
                    or bool(tract.get(("SM", "Structure with Ligand"))))
    loc, fam = uniprot_ann(UNIPROT[gene])

    # direction congruence: does an existing agent act the way the data imply?
    acts = {(m.get("action_type") or "").upper() for m in mechs} - {""}
    if direction == "inhibit":
        congruent = "yes" if acts & {"INHIBITOR", "ANTAGONIST", "BLOCKER", "NEGATIVE MODULATOR"} else "no agent"
    else:
        congruent = ("yes (replacement)" if self_drug else
                     "yes" if acts & {"AGONIST", "ACTIVATOR", "POSITIVE MODULATOR"} else "no agent")

    rows.append(dict(
        gene=gene, protein=alias, role=role, implied_direction=direction,
        uniprot=UNIPROT[gene], target_class=cls,
        chembl_target=(f"{tid} ({ct['pref_name']})" if ct else ""),
        n_mechanisms=len(mechs), n_drugs=len(mol_ids),
        n_named=int(sum(1 for n, _, _, _ in named)),
        max_phase=max(phases) if phases else 0.0,
        n_approved=int(sum(1 for p in phases if p >= 4)),
        action_types=";".join(sorted({m.get("action_type") or "" for m in mechs} - {""})),
        self_drug=self_drug["name"] if self_drug else "",
        self_drug_phase=self_drug["phase"] if self_drug else "",
        ot_n_clinical_candidates=cand.get("count") or 0,
        ot_max_clinical_stage=ot_max_stage,
        ot_sm_approved=tract.get(("SM", "Approved Drug"), ""),
        ot_sm_druggable_family=tract.get(("SM", "Druggable Family"), ""),
        ot_sm_pocket=tract.get(("SM", "Med-Quality Pocket"), "") or tract.get(("SM", "High-Quality Pocket"), ""),
        ot_ab_approved=tract.get(("AB", "Approved Drug"), ""),
        ot_ab_extracellular=tract.get(("AB", "UniProt loc high conf"), "")
                          or tract.get(("AB", "GO CC high conf"), ""),
        ot_ab_phase1=tract.get(("AB", "Phase 1 Clinical"), ""),
        n_safety_liabilities=len(sl),
        n_assoc_diseases=n_assoc,
        liver_assoc=("; ".join(f"{n} ({s})" for n, s in liver_hits[:3]) if liver_hits else ""),
        uniprot_loc=loc,
        protein_family=fam,
        ot_cirrhosis_score=CIRR_SC.get(gene, 0.0),
        ot_cirrhosis_rank=CIRR_RANK.get(gene, ""),
        ot_aclfs_score=ACLF_SC.get(gene, 0.0),
        ot_aclfs_rank=ACLF_RANK.get(gene, ""),
        direction_congruent=congruent,
        pharos_tdl=tdl,
        pharos_family="",
        n_trials_liver="", n_trials_any="",
    ))

    # trial names: ChEMBL returns salt forms ("ALISKIREN FUMARATE"), the registry
    # indexes the base compound, so both strings are tried and the larger is kept
    agents = [n for n, p, _, _ in named if p >= 0.5] + EXTRA_AGENTS.get(gene, [])
    for a in sorted(set(agents)):
        cond = "(cirrhosis OR \"liver failure\" OR \"hepatic insufficiency\")"
        n_any = max(ct_count(intr=a) or 0, ct_count(intr=base_name(a)) or 0)
        n_liver = max(ct_count(intr=a, cond=cond) or 0,
                      ct_count(intr=base_name(a), cond=cond) or 0)
        trials.append(dict(gene=gene, protein=alias, agent=base_name(a), n_trials_any=n_any,
                           n_trials_liver=n_liver))

CACHE.write_text(json.dumps(CACHE_DATA), "utf-8")

dr = pd.DataFrame(rows)
# roll trial counts up to the protein
tl = pd.DataFrame(trials)
if len(tl):
    agg = tl.groupby("gene").agg(n_trials_any=("n_trials_any", "sum"),
                                 n_trials_liver=("n_trials_liver", "sum"))
    dr = dr.set_index("gene")
    dr["n_trials_any"] = agg["n_trials_any"]
    dr["n_trials_liver"] = agg["n_trials_liver"]
    dr = dr.reset_index()

dr.to_csv(MAT / "druggability_v2.csv", index=False)
pd.DataFrame(drugs).to_csv(MAT / "drug_list_v2.csv", index=False)
tl.to_csv(MAT / "trials_landscape.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 50)
print(dr.to_string(index=False))
print("\n--- trials ---")
print(tl.to_string(index=False) if len(tl) else "(none)")
