# -*- coding: utf-8 -*-
"""Second-round review: verify every numerically checkable claim.

Reviewer's challenges handled here
--------------------------------------------------------------------
(1) LOCO  abstract quotes +0.044, which pools centres that took part in
           selection; the reviewer asks for the clean-centre value.
(2) 200x   "only 2 of the 200 allocations avoid overlap" is said to be
           incompatible with uniform 5-of-13 sampling (expected 0.16).
           Here we enumerate all C(13,5)=1287 allocations and apply the
           SAME acceptance rule the generator used (dev events>=60,
           val events>=15), which is the point the reviewer could not see.
(3) sex    994 vs 995 inconsistency between Table 1 and the footnote.
(4) CRP    clinical CRP (n=956) vs proteomic CRP: reviewer asks for the
           clinical-CRP comparator to be reported.
(5) PH     derivation vs validation Schoenfeld results; sex degeneracy.
(6) stabl  full stability-frequency table for both protocols (SREC-II/GH).
(7) age    is age present anywhere in the delivered clinical workbook?

Outputs -> analysis/output5/
"""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "analysis" / "output5"
OUT.mkdir(parents=True, exist_ok=True)

SRC = BASE / "figures" / "fig4" / "code" / "fig4_make.py"
SPLIT = "# ---------------- figure ----------------"
text = SRC.read_text(encoding="utf-8")
ns = {"__name__": "__panel_data__", "__file__": str(SRC)}
exec(compile(text[: text.index(SPLIT)], str(SRC), "exec"), ns)
fit, base, P5, clin = ns["fit"], ns["base"], ns["P5"], ns["clin"]
DEV_C = ns["DEV_C"]
dev = base[base.center.isin(DEV_C)].reset_index(drop=True)
val = base[~base.center.isin(DEV_C)].reset_index(drop=True)
BIG5 = set(DEV_C)

print("dev n=%d ev=%d | val n=%d ev=%d" % (len(dev), dev.event.sum(), len(val), val.event.sum()))

# ---------------------------------------------------------------- (1) LOCO
loco = pd.read_csv(BASE / "analysis" / "output4" / "loco_clean.csv")
loco["clean"] = loco["clean"].astype(str).str.lower() == "true"
def wavg(df):
    return float((df.d_c * df.events).sum() / df.events.sum())
loco_rows = []
for name, sub in [("all 9 evaluable centres", loco),
                  ("centres that took part in selection", loco[~loco.clean]),
                  ("held-out centres (no part in selection)", loco[loco.clean])]:
    loco_rows.append(dict(group=name, n_centres=len(sub), events=int(sub.events.sum()),
                          delta_c=round(wavg(sub), 4),
                          n_favour=int((sub.d_c > 0).sum())))
pd.DataFrame(loco_rows).to_csv(OUT / "loco_groups.csv", index=False, encoding="utf-8-sig")
print("\n[1] LOCO\n", pd.DataFrame(loco_rows).to_string(index=False))

# ------------------------------------------------------- (2) 200x allocations
centres = sorted(base.center.unique())
ev = base.groupby("center").event.sum().to_dict()
all_splits = list(itertools.combinations(centres, 5))
def ok(dev_c):
    e_dev = sum(ev[c] for c in dev_c)
    e_val = int(base.event.sum()) - e_dev
    return e_dev >= 60 and e_val >= 15
admissible = [c for c in all_splits if ok(c)]
clean = [c for c in admissible if set(c) == BIG5]
p_clean_cond = len(clean) / len(admissible)
p_clean_unif = 1 / len(all_splits)
obs = pd.read_csv(BASE / "analysis" / "output2" / "repeated_centre_split.csv")
obs["dev_set"] = obs.dev_centres.apply(lambda s: frozenset(str(s).split("|")))
obs_clean = int((obs.dev_set.apply(lambda s: s == frozenset(BIG5))).sum())
sim = []
rng = np.random.default_rng(1)
for _ in range(4000):                      # Monte-Carlo under the same rule
    k = 0
    for _ in range(200):
        while True:
            c = tuple(rng.choice(centres, 5, replace=False))
            if ok(c):
                break
        k += int(set(c) == BIG5)
    sim.append(k)
rows = dict(
    n_all_allocations=len(all_splits),
    n_admissible_under_event_rule=len(admissible),
    p_clean_if_uniform=round(p_clean_unif, 6),
    expected_clean_in_200_if_uniform=round(200 * p_clean_unif, 3),
    p_clean_given_event_rule=round(p_clean_cond, 4),
    expected_clean_in_200_given_rule=round(200 * p_clean_cond, 2),
    observed_clean=obs_clean,
    mc_median_clean=float(np.median(sim)),
    mc_p_ge_observed=float(np.mean(np.array(sim) >= obs_clean)),
)
pd.Series(rows).to_csv(OUT / "split_probability.csv", encoding="utf-8-sig", header=["value"])
print("\n[2] split probability\n", json.dumps(rows, indent=1))

# --------------------------------------------------------------- (3) sex
raw = clin["Gender"]
tab = raw.value_counts(dropna=False).rename_axis("raw_code").reset_index(name="n")
tab.to_csv(OUT / "sex_raw_codes.csv", index=False, encoding="utf-8-sig")
byc = (base.groupby("center").sex.agg(["size", "sum"])
       .assign(pct_male=lambda d: 100 * d["sum"] / d["size"])
       .sort_values("pct_male"))
byc.to_csv(OUT / "sex_by_centre.csv", encoding="utf-8-sig")
n_all_male = int((base.sex == 1).sum())
rows = dict(raw_male=int(tab.loc[tab.raw_code == 1, "n"].iloc[0]),
            raw_female=int(tab.loc[tab.raw_code == 0, "n"].iloc[0]),
            raw_code2=int(tab.loc[tab.raw_code == 2, "n"].iloc[0]) if (tab.raw_code == 2).any() else 0,
            in_model_male=n_all_male,
            centres=int(len(byc)),
            centres_above_90pct_single_sex=int(((byc.pct_male > 90) | (byc.pct_male < 10)).sum()),
            centres_100pct_single_sex=int(((byc.pct_male == 100) | (byc.pct_male == 0)).sum()))
pd.Series(rows).to_csv(OUT / "sex_summary.csv", encoding="utf-8-sig", header=["value"])
print("\n[3] sex\n", json.dumps(rows, indent=1))

# ------------------------------------------------------ (4) clinical CRP
crp = pd.to_numeric(clin["CRP"], errors="coerce").to_numpy()
base2 = base.copy()
base2["crp_clin"] = crp
base2["crp_clin_z"] = ((np.log2(base2.crp_clin.clip(lower=1))
                        - np.log2(base2.crp_clin.clip(lower=1)).mean())
                       / np.log2(base2.crp_clin.clip(lower=1)).std(ddof=1))
d2 = base2[base2.center.isin(DEV_C)].dropna(subset=["crp_clin_z"]).reset_index(drop=True)
v2 = base2[~base2.center.isin(DEV_C)].dropna(subset=["crp_clin_z"]).reset_index(drop=True)
rows = []
for label, cols in [("MELD", ["meld"]),
                    ("MELD + clinical CRP", ["meld", "crp_clin_z"]),
                    ("MELD + 5-protein panel", ["meld"] + P5),
                    ("MELD + panel + clinical CRP", ["meld"] + P5 + ["crp_clin_z"])]:
    m = fit(d2, cols)
    lp = np.log(m.predict_partial_hazard(v2[cols]).to_numpy())
    rows.append(dict(model=label, n_dev=len(d2), ev_dev=int(d2.event.sum()),
                     n_val=len(v2), ev_val=int(v2.event.sum()),
                     auc90=round(float(roc_auc_score(v2.event, lp)), 4),
                     c_index=round(float(concordance_index(v2.time, -lp, v2.event)), 4)))
pd.DataFrame(rows).to_csv(OUT / "crp_clinical_models.csv", index=False, encoding="utf-8-sig")
print("\n[4] clinical-CRP models (complete cases only)\n",
      pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- (5) PH
ph = pd.read_csv(BASE / "analysis" / "output4" / "ph_test.csv")
ph["holm_recomputed"] = np.nan
for ds, sub in ph.groupby("dataset"):
    p = sub.p.values
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val_ = min(1.0, (m - rank) * p[idx])
        running = max(running, val_)
        adj[idx] = running
    ph.loc[sub.index, "holm_recomputed"] = adj
ph.to_csv(OUT / "ph_test_holm.csv", index=False, encoding="utf-8-sig")
print("\n[5] PH\n", ph[["dataset", "variable", "p", "holm_recomputed"]].to_string(index=False))

# --------------------------------------------------- (6) stability frequencies
rows = []
for tag, f in [("primary883", "stability_primary883.csv"),
               ("filter665", "stability_filter665.csv")]:
    d = pd.read_csv(BASE / "analysis" / "output2" / f)
    d["protocol"] = tag
    rows.append(d[["protein", "freq", "protocol"]])
st = pd.concat(rows).pivot_table(index="protein", columns="protocol", values="freq")
st = st.fillna(0.0).sort_values("primary883", ascending=False)
st["max_freq"] = st.max(axis=1)
st["in_panel"] = [p.replace(".", "-") in {"TNF-RI", "DR6", "Renin", "SREC-II", "GH"} for p in st.index]
st.to_csv(OUT / "stability_both_protocols.csv", encoding="utf-8-sig")
top = st.head(15).reset_index()
top.to_csv(OUT / "stability_top15_both.csv", index=False, encoding="utf-8-sig")
print("\n[6] stability, top 15 by primary883\n", top.to_string(index=False))
for name in ["SREC.II", "GH", "ANG.2"]:
    if name in st.index:
        print("   ", name, st.loc[name, ["primary883", "filter665"]].round(3).to_dict())

# ---------------------------------------------------------------- (7) age
cols = [str(c).strip() for c in clin.columns]
age_hits = [c for c in cols if "age" in c.lower() or "年龄" in c]
rows = dict(n_fields=len(cols), age_like_fields=age_hits if age_hits else "NONE")
pd.Series({k: str(v) for k, v in rows.items()}).to_csv(
    OUT / "age_field_audit.csv", encoding="utf-8-sig", header=["value"])
print("\n[7] age\n", rows)

json.dump(dict(loco=loco_rows, splits=rows, sex=rows), open(OUT / "second_review_summary.json", "w"),
          indent=1, default=str)
print("\ndone ->", OUT)
