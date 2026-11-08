# -*- coding: utf-8 -*-
"""PH re-check (second round) + split-overlap stratification.

Why this file exists
--------------------
Round-1 checks reported Schoenfeld statistics (Renin 40.8, DR6 29.0,
SREC-II 9.0, sex 1035.9 in the derivation set) that CANNOT be reproduced
with lifelines' standard implementation on the same fitted model. Passing a
covariate frame whose column order/contents differ from the model's parameter
vector silently mis-assigns residuals. This script rebuilds the test from a
frame whose columns are taken in the model's own order, runs both the
Grambsch-Therneau (KM) and Harrell (rank) transforms, and Holm-corrects
within each dataset. The earlier numbers are superseded.

Outputs -> analysis/output5/
    ph_test_v2.csv          per-term test, both transforms, both sets
    split_overlap.csv       gain stratified by number of selection centres
                            reappearing in the validation set
"""
from pathlib import Path
import io
import contextlib
import numpy as np
import pandas as pd
from lifelines.statistics import proportional_hazard_test

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "analysis" / "output5"
OUT.mkdir(parents=True, exist_ok=True)

SRC = BASE / "figures" / "fig4" / "code" / "fig4_make.py"
text = SRC.read_text(encoding="utf-8")
ns = {"__name__": "__panel_data__", "__file__": str(SRC)}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(text[: text.index("# ---------------- figure ----------------")],
                 str(SRC), "exec"), ns)
fit, base, P5, DEV_C = ns["fit"], ns["base"], ns["P5"], ns["DEV_C"]
COLS = ["meld", "sex"] + P5
keep = ["time", "event", "center"] + COLS
dev = base.loc[base.center.isin(DEV_C), keep].reset_index(drop=True)
val = base.loc[~base.center.isin(DEV_C), keep].reset_index(drop=True)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[idx]))
        adj[idx] = run
    return adj


rows = []
for tag, d in [("derivation (n=995, 116 events)", dev), ("validation refit (n=395, 46 events)", val)]:
    model = fit(d, COLS)
    for tf in ["km", "rank"]:
        r = proportional_hazard_test(model, d[["time", "event", "center"] + COLS],
                                     time_transform=tf).summary
        sub = pd.DataFrame(dict(dataset=tag, transform=tf,
                                variable=r.index,
                                statistic=r["test_statistic"].to_numpy(),
                                p=r["p"].to_numpy()))
        sub["p_holm"] = holm(sub.p)
        rows.append(sub)
ph = pd.concat(rows)
ph.to_csv(OUT / "ph_test_v2.csv", index=False, encoding="utf-8-sig")
print(ph.round(5).to_string(index=False))

# ---------------- split-overlap stratification -------------------------------
sp = pd.read_csv(BASE / "analysis" / "output2" / "repeated_centre_split.csv")
BIG5 = set(DEV_C)
sp["n_sel_in_val"] = sp.dev_centres.apply(lambda s: len(BIG5 - set(str(s).split("|"))))
g = (sp.groupby("n_sel_in_val")
     .agg(n_splits=("split", "size"),
          events_in_val=("ev_val", "median"),
          median_d_auc=("d_auc", "median"),
          median_d_c=("d_c", "median"),
          min_d_auc=("d_auc", "min"))
     .round(4).reset_index())
g.to_csv(OUT / "split_overlap.csv", index=False, encoding="utf-8-sig")
print("\n", g.to_string(index=False))
print("\nSpearman(n overlap, d_auc) = %.3f"
      % sp[["n_sel_in_val", "d_auc"]].corr(method="spearman").iloc[0, 1])
