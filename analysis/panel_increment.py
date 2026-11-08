# -*- coding: utf-8 -*-
"""五蛋白面板在各临床锚点之上的增量（推导集拟合，验证集配对 bootstrap）"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "analysis" / "output2"
SRC = BASE / "figures" / "fig4" / "code" / "fig4_make.py"
txt = SRC.read_text(encoding="utf-8")
ns = {"__name__": "x", "__file__": str(SRC)}
exec(compile(txt[: txt.index("# ---------------- figure ----------------")], str(SRC), "exec"), ns)
base, clin, P5 = ns["base"].copy(), ns["clin"], ns["P5"]
base["id"] = clin["ID"].astype(str).str.strip().to_numpy()

sc = pd.read_csv(OUT / "scores_strict_1390.csv")
sc["id"] = sc["id"].astype(str).str.strip()
df = base.merge(sc[["id", "age", "MELD3", "CLIF_OF", "MELD", "MELD_Na", "aclf_any"]], on="id")
DEV = ["nfa", "xna", "rja", "dta", "xha"]
dev, val = df[df.center.isin(DEV)].reset_index(drop=True), df[~df.center.isin(DEV)].reset_index(drop=True)
SEED = 20261108


def fast_auc(y, x):
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit(d, cols):
    f = d[["time", "event", "center"] + cols].dropna()
    try:
        c = CoxPHFitter(penalizer=0.0); c.fit(f, "time", "event", strata=["center"]); return c
    except Exception:
        c = CoxPHFitter(penalizer=0.0); c.fit(f[["time", "event"] + cols], "time", "event"); return c


def delta_auc(v, a, b, B=2000):
    ok = ((v["time"] >= 90) | (v["event"] == 1)).to_numpy()
    y, a, b = v["event"].to_numpy()[ok], a[ok], b[ok]
    n = len(y); pt = fast_auc(y, a) - fast_auc(y, b); dr = []
    rng = np.random.default_rng(SEED)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() < 5 or y[i].sum() > n - 5:
            continue
        dr.append(fast_auc(y[i], a[i]) - fast_auc(y[i], b[i]))
    dr = np.array(dr); lo, hi = np.percentile(dr, [2.5, 97.5])
    return pt, lo, hi, min(2 * min((dr <= 0).mean(), (dr >= 0).mean()), 1.0), n, int(y.sum())


ANCHORS = {
    "MELD": ["meld"],
    "MELD + age": ["meld", "age"],
    "MELD-Na": ["MELD_Na"],
    "MELD 3.0": ["MELD3"],
    "CLIF-C OF": ["CLIF_OF"],
}
rows = []
for lab, cols in ANCHORS.items():
    m0 = fit(dev, cols)
    m1 = fit(dev, cols + P5)
    v = val.dropna(subset=cols + P5).reset_index(drop=True)
    p0 = np.log(m0.predict_partial_hazard(v[cols]).to_numpy())
    p1 = np.log(m1.predict_partial_hazard(v[cols + P5]).to_numpy())
    pt, lo, hi, p, n, ev = delta_auc(v, p1, p0)
    c0 = concordance_index(v["time"], -p0, v["event"])
    c1 = concordance_index(v["time"], -p1, v["event"])
    a0 = fast_auc(v.loc[(v.time >= 90) | (v.event == 1), "event"], p0[(v.time >= 90) | (v.event == 1)].astype(float))
    a1 = fast_auc(v.loc[(v.time >= 90) | (v.event == 1), "event"], p1[(v.time >= 90) | (v.event == 1)].astype(float))
    print(f"{lab:<14} AUC {a0:.3f}->{a1:.3f}  ΔAUC {pt:+.3f} ({lo:+.3f}~{hi:+.3f}) P={p:.3f}  "
          f"C {c0:.3f}->{c1:.3f}  n={n} ev={ev}")
    rows.append(dict(anchor=lab, auc_anchor=a0, auc_full=a1, delta=pt, lo=lo, hi=hi, p=p,
                     c_anchor=c0, c_full=c1, n=n, events=ev))
pd.DataFrame(rows).to_csv(OUT / "panel_increment_anchors_v2.csv", index=False, encoding="utf-8-sig")
(OUT / "panel_increment_anchors_v2.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("\n输出 panel_increment_anchors_v2.csv")
