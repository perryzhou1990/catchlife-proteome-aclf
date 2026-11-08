# -*- coding: utf-8 -*-
"""五蛋白与三蛋白面板的直接比较（同一验证集，配对 bootstrap）。

正文与 Table 2 脚注引用 ΔAUC +0.006 与 ΔC +0.004；此处独立复算并给出区间。
输出：analysis/output2/panel_5v3.json
"""
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
base, clin, P5, P3 = ns["base"].copy(), ns["clin"], ns["P5"], ns["P3"]
base["id"] = clin["ID"].astype(str).str.strip().to_numpy()

DEV = ["nfa", "xna", "rja", "dta", "xha"]
dev, val = base[base.center.isin(DEV)].reset_index(drop=True), base[~base.center.isin(DEV)].reset_index(drop=True)
SEED, B = 20261108, 2000


def fast_auc(y, x):
    r = rankdata(x)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit(d, cols):
    f = d[["time", "event", "center"] + cols].dropna()
    try:
        c = CoxPHFitter(penalizer=0.0)
        c.fit(f, "time", "event", strata=["center"])
        return c
    except Exception:
        c = CoxPHFitter(penalizer=0.0)
        c.fit(f[["time", "event"] + cols], "time", "event")
        return c


m5 = fit(dev, ["meld"] + P5)
m3 = fit(dev, ["meld"] + P3)
v = val.dropna(subset=["meld"] + P5 + P3).reset_index(drop=True)
p5 = np.log(m5.predict_partial_hazard(v[["meld"] + P5]).to_numpy())
p3 = np.log(m3.predict_partial_hazard(v[["meld"] + P3]).to_numpy())

t, y = v["time"].to_numpy(), v["event"].to_numpy()
n = len(y)
ok = (t >= 90) | (y == 1)
rng = np.random.default_rng(SEED)

d_auc_pt = fast_auc(y[ok], p5[ok]) - fast_auc(y[ok], p3[ok])
d_c_pt = concordance_index(t, -p5, y) - concordance_index(t, -p3, y)
au, cu = [], []
for _ in range(B):
    i = rng.integers(0, n, n)
    if y[i].sum() < 5 or y[i].sum() > n - 5:
        continue
    ii = i[ok[i]]
    if (y[ii].sum() < 5) or (y[ii].sum() > len(ii) - 5):
        continue
    au.append(fast_auc(y[ii], p5[ii]) - fast_auc(y[ii], p3[ii]))
    cu.append(concordance_index(t[i], -p5[i], y[i]) - concordance_index(t[i], -p3[i], y[i]))
au, cu = np.array(au), np.array(cu)


def summarise(pt, arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    p = min(2 * min((arr <= 0).mean(), (arr >= 0).mean()), 1.0)
    return dict(delta=float(pt), lo=float(lo), hi=float(hi), p=float(p))


res = dict(n=n, events=int(y.sum()),
           panel5=dict(auc90=float(fast_auc(y[ok], p5[ok])), cindex=float(concordance_index(t, -p5, y))),
           panel3=dict(auc90=float(fast_auc(y[ok], p3[ok])), cindex=float(concordance_index(t, -p3, y))),
           dAUC=summarise(d_auc_pt, au), dC=summarise(d_c_pt, cu))
print(f"5P: AUC90 {res['panel5']['auc90']:.3f}  C {res['panel5']['cindex']:.3f}")
print(f"3P: AUC90 {res['panel3']['auc90']:.3f}  C {res['panel3']['cindex']:.3f}")
for k in ("dAUC", "dC"):
    r = res[k]
    print(f"{k}: {r['delta']:+.3f} ({r['lo']:+.3f}~{r['hi']:+.3f}) P={r['p']:.3f}")
(OUT / "panel_5v3.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
