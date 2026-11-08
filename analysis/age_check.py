# -*- coding: utf-8 -*-
"""年龄是否解释了面板的增益：三个独立量。

1. 年龄单独在验证集上的 90 天 AUC（线性预测值即年龄本身，故为单变量 AUC）；
2. MELD + age 相对 MELD 的配对 ΔAUC（推导集拟合，验证集配对 bootstrap）；
3. 面板在 MELD + age 之上的配对 ΔAUC（与 panel_increment.py 的 "MELD + age" 行一致）。

前两项用于回答"面板只是年龄的替身"这一质疑；第三项是反驳的关键——
若面板在已含年龄的锚点上仍有增量，则增益不是年龄信息。

输出：analysis/output2/age_check.json
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
base, clin, P5 = ns["base"].copy(), ns["clin"], ns["P5"]
base["id"] = clin["ID"].astype(str).str.strip().to_numpy()

sc = pd.read_csv(OUT / "scores_strict_1390.csv")
sc["id"] = sc["id"].astype(str).str.strip()
df = base.merge(sc[["id", "age", "MELD"]], on="id")

DEV = ["nfa", "xna", "rja", "dta", "xha"]
dev, val = df[df.center.isin(DEV)].reset_index(drop=True), df[~df.center.isin(DEV)].reset_index(drop=True)
SEED = 20261108


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


def delta_auc(v, a, b, B=2000):
    ok = ((v["time"] >= 90) | (v["event"] == 1)).to_numpy()
    y, a, b = v["event"].to_numpy()[ok], a[ok], b[ok]
    n = len(y)
    pt = fast_auc(y, a) - fast_auc(y, b)
    dr = []
    rng = np.random.default_rng(SEED)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() < 5 or y[i].sum() > n - 5:
            continue
        dr.append(fast_auc(y[i], a[i]) - fast_auc(y[i], b[i]))
    dr = np.array(dr)
    lo, hi = np.percentile(dr, [2.5, 97.5])
    return dict(delta=float(pt), lo=float(lo), hi=float(hi),
                p=float(min(2 * min((dr <= 0).mean(), (dr >= 0).mean()), 1.0)),
                n=int(n), events=int(y.sum()))


res = {}

# ---- 1. 年龄单独 -------------------------------------------------------- #
ok = ((val["time"] >= 90) | (val["event"] == 1)) & val["age"].notna()
res["age_only"] = dict(
    auc90=float(fast_auc(val.loc[ok, "event"].to_numpy(), val.loc[ok, "age"].to_numpy())),
    cindex=float(concordance_index(val.loc[ok, "time"], -val.loc[ok, "age"], val.loc[ok, "event"])),
    n=int(ok.sum()), events=int(val.loc[ok, "event"].sum()))
print(f"age alone           AUC90={res['age_only']['auc90']:.3f}  "
      f"C={res['age_only']['cindex']:.3f}  n={res['age_only']['n']}  ev={res['age_only']['events']}")

# ---- 2. MELD + age vs MELD --------------------------------------------- #
m_meld = fit(dev, ["meld"])
m_meldage = fit(dev, ["meld", "age"])
v = val.dropna(subset=["meld", "age"]).reset_index(drop=True)
p_meld = np.log(m_meld.predict_partial_hazard(v[["meld"]]).to_numpy())
p_meldage = np.log(m_meldage.predict_partial_hazard(v[["meld", "age"]]).to_numpy())
res["meld_plus_age_vs_meld"] = delta_auc(v, p_meldage, p_meld)
res["meld_plus_age_vs_meld"]["c_anchor"] = float(concordance_index(v["time"], -p_meld, v["event"]))
res["meld_plus_age_vs_meld"]["c_full"] = float(concordance_index(v["time"], -p_meldage, v["event"]))
d = res["meld_plus_age_vs_meld"]
print(f"MELD+age vs MELD    dAUC={d['delta']:+.3f} ({d['lo']:+.3f}~{d['hi']:+.3f}) P={d['p']:.3f}  "
      f"C {d['c_anchor']:.3f}->{d['c_full']:.3f}")

# ---- 3. 面板在 MELD + age 之上 ----------------------------------------- #
m_full = fit(dev, ["meld", "age"] + P5)
p_full = np.log(m_full.predict_partial_hazard(v[["meld", "age"] + P5]).to_numpy())
res["panel_over_meld_plus_age"] = delta_auc(v, p_full, p_meldage)
res["panel_over_meld_plus_age"]["c_full"] = float(concordance_index(v["time"], -p_full, v["event"]))
d = res["panel_over_meld_plus_age"]
print(f"panel over MELD+age dAUC={d['delta']:+.3f} ({d['lo']:+.3f}~{d['hi']:+.3f}) P={d['p']:.3f}  "
      f"C {res['meld_plus_age_vs_meld']['c_full']:.3f}->{d['c_full']:.3f}")

(OUT / "age_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n输出 age_check.json")
