# -*- coding: utf-8 -*-
"""
严格版评分判别力比较

所有评分由 analysis/score_engine.py 按原始文献公式重算（scores_strict_1390.csv），
不使用任何外部提供的现成分数。

输出
----
analysis/output2/score_discrimination_strict.json
analysis/output2/table2_strict.csv          主表：全队列 / 推导集 / 验证集
analysis/output2/tableS_population.csv      人群特异评分（CLIF-C AD / CLIF-C ACLF / COSSH-II）
analysis/output2/panel_increment_anchors.csv 面板在各锚点评分之上的增量
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "analysis" / "output2"
SRC = BASE / "figures" / "fig4" / "code" / "fig4_make.py"
SPLIT = "# ---------------- figure ----------------"

# 复用 fig4 的数据段：同一队列、同一中心切分、同一 panel 蛋白
text = SRC.read_text(encoding="utf-8")
ns = {"__name__": "__panel_data__", "__file__": str(SRC)}
exec(compile(text[: text.index(SPLIT)], str(SRC), "exec"), ns)
base, dev, val, clin = ns["base"], ns["dev"], ns["val"], ns["clin"]
P5 = ns["P5"]
base = base.copy()
base["id"] = clin["ID"].astype(str).str.strip().to_numpy()

sc = pd.read_csv(OUT / "scores_strict_1390.csv")
sc["id"] = sc["id"].astype(str).str.strip()
COLS = ["id", "age", "event90", "time90", "cossh_day", "aclf_any", "hbv_aclf", "cirrhosis",
        "MELD", "MELD_Na", "MELD3", "CLIF_OF", "CLIF_AD", "CLIF_ACLF", "COSSH_II", "CTP"]
df = base.merge(sc[COLS], on="id", how="left", suffixes=("", "_s"))
assert len(df) == len(base) == 1390
assert df["event90"].eq(df["event"]).all(), "结局列对不齐"

LAB = {"MELD": "MELD", "MELD_Na": "MELD-Na", "MELD3": "MELD 3.0", "CLIF_OF": "CLIF-C OF",
       "CLIF_AD": "CLIF-C AD", "CLIF_ACLF": "CLIF-C ACLF", "COSSH_II": "COSSH-ACLF II",
       "CTP": "Child-Pugh"}
ALL = list(LAB)

RNG = np.random.default_rng(20261108)


def eval90(d: pd.DataFrame, col: str):
    """90 天 AUC：仅纳入随访满 90 天或 90 天内发生事件者。"""
    ok = d[col].notna() & ((d.time >= 90) | (d.event == 1))
    if ok.sum() < 30 or d.loc[ok, "event"].sum() < 5:
        return np.nan, int(ok.sum()), int(d.loc[ok, "event"].sum())
    return roc_auc_score(d.loc[ok, "event"], d.loc[ok, col]), int(ok.sum()), int(d.loc[ok, "event"].sum())


def cidx(d: pd.DataFrame, col: str):
    ok = d[col].notna()
    if ok.sum() < 30 or d.loc[ok, "event"].sum() < 5:
        return np.nan, int(ok.sum()), int(d.loc[ok, "event"].sum())
    return concordance_index(d.loc[ok, "time"], -d.loc[ok, col], d.loc[ok, "event"]), int(ok.sum()), int(
        d.loc[ok, "event"].sum())


def boot_delta_auc(d: pd.DataFrame, a: str, b: str, B: int = 2000):
    """配对 bootstrap：同一批完整病例上比较两评分的 90 天 AUC 之差。"""
    ok = d[a].notna() & d[b].notna() & ((d.time >= 90) | (d.event == 1))
    s = d[ok]
    y = s["event"].to_numpy()
    if len(s) < 50 or y.sum() < 10:
        return None
    xa, xb = s[a].to_numpy(), s[b].to_numpy()
    point = roc_auc_score(y, xa) - roc_auc_score(y, xb)
    diffs = []
    n = len(y)
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if y[i].sum() < 5 or y[i].sum() > n - 5:
            continue
        diffs.append(roc_auc_score(y[i], xa[i]) - roc_auc_score(y[i], xb[i]))
    v = np.array(diffs)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return dict(delta=float(point), lo=float(lo), hi=float(hi), p=float(min(p, 1.0)),
                n=int(len(s)), events=int(y.sum()))


# --------------------------------------------------------------------------- #
print("=" * 100)
print("A. 三个数据集上的判别力（各评分在自身可得病例上计算）")
print("=" * 100)
print(f"{'score':<16}{'C 全队列':>10}{'n':>6}{'C 推导':>9}{'C 验证':>9}{'AUC90 全':>10}{'AUC90 推导':>11}{'AUC90 验证':>11}")
rows = []
sets = {"all": df, "dev": df[df.center.isin(["nfa", "xna", "rja", "dta", "xha"])],
        "val": df[~df.center.isin(["nfa", "xna", "rja", "dta", "xha"])]}
for col in ALL:
    c_all, n_all, e_all = cidx(sets["all"], col)
    c_dev, _, _ = cidx(sets["dev"], col)
    c_val, n_val, e_val = cidx(sets["val"], col)
    a_all, _, _ = eval90(sets["all"], col)
    a_dev, _, _ = eval90(sets["dev"], col)
    a_val, an_val, ae_val = eval90(sets["val"], col)
    print(f"{LAB[col]:<16}{c_all:>10.3f}{n_all:>6}{c_dev:>9.3f}{c_val:>9.3f}"
          f"{a_all:>10.3f}{a_dev:>11.3f}{a_val:>11.3f}")
    rows.append(dict(score=LAB[col], key=col, c_all=c_all, n_all=n_all, ev_all=e_all,
                     c_dev=c_dev, c_val=c_val, n_val=n_val, ev_val=e_val,
                     auc_all=a_all, auc_dev=a_dev, auc_val=a_val,
                     auc_val_n=an_val, auc_val_ev=ae_val))
pd.DataFrame(rows).to_csv(OUT / "table2_strict.csv", index=False, encoding="utf-8-sig")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 100)
print("B. 验证集上各评分相对 MELD 的 ΔAUC（配对 bootstrap 2,000 次）")
print("=" * 100)
deltas = {}
for col in ALL:
    if col == "MELD":
        continue
    r = boot_delta_auc(sets["val"], col, "MELD")
    deltas[col] = r
    if r:
        print(f"  {LAB[col]:<16} ΔAUC {r['delta']:+.3f} (95%CI {r['lo']:+.3f}~{r['hi']:+.3f}; "
              f"P={r['p']:.3f})  n={r['n']} 事件={r['events']}")
    else:
        print(f"  {LAB[col]:<16} 事件数不足，未计算")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 100)
print("C. 人群特异评分（全队列；各评分仅在其设计人群内评估）")
print("=" * 100)
pop = []
subsets = {
    "Non-ACLF at any time": (df.aclf_any == False, ["CLIF_AD", "MELD", "MELD_Na", "MELD3", "CTP"]),
    "ACLF (admission or in-hospital)": (df.aclf_any == True, ["CLIF_ACLF", "COSSH_II", "MELD", "MELD3", "CLIF_OF"]),
    "HBV-ACLF": (df.hbv_aclf == True, ["COSSH_II", "CLIF_ACLF", "MELD", "MELD3"]),
}
for name, (mask, cols) in subsets.items():
    s = df[mask]
    print(f"\n  {name}: n={len(s)}, 90d 事件={int(s.event.sum())} ({s.event.mean()*100:.1f}%)")
    for col in cols:
        c, n, e = cidx(s, col)
        a, an, ae = eval90(s, col)
        print(f"    {LAB[col]:<16} C={c:.3f}  AUC90={a:.3f}  (n={n}/{an}, 事件={e})")
        pop.append(dict(population=name, score=LAB[col], cindex=c, auc90=a, n=n, events=e,
                        n_auc=an, ev_auc=ae))
pd.DataFrame(pop).to_csv(OUT / "tableS_population.csv", index=False, encoding="utf-8-sig")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 100)
print("D. 五蛋白面板在各锚点评分之上的增量（推导集拟合、验证集评估）")
print("=" * 100)


def fit_cox(d: pd.DataFrame, cols: list[str]):
    frame = d[["time", "event", "center"] + cols].dropna()
    try:
        c = CoxPHFitter(penalizer=0.0)
        c.fit(frame, "time", "event", strata=["center"])
        return c, frame.index
    except Exception:
        c = CoxPHFitter(penalizer=0.0)
        c.fit(frame[["time", "event"] + cols], "time", "event")
        return c, frame.index


inc_rows = []
for anchor in ["MELD", "MELD_Na", "MELD3", "CLIF_OF"]:
    cols = [anchor] + P5
    dv = sets["dev"].dropna(subset=cols)
    va = sets["val"].dropna(subset=cols)
    if len(dv) < 50 or len(va) < 50:
        continue
    m_anchor, _ = fit_cox(dv, [anchor])
    m_full, _ = fit_cox(dv, cols)
    lp_a = np.log(m_anchor.predict_partial_hazard(va[[anchor]]).to_numpy())
    lp_f = np.log(m_full.predict_partial_hazard(va[cols]).to_numpy())
    va2 = va.copy()
    va2["_a"], va2["_f"] = lp_a, lp_f
    r = boot_delta_auc(va2, "_f", "_a")
    c_a = concordance_index(va.time, -lp_a, va.event)
    c_f = concordance_index(va.time, -lp_f, va.event)
    if r:
        print(f"  {LAB[anchor]:<10} + 5 蛋白: ΔAUC {r['delta']:+.3f} "
              f"({r['lo']:+.3f}~{r['hi']:+.3f}; P={r['p']:.3f})  "
              f"C {c_a:.3f}→{c_f:.3f}  n={r['n']} 事件={r['events']}")
        inc_rows.append(dict(anchor=LAB[anchor], delta_auc=r["delta"], lo=r["lo"], hi=r["hi"],
                             p=r["p"], c_anchor=c_a, c_full=c_f, n=r["n"], events=r["events"]))
pd.DataFrame(inc_rows).to_csv(OUT / "panel_increment_anchors.csv", index=False, encoding="utf-8-sig")

res = dict(rows=rows, deltas=deltas, population=pop, increment=inc_rows,
           note="所有评分由 score_engine.py 依原始文献公式重算；未填补缺失值。")
(OUT / "score_discrimination_strict.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("\n输出: table2_strict.csv / tableS_population.csv / panel_increment_anchors.csv / score_discrimination_strict.json")
