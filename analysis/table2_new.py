# -*- coding: utf-8 -*-
"""
重建 Table 2：所有评分按同一方法评估
- 既有行（MELD / MELD-Na / CLIF-C OF / 四个面板模型）沿用 fig4 流水线的变量定义，
  保证与 Fig 4B 完全一致；
- 新增行（MELD 3.0 / COSSH-ACLF II）用 score_engine.py 严格重算，在各自完整病例上
  拟合单变量 Cox（推导集，按中心分层）并施加于验证集；
- Δ 值一律在同一批配对病例上做 2,000 次 bootstrap（百分位法）。

输出：analysis/output2/table2_v2.csv / .md、tableS13_population.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "analysis" / "output2"
SRC = BASE / "figures" / "fig4" / "code" / "fig4_make.py"
text = SRC.read_text(encoding="utf-8")
ns = {"__name__": "__panel_data__", "__file__": str(SRC)}
exec(compile(text[: text.index("# ---------------- figure ----------------")], str(SRC), "exec"), ns)
clin, P5, P3 = ns["clin"], ns["P5"], ns["P3"]
base = ns["base"].copy()
base["id"] = clin["ID"].astype(str).str.strip().to_numpy()
base["sex"] = ns["base"]["sex"].to_numpy()

sc = pd.read_csv(OUT / "scores_strict_1390.csv")
sc["id"] = sc["id"].astype(str).str.strip()
keep = ["id", "age", "event90", "time90", "cossh_day", "aclf_any", "hbv_aclf",
        "MELD", "MELD_Na", "MELD3", "CLIF_OF", "CLIF_AD", "CLIF_ACLF", "COSSH_II", "CTP"]
df = base.merge(sc[keep], on="id", how="left")

# 1 年结局：与流水线完全一致，用 O365_更新（0 存活, 1 死亡, 2 移植, 4 失访→剔除）
df["o365"] = pd.to_numeric(clin["O365_更新"], errors="coerce").to_numpy()

DEV_C = ["nfa", "xna", "rja", "dta", "xha"]
dev = df[df.center.isin(DEV_C)].reset_index(drop=True)
val = df[~df.center.isin(DEV_C)].reset_index(drop=True)
SEED = 20261108


def fit(df_, cols):
    frame = df_[["time", "event", "center"] + cols].dropna()
    try:
        c = CoxPHFitter(penalizer=0.0)
        c.fit(frame, "time", "event", strata=["center"])
        return c, frame
    except Exception:
        c = CoxPHFitter(penalizer=0.0)
        c.fit(frame[["time", "event"] + cols], "time", "event")
        return c, frame


def baseline_h(df_, cph, t):
    b = cph.baseline_cumulative_hazard_
    rows = b.index[b.index <= t]
    if not len(rows):
        return 0.0
    h = {s: float(b[s].loc[rows[-1]]) for s in b.columns}
    w = pd.Series(df_["center"]).value_counts(normalize=True)
    return float(sum(h.get(s, 0.0) * w.get(s, 0.0) for s in w.index))


def metrics(val_, dev_, cols, label):
    """拟合 -> 验证集上的 C-index / AUC90 / AUC1y / Brier90。"""
    c, fitted = fit(dev_, cols)
    v = val_.dropna(subset=cols + ["time", "event"])
    lp = np.log(c.predict_partial_hazard(v[cols]).to_numpy())
    cidx = concordance_index(v["time"], -lp, v["event"])
    ok90 = (v["time"] >= 90) | (v["event"] == 1)
    auc90 = roc_auc_score(v.loc[ok90, "event"], lp[ok90.to_numpy()]) if v.loc[ok90, "event"].sum() > 5 else np.nan
    y1 = v["o365"].isin([1, 2])
    ok1y = v["o365"].isin([0, 1, 2]).to_numpy()
    auc1y = roc_auc_score(y1[ok1y], lp[ok1y]) if y1[ok1y].sum() > 5 else np.nan
    h0 = baseline_h(fitted, c, 90)
    risk = 1 - np.exp(-h0 * np.exp(lp))
    brier = float(np.mean((v["event"].to_numpy() - risk) ** 2))
    c_dev = concordance_index(fitted["time"], -np.log(c.predict_partial_hazard(fitted[cols]).to_numpy()),
                              fitted["event"])
    lp_full = pd.Series(np.nan, index=val_.index)
    lp_full.loc[v.index] = lp          # 与 val_ 行对齐，缺失处为 NaN
    return dict(model=label, n=len(v), events=int(v["event"].sum()), c_dev=c_dev, c_val=cidx,
                auc90=auc90, auc1y=auc1y, brier=brier), lp_full.to_numpy()


def fast_auc(y, x):
    """秩统计法 AUC（Mann-Whitney），比 sklearn 快一个量级，用于 bootstrap。"""
    from scipy.stats import rankdata
    y = np.asarray(y, float)
    r = rankdata(x)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired_delta_c(v, lp_a, lp_b, B=2000):
    """同一批病例上的 C-index 配对差（2000 次 bootstrap，百分位 CI）。

    C-index 用到完整随访（不设 90 天界标），故与 AUC 的配对函数分开。
    """
    t = v["time"].to_numpy()
    y = v["event"].to_numpy()
    keep = np.isfinite(lp_a) & np.isfinite(lp_b)
    t, y, a, b = t[keep], y[keep], lp_a[keep], lp_b[keep]
    n = len(y)
    if n < 50 or y.sum() < 10:
        return None
    point = float(concordance_index(t, -a, y) - concordance_index(t, -b, y))
    rng = np.random.default_rng(SEED)      # 每个对比独立同种子：可复现，不受调用顺序影响
    draws = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() < 5 or y[i].sum() > n - 5:
            continue
        draws.append(concordance_index(t[i], -a[i], y[i]) - concordance_index(t[i], -b[i], y[i]))
    v_ = np.array(draws)
    lo, hi = np.percentile(v_, [2.5, 97.5])
    p = 2 * min((v_ <= 0).mean(), (v_ >= 0).mean())
    return dict(delta=point, lo=float(lo), hi=float(hi), p=float(min(p, 1.0)), n=int(n))


def paired_delta_auc(v, lp_a, lp_b, B=2000):
    """同一批病例上的 90 天 AUC 配对差（2000 次 bootstrap，百分位 CI）。"""
    ok = ((v["time"] >= 90) | (v["event"] == 1)).to_numpy()
    y = v["event"].to_numpy()[ok]
    a, b = lp_a[ok], lp_b[ok]
    n = len(y)
    if n < 50 or y.sum() < 10:
        return None
    point = fast_auc(y, a) - fast_auc(y, b)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() < 5 or y[i].sum() > n - 5:
            continue
        draws.append(fast_auc(y[i], a[i]) - fast_auc(y[i], b[i]))
    v_ = np.array(draws)
    lo, hi = np.percentile(v_, [2.5, 97.5])
    p = 2 * min((v_ <= 0).mean(), (v_ >= 0).mean())
    return dict(delta=float(point), lo=float(lo), hi=float(hi), p=float(min(p, 1.0)), n=int(n))


# --------------------------------------------------------------------------- #
rows, lps = {}, {}
rows["MELD"], lps["MELD"] = metrics(val, dev, ["meld"], "MELD")
rows["MELD-Na"], lps["MELD-Na"] = metrics(val, dev, ["meldna"], "MELD-Na")
rows["CLIF-C OF"], lps["CLIF-C OF"] = metrics(val, dev, ["of_sum"], "CLIF-C OF")
rows["MELD 3.0"], lps["MELD 3.0"] = metrics(val, dev, ["MELD3"], "MELD 3.0")
rows["COSSH-ACLF II"], lps["COSSH-ACLF II"] = metrics(val, dev, ["COSSH_II"], "COSSH-ACLF II")
rows["MELD + 5-protein panel"], lps["MELD + 5-protein panel"] = metrics(val, dev, ["meld"] + P5, "MELD+5P")
rows["MELD + 3-protein panel"], lps["MELD + 3-protein panel"] = metrics(val, dev, ["meld"] + P3, "MELD+3P")
rows["5-protein panel alone"], lps["5-protein panel alone"] = metrics(val, dev, P5, "5P")
rows["3-protein panel alone"], lps["3-protein panel alone"] = metrics(val, dev, P3, "3P")

# 与 MELD 的配对差值（只在两评分都可得、且 90 天可评估的病例上）
ref = lps["MELD"]
tbl = []
for lab, r in rows.items():
    lp = lps[lab]
    if lab == "MELD":
        d_auc = d_c = None
    else:
        ok = np.isfinite(lp) & np.isfinite(ref)
        vv = val[ok].reset_index(drop=True)
        a = lp[ok]
        b = ref[ok]
        d_c = paired_delta_c(vv, a, b)
        d_auc = paired_delta_auc(vv, a, b)
    tbl.append(dict(row=lab, **r,
                    dC=None if d_c is None else d_c["delta"],
                    dC_lo=None if d_c is None else d_c["lo"],
                    dC_hi=None if d_c is None else d_c["hi"],
                    dC_p=None if d_c is None else d_c["p"],
                    dAUC=None if d_auc is None else d_auc["delta"],
                    dAUC_lo=None if d_auc is None else d_auc["lo"],
                    dAUC_hi=None if d_auc is None else d_auc["hi"],
                    dAUC_p=None if d_auc is None else d_auc["p"],
                    d_n=None if d_auc is None else d_auc["n"]))
T = pd.DataFrame(tbl)
T.to_csv(OUT / "table2_v2.csv", index=False, encoding="utf-8-sig")

print("=" * 118)
print(f"{'model':<28}{'n':>5}{'ev':>5}{'C dev':>8}{'C val':>8}{'dC':>7}{'AUC90':>8}{'dAUC (95%CI)':>26}{'P':>7}{'AUC1y':>8}{'Brier':>7}")
print("=" * 118)
for r in tbl:
    dauc = "—" if r["dAUC"] is None else f"{r['dAUC']:+.3f} ({r['dAUC_lo']:+.3f}~{r['dAUC_hi']:+.3f})"
    dc = "—" if r["dC"] is None else f"{r['dC']:+.3f}"
    p = "—" if r["dAUC_p"] is None else f"{r['dAUC_p']:.3f}"
    print(f"{r['row']:<28}{r['n']:>5}{r['events']:>5}{r['c_dev']:>8.3f}{r['c_val']:>8.3f}"
          f"{dc:>7}{r['auc90']:>8.3f}{dauc:>26}{p:>7}{r['auc1y']:>8.3f}{r['brier']:>7.3f}")

# --------------------------------------------------------------------------- #
# Table S13：人群特异评分 + 面板在该人群内
# --------------------------------------------------------------------------- #
print("\n" + "=" * 110)
print("Table S13 人群特异比较（全队列；面板线性预测值取自 MELD+5P 模型，不重新拟合）")
print("=" * 110)
m5 = fit(dev, ["meld"] + P5)[0]
df["panel_lp"] = np.log(m5.predict_partial_hazard(df[["meld"] + P5]).to_numpy())

pop_rows = []
for name, mask, cols in [
    ("No ACLF at any time", df.aclf_any == False, ["CLIF_AD", "MELD", "MELD 3.0", "CTP"]),
    ("ACLF (admission or in-hospital)", df.aclf_any == True, ["CLIF_ACLF", "COSSH-ACLF II", "MELD", "MELD 3.0"]),
    ("HBV-ACLF", df.hbv_aclf == True, ["COSSH-ACLF II", "CLIF_ACLF", "MELD", "MELD 3.0"]),
]:
    s = df[mask].copy()
    s["COSSH-ACLF II"] = s["COSSH_II"]
    s["MELD 3.0"] = s["MELD3"]
    s["CLIF-C AD"] = s["CLIF_AD"]
    s["CLIF-C ACLF"] = s["CLIF_ACLF"]
    s["panel_lp"] = s["panel_lp"]
    print(f"\n{name}: n={len(s)}, 90-d events={int(s.event.sum())} ({s.event.mean()*100:.1f}%)")
    for col in cols + ["panel_lp"]:
        ok = s[col].notna() & ((s.time >= 90) | (s.event == 1))
        if ok.sum() < 40 or s.loc[ok, "event"].sum() < 10:
            continue
        a = roc_auc_score(s.loc[ok, "event"], s.loc[ok, col])
        c = concordance_index(s.loc[ok, "time"], -s.loc[ok, col], s.loc[ok, "event"])
        lab = "5-protein panel" if col == "panel_lp" else col
        print(f"   {lab:<18} AUC90={a:.3f}  C-index={c:.3f}  n={int(ok.sum())}  事件={int(s.loc[ok,'event'].sum())}")
        pop_rows.append(dict(population=name, score=lab, auc90=a, cindex=c,
                             n=int(ok.sum()), events=int(s.loc[ok, "event"].sum())))
pd.DataFrame(pop_rows).to_csv(OUT / "tableS13_population.csv", index=False, encoding="utf-8-sig")
print("\n输出: table2_v2.csv, tableS13_population.csv")
