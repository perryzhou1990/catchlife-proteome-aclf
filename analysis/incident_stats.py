# -*- coding: utf-8 -*-
"""第二篇探针的最终统计：incident-ACLF 阴性的置信区间与可排除效应量。

写入正文 §5.8 之前必须先有 CI —— 只写"没有信号"会被审稿人以 power 不足驳回；
必须写明"可排除多大的效应"。全部数字由本脚本产出，不手工录入。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\proteinarray_matrix_1390人蛋白质组结果（质控后883蛋白）.xlsx")
RNG = np.random.default_rng(20261108)

clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
clin.columns = [str(c).strip() for c in clin.columns]
clin["center"] = clin["ID"].astype(str).str.split("-").str[0]

prot = pd.read_excel(PROT, sheet_name="1390已测蛋白质组")
prot.columns = [str(c).strip() for c in prot.columns]
prot = prot.set_index(prot.columns[0]).loc[clin["ID"]].astype(float)
log2 = np.log2(prot.clip(lower=1))
Z = (log2 - log2.mean()) / log2.std(ddof=1)
zcol = {c: i for i, c in enumerate(Z.columns)}
Zarr = Z.to_numpy()

cossh = pd.to_numeric(clin["COSSH_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"], errors="coerce")
o90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr = pd.to_numeric(clin["INR"], errors="coerce"); inr = inr.fillna(inr.median())
cr = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce"); cr = cr.fillna(cr.median())
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.clip(lower=1))
        + 9.57 * np.log(cr.clip(lower=1)) + 6.43).clip(6, 40).to_numpy()
sex = clin["Gender"].replace(2, np.nan); sex = sex.fillna(sex.mode()[0]).astype(float).to_numpy()

BETA = {"TNF.RI": 0.29468, "DR6": 0.50945, "Renin": 0.28793, "SREC.II": -0.24673, "GH": -0.28346}
lp = sum(b * Zarr[:, zcol[p]] for p, b in BETA.items())

keep = cossh.isin([0, 2]).to_numpy()
inc = (cossh == 2).astype(int).to_numpy()[keep]
comp = ((cossh == 2) | o90.isin([1, 2])).astype(int).to_numpy()[keep]
mel = meld[keep]; lpv = lp[keep]; cen = clin["center"].to_numpy()[keep]
DEV = ["nfa", "xna", "rja", "dta", "xha"]
dev = np.isin(cen, DEV)


def boot_auc(y, s, n=3000):
    a = roc_auc_score(y, s)
    idx = RNG.integers(0, len(y), (n, len(y)))
    bs = []
    for b in idx:
        if y[b].min() == y[b].max():
            continue
        bs.append(roc_auc_score(y[b], s[b]))
    return a, np.percentile(bs, [2.5, 97.5])


def boot_delta(y, s_meld, s_joint, n=3000):
    d = roc_auc_score(y, s_joint) - roc_auc_score(y, s_meld)
    idx = RNG.integers(0, len(y), (n, len(y)))
    bs = []
    for b in idx:
        yb = y[b]
        if yb.min() == yb.max():
            continue
        bs.append(roc_auc_score(yb, s_joint[b]) - roc_auc_score(yb, s_meld[b]))
    return d, np.percentile(bs, [2.5, 97.5])


out = {}
for tag, y in [("incident", inc), ("composite", comp)]:
    print(f"\n{'='*60}\n终点 = {tag}: n={len(y)} 事件={y.sum()}")
    rec = {"n": int(len(y)), "events": int(y.sum())}

    for lab, s in [("panel", lpv), ("meld", mel)]:
        a, ci = boot_auc(y[~dev], s[~dev])
        rec[lab] = {"auc": round(a, 3), "ci": [round(x, 3) for x in ci]}
        print(f"  验证集 {lab:6s} AUC {a:.3f} ({ci[0]:.3f}–{ci[1]:.3f})")

    # 联合模型：推导集拟合
    Xtr = np.column_stack([mel[dev], lpv[dev]])
    Xte = np.column_stack([mel[~dev], lpv[~dev]])
    mu, sd = Xtr.mean(0), Xtr.std(0, ddof=1)
    m1 = LogisticRegression(max_iter=2000).fit(((Xtr - mu) / sd)[:, [0]], y[dev])
    m2 = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, y[dev])
    p1 = m1.predict_proba(((Xte - mu) / sd)[:, [0]])[:, 1]
    p2 = m2.predict_proba((Xte - mu) / sd)[:, 1]
    d, ci = boot_delta(y[~dev], p1, p2)
    rec["delta"] = {"d": round(d, 3), "ci": [round(x, 3) for x in ci], "beta_panel": round(m2.coef_[0][1], 3)}
    print(f"  ΔAUC(联合−MELD) {d:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f})  panel beta={m2.coef_[0][1]:.3f}")

    # 按 MELD 三分位的新发率
    q = pd.qcut(mel, 3, labels=False)
    rec["by_meld"] = [{"n": int((q == k).sum()), "ev": int(y[q == k].sum())} for k in range(3)]
    print("  MELD 三分位事件率:", [f"{v['ev']}/{v['n']}" for v in rec["by_meld"]])
    out[tag] = rec

# 新发 ACLF 的死亡率对照
adm = cossh == 1
new = cossh == 2
o = o90.isin([1, 2])
out["mortality"] = {
    "on_admission": [int(o[adm].sum()), int(adm.sum())],
    "incident": [int(o[new].sum()), int(new.sum())],
    "never": [int(o[(cossh == 0)].sum()), int((cossh == 0).sum())],
}
print(f"\n入院即 ACLF 死亡 {out['mortality']['on_admission'][0]}/{out['mortality']['on_admission'][1]}")
print(f"院内新发     死亡 {out['mortality']['incident'][0]}/{out['mortality']['incident'][1]}")
print(f"始终无       死亡 {out['mortality']['never'][0]}/{out['mortality']['never'][1]}")

Path(BASE / "analysis" / "output_second" / "incident_stats.json").write_text(
    json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("\n已保存 analysis/output_second/incident_stats.json")
