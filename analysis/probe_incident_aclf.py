# -*- coding: utf-8 -*-
"""第二篇选题可行性试算：五蛋白 panel 能否预测院内新发 ACLF。

人群：入院时无 ACLF（COSSH_DAY = 0 或 2），n=1,178。
终点：incident ACLF（COSSH_DAY = 2），60 例；复合 ACLF-free 终点 109 例。
目的：在决定立项之前，用真实数据回答"有没有信号、信号多大"。
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\proteinarray_matrix_1390人蛋白质组结果（质控后883蛋白）.xlsx")

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

cossh = pd.to_numeric(clin["COSSH_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"],
                      errors="coerce")
o90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr_raw = pd.to_numeric(clin["INR"], errors="coerce")
cr_raw = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce")
inr = inr_raw.fillna(inr_raw.median())
cr = cr_raw.fillna(cr_raw.median())
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.clip(lower=1))
        + 9.57 * np.log(cr.clip(lower=1)) + 6.43).clip(6, 40).to_numpy()

# 五蛋白线性预测子（系数 = 推导集多变量 Cox 的 log-HR）
BETA = {"TNF.RI": 0.29468, "DR6": 0.50945, "Renin": 0.28793,
        "SREC.II": -0.24673, "GH": -0.28346}
lp = np.zeros(len(clin))
for p, b in BETA.items():
    lp += b * Zarr[:, zcol[p]]

base = pd.DataFrame({
    "id": clin["ID"], "center": clin["center"], "cossh": cossh,
    "o90": o90.isin([1, 2]).astype(int), "meld": meld, "panel_lp": lp,
})
for p in BETA:
    base[p] = Zarr[:, zcol[p]]

adm_free = base[base.cossh.isin([0, 2])].copy()
adm_free["incident"] = (adm_free.cossh == 2).astype(int)
adm_free["aclf_free"] = ((adm_free.cossh == 2) | (adm_free.o90 == 1)).astype(int)

print("=" * 66)
print("人群：入院无 ACLF  n=%d" % len(adm_free))
print("  院内新发 ACLF  %d (%.1f%%)" % (adm_free.incident.sum(), adm_free.incident.mean() * 100))
print("  ACLF-free 复合 %d (%.1f%%)" % (adm_free.aclf_free.sum(), adm_free.aclf_free.mean() * 100))
print("=" * 66)

DEV = ["nfa", "xna", "rja", "dta", "xha"]


def auc_block(df, tag):
    if df.incident.nunique() < 2:
        print(f"{tag}: 单类别，跳过")
        return
    print(f"\n[{tag}]  n={len(df)}  新发={int(df.incident.sum())}")
    for name, col in [("panel (5 蛋白 LP)", "panel_lp"), ("MELD", "meld")]:
        a = roc_auc_score(df.incident, df[col])
        print(f"    {name:22s} AUC = {a:.3f}")
    # MELD + panel 的联合 logistic（推导集拟合，本集评估）
    return


auc_block(adm_free, "全队列")
auc_block(adm_free[adm_free.center.isin(DEV)], "推导 5 中心")
auc_block(adm_free[~adm_free.center.isin(DEV)], "验证 8 中心")

# 联合模型：推导集拟合 logistic，验证集评估
dev = adm_free[adm_free.center.isin(DEV)].copy()
val = adm_free[~adm_free.center.isin(DEV)].copy()
X_cols = ["meld", "panel_lp"]
Xs = adm_free[X_cols].to_numpy()
mu, sd = Xs.mean(0), Xs.std(0, ddof=1)


def fit_eval(tr, te, cols):
    Xtr = ((tr[cols].to_numpy() - mu[: len(cols)]) / sd[: len(cols)])
    Xte = ((te[cols].to_numpy() - mu[: len(cols)]) / sd[: len(cols)])
    m = LogisticRegression(max_iter=2000).fit(Xtr, tr.incident)
    return m, roc_auc_score(te.incident, m.predict_proba(Xte)[:, 1])


if dev.incident.nunique() > 1 and val.incident.nunique() > 1:
    _, auc_meld = fit_eval(dev, val, ["meld"])
    m2, auc_joint = fit_eval(dev, val, ["meld", "panel_lp"])
    print("\n[验证集 8 中心] 联合 logistic（推导集拟合）")
    print(f"    MELD only          AUC = {auc_meld:.3f}")
    print(f"    MELD + panel       AUC = {auc_joint:.3f}   ΔAUC = {auc_joint - auc_meld:+.3f}")
    print("    系数:", dict(zip(["MELD", "panel"], np.round(m2.coef_[0], 3))))

# 按 CLIF_OF 分层的发生率（严重度梯度）
print("\n[新发 ACLF 发生率 by MELD 三分位]")
adm_free["meld_t"] = pd.qcut(adm_free.meld, 3, labels=["T1", "T2", "T3"])
print(adm_free.groupby("meld_t", observed=True)["incident"].agg(["size", "sum", "mean"]).round(3).to_string())

print("\n[新发 ACLF 发生率 by panel LP 三分位]")
adm_free["lp_t"] = pd.qcut(adm_free.panel_lp, 3, labels=["T1", "T2", "T3"])
print(adm_free.groupby("lp_t", observed=True)["incident"].agg(["size", "sum", "mean"]).round(3).to_string())

# 保存供后续使用
adm_free.to_csv("analysis/output_second/adm_free_cohort.csv", index=False)
print("\n已保存 analysis/output_second/adm_free_cohort.csv")
