# -*- coding: utf-8 -*-
"""第二篇选题可行性试算（二）：883 蛋白中是否存在超越 MELD 的"ACLF 发生"预测信号。

若存在 → 方向一可重做（重新选蛋白，而非借用死亡模型系数）。
若不存在 → 方向一应放弃，避免三个月后才发现天花板。

设计：推导 5 中心内，逐蛋白 logistic（校正 MELD + sex + 中心固定效应），
BH-FDR 控制；随后用验证 8 中心的外样本 AUC 增益判断是否可复现。
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
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
proteins = list(Z.columns)
Zarr = Z.to_numpy()

cossh = pd.to_numeric(clin["COSSH_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"], errors="coerce")
o90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
inr_raw = pd.to_numeric(clin["INR"], errors="coerce")
cr_raw = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce")
inr = inr_raw.fillna(inr_raw.median()); cr = cr_raw.fillna(cr_raw.median())
meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.clip(lower=1))
        + 9.57 * np.log(cr.clip(lower=1)) + 6.43).clip(6, 40).to_numpy()
sex = clin["Gender"].replace(2, np.nan)
sex = sex.fillna(sex.mode()[0]).astype(float).to_numpy()

keep = cossh.isin([0, 2]).to_numpy()
y_all = (cossh == 2).astype(int).to_numpy()
zz = Zarr[keep]
yy = y_all[keep]
mel = meld[keep]
sx = sex[keep]
cen = clin["center"].to_numpy()[keep]

DEV = ["nfa", "xna", "rja", "dta", "xha"]
dev = np.isin(cen, DEV)

print("=" * 70)
print(f"入院无 ACLF: n={keep.sum()}  新发={yy.sum()}")
print(f"  推导 5 中心: n={dev.sum()} 新发={yy[dev].sum()}")
print(f"  验证 8 中心: n={(~dev).sum()} 新发={yy[~dev].sum()}")
print("=" * 70)

# ---- 逐蛋白 logistic（校正 MELD + sex + 中心哑变量），仅推导集 ----
centres = sorted(set(cen[dev]))
D = np.column_stack([mel[dev], sx[dev]] + [(cen[dev] == c).astype(float) for c in centres[1:]])
Xd = zz[dev]
yd = yy[dev]

ps = np.ones(len(proteins))
betas = np.zeros(len(proteins))
for j in range(len(proteins)):
    X = np.column_stack([Xd[:, j], D])
    try:
        m = LogisticRegression(max_iter=400, penalty=None).fit(X, yd)
        # Wald: 用 fisher information 近似标准误
        p1 = m.predict_proba(X)[:, 1]
        W = p1 * (1 - p1)
        XtWX = X.T @ (X * W[:, None])
        cov = np.linalg.pinv(XtWX)
        se = np.sqrt(cov[0, 0])
        betas[j] = m.coef_[0][0]
        ps[j] = 2 * stats.norm.sf(abs(betas[j] / se))
    except Exception:
        ps[j] = 1.0

# BH-FDR
order = np.argsort(ps)
m = len(ps)
q = np.empty(m)
prev = 1.0
for rank, idx in enumerate(order[::-1], start=1):
    k = m - rank + 1
    prev = min(prev, ps[idx] * m / k)
    q[idx] = prev

res = pd.DataFrame({"protein": proteins, "beta": betas, "p": ps, "FDR": q})
res = res.sort_values("FDR")
n_sig = int((res.FDR < 0.05).sum())
n_nom = int((res.p < 0.05).sum())
print(f"\n逐蛋白（校正 MELD+sex+中心）在推导集的显著性：")
print(f"  FDR<0.05: {n_sig} 个 | 名义 P<0.05: {n_nom} 个")
print("\n前 12 名：")
print(res.head(12).to_string(index=False))

# ---- 外样本检验：取 FDR 前 10 名的简单平均 z 分，看验证集能否加分 ----
top = res.head(10).protein.tolist()
idx = [proteins.index(p) for p in top]
score = zz[:, idx].mean(1)

if yy[~dev].sum() > 0:
    print(f"\n验证 8 中心（n={(~dev).sum()}，新发 {yy[~dev].sum()}）")
    print(f"  前10蛋白均分 单变量 AUC = {roc_auc_score(yy[~dev], score[~dev]):.3f}")
    print(f"  MELD                AUC = {roc_auc_score(yy[~dev], mel[~dev]):.3f}")
    # MELD + score 联合，在推导集拟合
    Xtr = np.column_stack([mel[dev], score[dev]])
    Xte = np.column_stack([mel[~dev], score[~dev]])
    mu, sd = Xtr.mean(0), Xtr.std(0, ddof=1)
    m1 = LogisticRegression(max_iter=2000).fit(((Xtr - mu) / sd)[:, [0]], yd)
    m2 = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, yd)
    a1 = roc_auc_score(yy[~dev], m1.predict_proba(((Xte - mu) / sd)[:, [0]])[:, 1])
    a2 = roc_auc_score(yy[~dev], m2.predict_proba((Xte - mu) / sd)[:, 1])
    print(f"  MELD 单用           AUC = {a1:.3f}")
    print(f"  MELD + 前10蛋白均分 AUC = {a2:.3f}   ΔAUC = {a2 - a1:+.3f}")

res.to_csv("analysis/output_second/incident_aclf_screen.csv", index=False)
print("\n已保存 analysis/output_second/incident_aclf_screen.csv")
