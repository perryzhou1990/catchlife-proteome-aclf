# -*- coding: utf-8 -*-
"""
审计：稿件现有对照 "CLIF-OF (4 organ)" vs 已发表 CLIF-C OF（6 器官，6–18 分）

发现
----
fig4_make.py 第 73-80 行的 of_sum 是 home-made 4 成分之和：
    of_liver: digitize(TB, [34,102,204,306]) + 1   → 取值 1–5
    of_kid  : digitize(Cr, [0.7,1.5,2.3,3.5]) + 1  → 取值 1–5
    of_coag : digitize(INR,[1.1,1.25,1.5,2.5]) + 1 → 取值 1–5
    of_circ : 5 - digitize(MAP, [60,65,70])        → 取值 2–5
即：只有 4 个器官（缺脑、呼吸）、每个器官 1–5 分（部分 2–5 分），总分不是
6–18，与已发表 CLIF-C OF 不可通约。审稿人若熟悉 CLIF-C 会直接质疑。

本脚本把两种口径放在同一拆分下比较，供决定是否替换。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
DEV_C = ["nfa", "xna", "rja", "dta", "xha"]


def main() -> None:
    clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
    sc = pd.read_csv(BASE / "analysis" / "output2" / "scores_by_patient.csv",
                     dtype={"id": str})

    # ---- 主表（与 fig4 完全同口径）----
    s90 = pd.to_numeric(clin["O90_2020.10更新后"], errors="coerce")
    event = s90.isin([1, 2]).astype(int).to_numpy()
    t1y = pd.to_numeric(clin["OS_1y_更新"], errors="coerce")
    time90 = np.minimum(t1y.fillna(90), 90).to_numpy()      # 与主流水线一致
    tb = pd.to_numeric(clin["TB(mg/dL)"], errors="coerce")
    inr = pd.to_numeric(clin["INR"], errors="coerce")
    cr = pd.to_numeric(clin["CR(mg/dL)"], errors="coerce")
    meld = (3.78 * np.log(tb.clip(lower=1)) + 11.2 * np.log(inr.clip(lower=1))
            + 9.57 * np.log(cr.clip(lower=1)) + 6.43).clip(6, 40).to_numpy()

    # ---- 稿件现有 home-made 4 器官版（原样复现）----
    tb_u = pd.to_numeric(clin["TB(umol/L)"], errors="coerce").fillna(17.1).to_numpy()
    cr_f = cr.fillna(cr.median()).to_numpy()
    inr_f = inr.fillna(inr.median()).to_numpy()
    map_f = pd.to_numeric(clin["MAP"], errors="coerce").fillna(70).to_numpy()
    of_sum = (np.digitize(tb_u, [34, 102, 204, 306]) + 1
              + np.digitize(cr_f, [0.7, 1.5, 2.3, 3.5]) + 1
              + np.digitize(inr_f, [1.1, 1.25, 1.5, 2.5]) + 1
              + (5 - np.digitize(map_f, [60, 65, 70])))

    # ---- 官方 CLIF-C OF（6 器官，6–18）----
    clif_of = sc["CLIF_OF"].to_numpy(dtype=float)

    center = clin["ID"].astype(str).str.split("-").str[0].to_numpy()   # 与 fig4 同口径
    dev = np.isin(center, DEV_C)
    val = ~dev

    rows = []
    for name, x in [("MELD", meld), ("CLIF-OF (4 organ, 稿件现有)", of_sum.astype(float)),
                    ("CLIF-C OF (6 organ, 已发表)", clif_of)]:
        r = {"model": name,
             "range": "%d-%d" % (np.nanmin(x), np.nanmax(x)),
             "median": round(float(np.nanmedian(x)), 1)}
        for lab, m in [("all", np.ones_like(dev)), ("dev", dev), ("val", val)]:
            xx, ee = x[m], event[m]
            ok = ~np.isnan(xx)
            r["C_" + lab] = round(float(concordance_index(
                time90[m][ok], -xx[ok], ee[ok])), 3)
            r["AUC90_" + lab] = round(float(roc_auc_score(ee[ok], xx[ok])), 3)
        rows.append(r)

    df = pd.DataFrame(rows)
    print("=" * 96)
    print("对照模型审计：稿件现有 4 器官版 vs 已发表 CLIF-C OF")
    print("=" * 96)
    print(df.to_string(index=False))

    out = BASE / "analysis" / "output2" / "clif_variant_audit.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n注意：C-index 用 -score 排序（分数越高越差），与主流水线方向一致。")
    print("输出：%s" % out)


if __name__ == "__main__":
    main()
