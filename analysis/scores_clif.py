# -*- coding: utf-8 -*-
"""
CLIF-C / COSSH-ACLF II 评分计算模块（为补做"领域标准评分对照"准备）

背景
----
v0.3 稿件的对照模型只有 MELD / MELD-Na / CLIF-OF-4，缺少 ACLF 领域默认的两个
预后评分：CLIF-C ACLF 与 COSSH-ACLF II。二者以及 CLIF-C AD 都需要"年龄"，
该字段正在申请获取。本脚本先把不依赖年龄的部分（CLIF-C OF，6 器官 6–18 分）
跑通并验证，年龄到位后只需放入 analysis/age.csv（列：ID, age）即可一次算出全部
四个评分。

公式出处（均已核对）
--------------------
CLIF-C OF（Jalan 2014, J Hepatol 61:1038；EF CLIF 官网评分器）
  每个器官 1–3 分，总分 6–18
CLIF-C ACLF（同上）
  = 10 × (0.33 × CLIF-C OF + 0.04 × age + 0.63 × ln(WBC ×10^9/L) - 2)
CLIF-C AD（Jalan 2015, J Hepatol 62:831；EF CLIF 官网）
  = 10 × (0.03 × age + 0.66 × ln(Cr mg/dL) + 1.71 × ln(INR)
          + 0.88 × ln(WBC ×10^9/L) - 0.05 × Na + 8)
  注意：不含白蛋白；钠为负系数；常数 +8。切勿用记忆版本。
COSSH-ACLF II（Li 2021, J Hepatol；2025 版中国 ACLF 指南附录）
  = 1.649 × ln(INR) + 0.457 × HE 评分(0级=1, I–II=2, III–IV=3)
    + 0.425 × ln(NEU ×10^9/L) + 0.396 × ln(TB μmol/L)
    + 0.576 × ln(BUN mmol/L) + 0.033 × age

用法
----
    python analysis/scores_clif.py                # 无年龄：只算 CLIF-C OF
    python analysis/scores_clif.py --age age.csv  # 有年龄：四个评分全算

输出
----
    analysis/output2/scores_by_patient.csv        逐例评分
    analysis/output2/scores_summary.json          汇总与验证统计
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

BASE = Path(__file__).resolve().parents[1]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
OUT = BASE / "analysis" / "output2"

COL = {
    "id": "ID",
    "tb_umol": "TB(umol/L)",
    "cr_umol": "CR(μmol/L)",
    "inr": "INR",
    "he": "病程有脑病及最大等级(0.no1.yes)_部分核实",
    "map": "MAP",
    "pf": "FIO_2",                       # 该列实为 PaO2/FiO2（上限 476.19 = 100/0.21）
    "wbc": "WBC",
    "neu": "neutrophil_number",
    "na": "Na",
    "alb": "ALB",
    "bun": "BUN",
    "o90": "O90_2020.10更新后",
    "o365": "O365_更新",
    "cossh_day": "COSSH_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳",
    "easl_day": "EASL_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳",
    "center": "Cohort",
}


def _num(s: pd.Series) -> pd.Series:
    """'-' / '2已核实' 等脏值统一转 NaN。"""
    return pd.to_numeric(s.replace({"-": np.nan, "—": np.nan}), errors="coerce")


# ---------------- CLIF-C OF：六个器官，各 1–3 分 ----------------
def of_liver(tb_mgdl: pd.Series) -> pd.Series:
    return np.select([tb_mgdl < 6.0, tb_mgdl < 12.0], [1, 2], default=3)


def of_kidney(cr_mgdl: pd.Series) -> pd.Series:
    # 无 RRT 字段：Cr ≥3.5 记 3 分（不额外给 RRT 加分，偏保守）
    return np.select([cr_mgdl < 2.0, cr_mgdl < 3.5], [1, 2], default=3)


def of_brain(he: pd.Series) -> pd.Series:
    return np.select([he == 0, he <= 2], [1, 2], default=3)


def of_coag(inr: pd.Series) -> pd.Series:
    return np.select([inr < 2.0, inr < 2.5], [1, 2], default=3)


def of_circ(map_: pd.Series) -> pd.Series:
    # 无升压药字段：MAP <70 记 2 分，不加 3 分（偏保守）
    return np.where(map_ >= 70, 1, 2)


def of_resp(pf: pd.Series) -> pd.Series:
    return np.select([pf > 400, pf > 300], [1, 2], default=3)


def build(age_path: Path | None) -> pd.DataFrame:
    clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
    df = pd.DataFrame({
        "id": clin[COL["id"]].astype(str),
        "center": clin[COL["center"]].astype(str),
        "tb_umol": _num(clin[COL["tb_umol"]]),
        "cr_umol": _num(clin[COL["cr_umol"]]),
        "inr": _num(clin[COL["inr"]]),
        "he": _num(clin[COL["he"]]),
        "map": _num(clin[COL["map"]]),
        "pf": _num(clin[COL["pf"]]),
        "wbc": _num(clin[COL["wbc"]]),
        "neu": _num(clin[COL["neu"]]),
        "na": _num(clin[COL["na"]]),
        "alb_gL": _num(clin[COL["alb"]]),
        "bun": _num(clin[COL["bun"]]),
        "o90": _num(clin[COL["o90"]]),
        "o365": _num(clin[COL["o365"]]),
        "cossh_day": _num(clin[COL["cossh_day"]]),
        "easl_day": _num(clin[COL["easl_day"]]),
    })

    # 90 天复合终点（与主流水线 reselect_panel.py 完全一致）
    df["event90"] = df["o90"].isin([1, 2]).astype(int)
    df["time90"] = 90.0  # 随访 90 天零缺失，事件时间不参与 C-index 排序以外的计算

    df["tb_mgdl"] = df["tb_umol"] / 17.1
    df["cr_mgdl"] = df["cr_umol"] / 88.4

    df["OF_liver"] = of_liver(df["tb_mgdl"])
    df["OF_kidney"] = of_kidney(df["cr_mgdl"])
    df["OF_brain"] = of_brain(df["he"])
    df["OF_coag"] = of_coag(df["inr"])
    df["OF_circ"] = of_circ(df["map"])
    df["OF_resp"] = of_resp(df["pf"])
    df["CLIF_OF"] = df[["OF_liver", "OF_kidney", "OF_brain",
                        "OF_coag", "OF_circ", "OF_resp"]].sum(axis=1)

    # 年龄
    df["age"] = np.nan
    if age_path is not None and Path(age_path).exists():
        ag = pd.read_csv(age_path, dtype={"ID": str})
        ag.columns = [c.strip().lower() for c in ag.columns]
        key = "id" if "id" in ag.columns else ag.columns[0]
        ag = ag.rename(columns={key: "id"})
        ag["id"] = ag["id"].astype(str).str.strip()
        df["id"] = df["id"].str.strip()
        df = df.merge(ag[["id", "age"]].drop_duplicates("id"), on="id",
                      how="left", suffixes=("", "_new"))
        df["age"] = df["age_new"].combine_first(df["age"]) if "age_new" in df else df["age"]
        df = df.drop(columns=[c for c in df.columns if c.endswith("_new")])

    have_age = df["age"].notna()

    # ---- 需要年龄的三个评分 ----
    ln_wbc = np.log(df["wbc"].where(df["wbc"] > 0))
    ln_inr = np.log(df["inr"].where(df["inr"] > 0))
    ln_cr = np.log(df["cr_mgdl"].where(df["cr_mgdl"] > 0))
    ln_neu = np.log(df["neu"].where(df["neu"] > 0))
    ln_tb = np.log(df["tb_umol"].where(df["tb_umol"] > 0))
    ln_bun = np.log(df["bun"].where(df["bun"] > 0))

    df["CLIF_ACLF"] = np.where(
        have_age,
        10 * (0.33 * df["CLIF_OF"] + 0.04 * df["age"] + 0.63 * ln_wbc - 2), np.nan)
    df["CLIF_AD"] = np.where(
        have_age,
        10 * (0.03 * df["age"] + 0.66 * ln_cr + 1.71 * ln_inr
              + 0.88 * ln_wbc - 0.05 * df["na"] + 8), np.nan)
    he_score = np.select([df["he"] == 0, df["he"] <= 2], [1, 2], default=3)
    df["COSSH_II"] = np.where(
        have_age,
        1.649 * ln_inr + 0.457 * he_score + 0.425 * ln_neu
        + 0.396 * ln_tb + 0.576 * ln_bun + 0.033 * df["age"], np.nan)

    return df


def summarise(df: pd.DataFrame) -> dict:
    res: dict = {"n": int(len(df)), "events90": int(df["event90"].sum())}

    # 完整度
    res["completeness"] = {
        c: int(df[c].notna().sum()) for c in
        ["tb_mgdl", "cr_mgdl", "inr", "he", "map", "pf", "wbc", "neu", "na", "bun", "age"]
    }

    # CLIF-C OF 分布
    res["CLIF_OF"] = {
        "median": float(df["CLIF_OF"].median()),
        "q1": float(df["CLIF_OF"].quantile(.25)),
        "q3": float(df["CLIF_OF"].quantile(.75)),
        "min": int(df["CLIF_OF"].min()),
        "max": int(df["CLIF_OF"].max()),
    }

    # 分器官得分构成（各分值占比，用于发现"某器官几乎不变"的退化）
    res["subscore_variance"] = {
        k: {int(v): int(c) for v, c in df["OF_" + k].value_counts().sort_index().items()}
        for k in ["liver", "kidney", "brain", "coag", "circ", "resp"]
    }

    # 按 COSSH / EASL ACLF 状态分层
    res["by_cossh"] = {}
    for k, sub in df.groupby("cossh_day"):
        res["by_cossh"][str(int(k)) if not np.isnan(k) else "NA"] = {
            "n": int(len(sub)),
            "CLIF_OF_median": float(sub["CLIF_OF"].median()),
            "death90": round(float(sub["event90"].mean()) * 100, 1),
        }
    res["by_easl"] = {}
    for k, sub in df.groupby("easl_day"):
        res["by_easl"][str(int(k)) if not np.isnan(k) else "NA"] = {
            "n": int(len(sub)),
            "CLIF_OF_median": float(sub["CLIF_OF"].median()),
            "death90": round(float(sub["event90"].mean()) * 100, 1),
        }

    # 各评分的 C-index（90 天死亡）
    scores = ["CLIF_OF", "CLIF_ACLF", "CLIF_AD", "COSSH_II"]
    res["cindex_all"] = {}
    res["cindex_aclF"] = {}
    aclf = df["cossh_day"].isin([1, 2])
    for s in scores:
        d = df[[s, "event90"]].dropna()
        if len(d) > 20 and d["event90"].sum() > 5:
            res["cindex_all"][s] = round(float(
                concordance_index(np.ones(len(d)), -d[s], d["event90"])), 4)
        d2 = df.loc[aclf, [s, "event90"]].dropna()
        if len(d2) > 20 and d2["event90"].sum() > 5:
            res["cindex_aclF"][s] = round(float(
                concordance_index(np.ones(len(d2)), -d2[s], d2["event90"])), 4)

    res["age_available"] = bool(df["age"].notna().any())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--age", default=None,
                    help="CSV with columns ID,age (患者年龄)；缺省则只算 CLIF-C OF")
    args = ap.parse_args()

    df = build(Path(args.age) if args.age else None)
    OUT.mkdir(parents=True, exist_ok=True)

    keep = ["id", "center", "age", "event90", "cossh_day", "easl_day",
            "tb_mgdl", "cr_mgdl", "inr", "he", "map", "pf", "wbc", "neu", "na", "alb_gL", "bun",
            "OF_liver", "OF_kidney", "OF_brain", "OF_coag", "OF_circ", "OF_resp",
            "CLIF_OF", "CLIF_ACLF", "CLIF_AD", "COSSH_II"]
    df[keep].to_csv(OUT / "scores_by_patient.csv", index=False, encoding="utf-8-sig")

    res = summarise(df)
    (OUT / "scores_summary.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 66)
    print("CLIF-C OF  CLIF-C ACLF  CLIF-C AD  COSSH-ACLF II  —— 计算与验证")
    print("=" * 66)
    print("n = %d | 90 天死亡 %d 例 (%.1f%%)"
          % (res["n"], res["events90"], 100 * res["events90"] / res["n"]))
    print("\n字段完整度：")
    for k, v in res["completeness"].items():
        flag = "" if v == res["n"] else "   <-- 缺失 %d" % (res["n"] - v)
        print("  %-8s %4d / %d%s" % (k, v, res["n"], flag))

    o = res["CLIF_OF"]
    print("\nCLIF-C OF：中位 %.0f（IQR %.0f–%.0f），范围 %d–%d"
          % (o["median"], o["q1"], o["q3"], o["min"], o["max"]))
    print("分器官分值分布（1/2/3 分各多少例）：")
    names = {"liver": "肝", "kidney": "肾", "brain": "脑",
             "coag": "凝血", "circ": "循环", "resp": "呼吸"}
    for k, v in res["subscore_variance"].items():
        tot = sum(v.values())
        print("  %-4s %s   （3 分占 %.1f%%）"
              % (names[k], " | ".join("%d分:%d" % (a, b) for a, b in v.items()),
                 100 * v.get(3, 0) / tot if tot else 0))

    print("\n按 COSSH 标准分层：")
    for k, v in res["by_cossh"].items():
        lab = {"0": "non-ACLF", "1": "ACLF 入院时", "2": "院内新发"}.get(k, k)
        print("  %-12s n=%4d  CLIF-C OF 中位 %.0f  90 天死亡 %.1f%%"
              % (lab, v["n"], v["CLIF_OF_median"], v["death90"]))

    print("\nC-index（90 天死亡）：")
    print("  全队列   ", res["cindex_all"] or "（需要年龄，暂缺）")
    print("  ACLF 亚组", res["cindex_aclF"] or "（需要年龄，暂缺）")

    if not res["age_available"]:
        print("\n[年龄未提供] CLIF-C ACLF / CLIF-C AD / COSSH-ACLF II 未计算。")
        print("  获取后：python analysis/scores_clif.py --age analysis/age.csv")
        print("  age.csv 需两列：ID, age（ID 与主表一致）")
    print("\n输出：analysis/output2/scores_by_patient.csv, scores_summary.json")


if __name__ == "__main__":
    main()
