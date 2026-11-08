# -*- coding: utf-8 -*-
"""
逐例核验「catch-life队列蛋白质组患者1390人-年龄-疾病评分.xlsx」中的既有评分列

目的
----
拿到年龄后，用原始化验值 + 年龄重算全部评分，与表格里已有的列逐例比对，
判断哪些列可信、哪些列用了错误口径（未下限截断 / 未封顶 / 常数写错）。

比对口径（每个评分都试多个变体，按 MAE 挑最接近者反推对方用了哪套公式）
------------------------------------------------------------------------
MELD        : raw(不截断) / floor1(值<1 记 1) / floor1+cap(Cr 封顶 4, 总分 6–40)
MELD-Na     : MELD+1.32(137-Na)-0.033*MELD*(137-Na)，Na 截 125–137，总分 6–40
MELD 3.0    : 1.33*女 + 4.56*ln(TB) + 0.82*(137-Na) - 0.24*(137-Na)*ln(TB)
              + 9.53*ln(INR) + 11.14*ln(Cr) + 1.40（TB/INR/Cr 截 1，Cr 封顶 4）
CLIF-C OF   : 六器官各 1–3 分
CLIF-C AD   : 10*(0.03*age + 0.66*ln(Cr) + 1.71*ln(INR) + 0.88*ln(WBC) - 0.05*Na + 8)
CLIF-C ACLF : 10*(0.33*OF + 0.04*age + 0.63*ln(WBC) - 2)
COSSH-ACLF II: 1.649*ln(INR) + 0.457*HE + 0.425*ln(NEU) + 0.396*ln(TB)
              + 0.576*ln(BUN) + 0.033*age
CTP         : TB / ALB / INR / 腹水 / 肝性脑病 各 1–3 分（腹水仅有二分类，按 1/2 分计）

输出
----
analysis/output2/score_concordance.json      每个评分的吻合度指标
analysis/output2/score_compare_by_patient.csv  逐例对照（供人工抽查）
analysis/output2/score_discrepancies_top.csv   差异最大的 40 例
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROV = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\catch-life队列蛋白质组患者1390人-年龄-疾病评分.xlsx")
OUT = BASE / "analysis" / "output2"

TB = "TB(umol/L)"
CR = "CR(μmol/L)"
INR = "INR"
HE = "病程有脑病及最大等级(0.no1.yes)_部分核实"   # 0–4 级
MAP = "MAP"
PF = "FIO_2"
WBC = "WBC"
NEU = "neutrophil_number"
NA = "Na"
ALB = "ALB"
BUN = "BUN"
ASC = "AD_ascites(0.no1.yes)"
SEX = "Gender"          # 0/1，取值需确认（1 是否为男）
O90 = "O90_2020.10更新后"


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"-": np.nan, "—": np.nan, " ": np.nan}),
                         errors="coerce")


# ---------- CLIF-C OF 子分 ----------
def of_liver(tb_mgdl):
    return np.select([tb_mgdl < 6.0, tb_mgdl < 12.0], [1, 2], default=3)


def of_kidney(cr_mgdl):
    return np.select([cr_mgdl < 2.0, cr_mgdl < 3.5], [1, 2], default=3)


def of_brain(he):
    return np.select([he == 0, he <= 2], [1, 2], default=3)


def of_coag(inr):
    return np.select([inr < 2.0, inr < 2.5], [1, 2], default=3)


def of_circ(map_):
    return np.where(map_ >= 70, 1, 2)


def of_resp(pf):
    return np.select([pf > 400, pf > 300], [1, 2], default=3)


# ---------- CTP 子分 ----------
def ctp_bili(tb_mgdl):
    return np.select([tb_mgdl < 2.0, tb_mgdl <= 3.0], [1, 2], default=3)


def ctp_alb(alb_gL):
    alb = alb_gL / 10.0
    return np.select([alb > 3.5, alb >= 2.8], [1, 2], default=3)


def ctp_inr(inr):
    return np.select([inr < 1.7, inr <= 2.3], [1, 2], default=3)


def ctp_asc(asc):
    # 仅有 0/1：无腹水记 1，有腹水记 2（少给 1 分，偏保守）
    return np.where(asc > 0, 2, 1)


def ctp_he(he):
    return np.select([he == 0, he <= 2], [1, 2], default=3)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
    prov = pd.read_excel(PROV)
    df = pd.DataFrame({
        "id": clin["ID"].astype(str).str.strip(),
        "center": clin["Cohort"].astype(str),
        "sex_raw": num(clin[SEX]),
        "tb_umol": num(clin[TB]),
        "cr_umol": num(clin[CR]),
        "inr": num(clin[INR]),
        "he": num(clin[HE]),
        "map": num(clin[MAP]),
        "pf": num(clin[PF]),
        "wbc": num(clin[WBC]),
        "neu": num(clin[NEU]),
        "na": num(clin[NA]),
        "alb_gL": num(clin[ALB]),
        "bun": num(clin[BUN]),
        "ascites": num(clin[ASC]),
        "o90": num(clin[O90]),
    })
    prov["id"] = prov["ID"].astype(str).str.strip()
    return df, prov


def build(df: pd.DataFrame, ages: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(ages[["ID", "AGE"]].rename(columns={"ID": "id", "AGE": "age"}),
                  on="id", how="left")
    df["event90"] = df["o90"].isin([1, 2]).astype(int)
    df["tb_mgdl"] = df["tb_umol"] / 17.1
    df["cr_mgdl"] = df["cr_umol"] / 88.4

    # ---- CLIF-C OF ----
    df["OF_liver"] = of_liver(df["tb_mgdl"])
    df["OF_kidney"] = of_kidney(df["cr_mgdl"])
    df["OF_brain"] = of_brain(df["he"])
    df["OF_coag"] = of_coag(df["inr"])
    df["OF_circ"] = of_circ(df["map"])
    df["OF_resp"] = of_resp(df["pf"])
    df["CLIF_OF"] = df[["OF_liver", "OF_kidney", "OF_brain",
                        "OF_coag", "OF_circ", "OF_resp"]].sum(axis=1)

    # ---- MELD 三个变体 ----
    eps = dict(lower=1.0)
    df["MELD_raw"] = (3.78 * np.log(df["tb_mgdl"].where(df["tb_mgdl"] > 0))
                      + 11.2 * np.log(df["inr"].where(df["inr"] > 0))
                      + 9.57 * np.log(df["cr_mgdl"].where(df["cr_mgdl"] > 0)) + 6.43)
    df["MELD_floor1"] = (3.78 * np.log(df["tb_mgdl"].clip(**eps))
                         + 11.2 * np.log(df["inr"].clip(**eps))
                         + 9.57 * np.log(df["cr_mgdl"].clip(**eps)) + 6.43)
    df["MELD_cap"] = np.where(
        df["tb_mgdl"].notna() & df["inr"].notna() & df["cr_mgdl"].notna(),
        (3.78 * np.log(df["tb_mgdl"].clip(**eps))
         + 11.2 * np.log(df["inr"].clip(**eps))
         + 9.57 * np.log(df["cr_mgdl"].clip(lower=1.0, upper=4.0)) + 6.43).clip(6, 40),
        np.nan)

    # ---- MELD-Na ----
    na_c = df["na"].clip(125, 137)
    for base, tag in (("MELD_cap", "MELDNa_cap"), ("MELD_floor1", "MELDNa_floor1")):
        m = df[base].clip(6, 40)
        df[tag] = (m + 1.32 * (137 - na_c) - 0.033 * m * (137 - na_c)).clip(6, 40)

    # ---- MELD 3.0（sex 待确认：同时给两种）----
    lnb = np.log(df["tb_mgdl"].clip(**eps))
    lni = np.log(df["inr"].clip(**eps))
    lnc = np.log(df["cr_mgdl"].clip(lower=1.0, upper=4.0))
    female = (df["sex_raw"] == 0).astype(float)      # 假设 0=女，稍后与检出率对照
    core = (4.56 * lnb + 0.82 * (137 - na_c) - 0.24 * (137 - na_c) * lnb
            + 9.53 * lni + 11.14 * lnc + 1.40)
    df["MELD3"] = (core + 1.33 * female).clip(6, 40).round()

    # ---- CLIF-C AD / ACLF / COSSH-II ----
    ln_wbc = np.log(df["wbc"].where(df["wbc"] > 0))
    ln_inr = np.log(df["inr"].where(df["inr"] > 0))
    ln_cr = np.log(df["cr_mgdl"].where(df["cr_mgdl"] > 0))
    ln_neu = np.log(df["neu"].where(df["neu"] > 0))
    ln_tb = np.log(df["tb_umol"].where(df["tb_umol"] > 0))
    ln_bun = np.log(df["bun"].where(df["bun"] > 0))

    df["CLIF_AD"] = 10 * (0.03 * df["age"] + 0.66 * ln_cr + 1.71 * ln_inr
                          + 0.88 * ln_wbc - 0.05 * df["na"] + 8)
    df["CLIF_ACLF"] = 10 * (0.33 * df["CLIF_OF"] + 0.04 * df["age"]
                            + 0.63 * ln_wbc - 2)
    he_score = np.select([df["he"] == 0, df["he"] <= 2], [1, 2], default=3)
    df["COSSH_II"] = (1.649 * ln_inr + 0.457 * he_score + 0.425 * ln_neu
                      + 0.396 * ln_tb + 0.576 * ln_bun + 0.033 * df["age"])

    # ---- CTP ----
    df["CTP_liver"] = ctp_bili(df["tb_mgdl"])
    df["CTP_alb"] = ctp_alb(df["alb_gL"])
    df["CTP_inr"] = ctp_inr(df["inr"])
    df["CTP_asc"] = ctp_asc(df["ascites"])
    df["CTP_he"] = ctp_he(df["he"])
    df["CTP"] = df[["CTP_liver", "CTP_alb", "CTP_inr", "CTP_asc", "CTP_he"]].sum(axis=1)
    return df


def agreement(provided: pd.Series, calc: pd.Series) -> dict:
    m = provided.notna() & calc.notna()
    if m.sum() < 10:
        return {"n": int(m.sum()), "note": "可比样本不足"}
    p, c = provided[m].astype(float), calc[m].astype(float)
    d = p - c
    r = float(np.corrcoef(p, c)[0, 1]) if p.std() > 0 and c.std() > 0 else np.nan
    return {
        "n": int(m.sum()),
        "provided_mean": round(float(p.mean()), 2),
        "calc_mean": round(float(c.mean()), 2),
        "mae": round(float(np.abs(d).mean()), 3),
        "median_abs_diff": round(float(np.abs(d).median()), 3),
        "max_abs_diff": round(float(np.abs(d).max()), 2),
        "bias_mean": round(float(d.mean()), 3),
        "pearson_r": round(r, 4) if np.isfinite(r) else None,
        "exact_or_0.01": round(float((np.abs(d) <= 0.011).mean()) * 100, 1),
        "within_1pt": round(float((np.abs(d) <= 1.0).mean()) * 100, 1),
        "within_2pt": round(float((np.abs(d) <= 2.0).mean()) * 100, 1),
    }


def main() -> None:
    df, prov = load()
    ages = pd.read_excel(PROV)[["ID", "AGE"]]
    df = build(df, ages)
    prov2 = prov.drop(columns=["AGE"]).rename(columns={
        "MELD": "MELD_prov", "MELD-Na": "MELDNa_prov", "CLIF-c Ofs": "CLIF_OF_prov",
        "CLIF-C Ads": "CLIF_AD_prov", "CLIF-C ACLFs": "CLIF_ACLF_prov",
        "COOSH-ACLF": "COSSH_II_prov", "CTP": "CTP_prov"})
    m = df.merge(prov2, on="id", how="inner")
    print("合并例数:", len(m))

    pairs = [
        ("MELD_prov", ["MELD_raw", "MELD_floor1", "MELD_cap"]),
        ("MELDNa_prov", ["MELDNa_cap", "MELDNa_floor1"]),
        ("CLIF_OF_prov", ["CLIF_OF"]),
        ("CLIF_AD_prov", ["CLIF_AD"]),
        ("CLIF_ACLF_prov", ["CLIF_ACLF"]),
        ("COSSH_II_prov", ["COSSH_II"]),
        ("CTP_prov", ["CTP"]),
    ]

    res: dict = {"n_merged": int(len(m)), "scores": {}}
    print("\n" + "=" * 92)
    print("表内既有列 vs 原始化验重算 —— 逐例吻合度")
    print("=" * 92)
    for prov_col, cands in pairs:
        print(f"\n### {prov_col}（表内分布：min {m[prov_col].min():.2f} / "
              f"中位 {m[prov_col].median():.2f} / max {m[prov_col].max():.2f}）")
        best = None
        for c in cands:
            a = agreement(m[prov_col], m[c])
            res["scores"].setdefault(prov_col, {})[c] = a
            flag = ""
            if a.get("mae") is not None:
                if a["mae"] < 0.5:
                    flag = "  <== 几乎完全一致"
                elif a["mae"] < 1.5 and a["pearson_r"] and a["pearson_r"] > 0.97:
                    flag = "  <== 高度一致"
            print(f"  {c:16s} n={a['n']:4d}  MAE={a['mae']:7.3f}  "
                  f"±1pt={a['within_1pt']:5.1f}%  r={a['pearson_r']}{flag}")
            if a.get("mae") is not None and (best is None or a["mae"] < best[1]["mae"]):
                best = (c, a)
        if best:
            res["scores"][prov_col]["_best_match"] = {"calc": best[0], **best[1]}

    # ---- 越界审计 ----
    bounds = {"MELD_prov": (6, 40), "MELDNa_prov": (6, 40), "CLIF_OF_prov": (6, 18),
              "CLIF_AD_prov": (0, 100), "CLIF_ACLF_prov": (0, 100),
              "COSSH_II_prov": (0, 20), "CTP_prov": (5, 15)}
    print("\n" + "=" * 92)
    print("表内既有列的取值越界审计")
    print("=" * 92)
    res["bounds"] = {}
    for c, (lo, hi) in bounds.items():
        s = m[c]
        nlo = int((s < lo).sum())
        nhi = int((s > hi).sum())
        res["bounds"][c] = {"below": nlo, "above": nhi, "min": float(s.min()),
                            "max": float(s.max()), "legal_range": [lo, hi]}
        mark = "  <== 越界" if (nlo or nhi) else ""
        print(f"  {c:14s} 合法区间 [{lo}, {hi}]  实际 [{s.min():.2f}, {s.max():.2f}]  "
              f"低于下限 {nlo} 例 / 高于上限 {nhi} 例{mark}")

    # ---- 缺失审计 ----
    print("\n缺失数：", {c: int(m[c].isna().sum()) for c, _ in pairs})

    # ---- 保存 ----
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["id", "center", "age", "sex_raw", "event90", "tb_mgdl", "cr_mgdl", "inr", "he",
            "map", "pf", "wbc", "neu", "na", "alb_gL", "bun",
            "CLIF_OF", "CLIF_AD", "CLIF_ACLF", "COSSH_II",
            "MELD_raw", "MELD_floor1", "MELD_cap", "MELDNa_cap", "MELD3", "CTP",
            "MELD_prov", "MELDNa_prov", "CLIF_OF_prov", "CLIF_AD_prov",
            "CLIF_ACLF_prov", "COSSH_II_prov", "CTP_prov"]
    out = m[keep]
    out.to_csv(OUT / "score_compare_by_patient.csv", index=False, encoding="utf-8-sig")

    # 差异最大 40 例（以最佳匹配口径计）
    diffs = []
    for prov_col, info in res["scores"].items():
        best = info.get("_best_match", {}).get("calc")
        if not best:
            continue
        d = (m[prov_col] - m[best]).abs()
        tmp = pd.DataFrame({"id": m["id"], "score": prov_col, "calc_variant": best,
                            "provided": m[prov_col], "recomputed": m[best].round(3),
                            "abs_diff": d.round(3)})
        diffs.append(tmp)
    if diffs:
        pd.concat(diffs).sort_values("abs_diff", ascending=False).head(40).to_csv(
            OUT / "score_discrepancies_top.csv", index=False, encoding="utf-8-sig")

    (OUT / "score_concordance.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n输出：analysis/output2/score_concordance.json, score_compare_by_patient.csv, "
          "score_discrepancies_top.csv")


if __name__ == "__main__":
    main()
