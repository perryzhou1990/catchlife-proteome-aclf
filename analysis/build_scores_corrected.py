# -*- coding: utf-8 -*-
"""
生成修正版评分表，并量化「表内既有评分 vs 正确重算」对 90 天判别力的影响

输入
----
CATCH-LIFE临床队列数据_1390人.xlsx      原始化验
catch-life队列蛋白质组患者1390人-年龄-疾病评分.xlsx   年龄 + 待核对的既有评分

输出
----
analysis/output2/scores_corrected_1390.csv   修正版评分（可直接供 Table 2 使用）
analysis/output2/score_discrimination.json   各评分全队列 / ACLF 亚组 C-index（提供值 vs 重算值）
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROV = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\catch-life队列蛋白质组患者1390人-年龄-疾病评分.xlsx")
OUT = BASE / "analysis" / "output2"


def num(s):
    return pd.to_numeric(s.replace({"-": np.nan, "—": np.nan, " ": np.nan}), errors="coerce")


def main() -> None:
    clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
    prov = pd.read_excel(PROV)

    d = pd.DataFrame({
        "id": clin["ID"].astype(str).str.strip(),
        "center": clin["Cohort"].astype(str),
        "sex_code": num(clin["Gender"]),
        "tb_umol": num(clin["TB(umol/L)"]),
        "cr_umol": num(clin["CR(μmol/L)"]),
        "inr": num(clin["INR"]),
        "he": num(clin["病程有脑病及最大等级(0.no1.yes)_部分核实"]),
        "map_mmhg": num(clin["MAP"]),
        "pf": num(clin["FIO_2"]),
        "wbc": num(clin["WBC"]),
        "neu": num(clin["neutrophil_number"]),
        "na": num(clin["Na"]),
        "alb_gL": num(clin["ALB"]),
        "bun": num(clin["BUN"]),
        "ascites": num(clin["AD_ascites(0.no1.yes)"]),
        "o90": num(clin["O90_2020.10更新后"]),
    }).merge(prov[["ID", "AGE"]].rename(columns={"ID": "id", "AGE": "age"}), on="id", how="left")

    d["sex_male"] = (d["sex_code"] == 1).astype(int)
    d["event90"] = d["o90"].isin([1, 2]).astype(int)
    d["tb_mgdl"] = d["tb_umol"] / 17.1
    d["cr_mgdl"] = d["cr_umol"] / 88.4

    # ---------------- CLIF-C OF ----------------
    d["OF_liver"] = np.select([d.tb_mgdl < 6, d.tb_mgdl < 12], [1, 2], default=3)
    d["OF_kidney"] = np.select([d.cr_mgdl < 2, d.cr_mgdl < 3.5], [1, 2], default=3)
    d["OF_brain"] = np.select([d.he == 0, d.he <= 2], [1, 2], default=3)
    d["OF_coag"] = np.select([d.inr < 2, d.inr < 2.5], [1, 2], default=3)
    d["OF_circ"] = np.where(d.map_mmhg >= 70, 1, 2)
    d["OF_resp"] = np.select([d.pf > 400, d.pf > 300], [1, 2], default=3)
    d["CLIF_OF"] = d[["OF_liver", "OF_kidney", "OF_brain", "OF_coag", "OF_circ", "OF_resp"]].sum(axis=1)

    # ---------------- MELD 系列 ----------------
    lnb = np.log(d.tb_mgdl.clip(lower=1))
    lni = np.log(d.inr.clip(lower=1))
    lnc = np.log(d.cr_mgdl.clip(lower=1))
    d["MELD"] = (3.78 * lnb + 11.2 * lni + 9.57 * lnc + 6.43).clip(6, 40).round(1)
    # 与主流水线 reselect_panel.py 完全同口径（不封顶不取整，缺失用中位数填补）
    d["MELD_pipe"] = (3.78 * np.log(d.tb_mgdl.clip(lower=1))
                      + 11.2 * np.log(d.inr.fillna(d.inr.median()).clip(lower=1))
                      + 9.57 * np.log(d.cr_mgdl.fillna(d.cr_mgdl.median()).clip(lower=1))
                      + 6.43)

    na_c = d.na.clip(125, 137)
    d["MELD_Na"] = (d.MELD.clip(6, 40) + 1.32 * (137 - na_c)
                    - 0.033 * d.MELD.clip(6, 40) * (137 - na_c)).clip(6, 40).round(1)

    # MELD 3.0（Kim 2021, Gastroenterology 161:1887）：INR 系数 9.09，含白蛋白两项，常数 6
    # 约束：TB/INR/Cr 下限 1.0；Cr 上限 3.0 mg/dL；Na 125–137；ALB 1.5–3.5 g/dL
    lnc3 = np.log(d.cr_mgdl.clip(lower=1, upper=3))
    alb_dl = (d.alb_gL / 10).clip(1.5, 3.5)
    female = (d.sex_code == 0).astype(float)
    d["MELD_3"] = (1.33 * female + 4.56 * lnb + 0.82 * (137 - na_c)
                   - 0.24 * (137 - na_c) * lnb + 9.09 * lni + 11.14 * lnc3
                   + 1.85 * (3.5 - alb_dl) - 1.83 * (3.5 - alb_dl) * lnc3
                   + 6).clip(6, 40).round(1)

    # ---------------- CLIF-C AD / ACLF ----------------
    d["CLIF_AD"] = 10 * (0.03 * d.age + 0.66 * np.log(d.cr_mgdl.clip(lower=0.1))
                         + 1.71 * np.log(d.inr.clip(lower=0.1))
                         + 0.88 * np.log(d.wbc.clip(lower=0.1))
                         - 0.05 * d.na + 8)
    d["CLIF_ACLF"] = 10 * (0.33 * d.CLIF_OF + 0.04 * d.age
                           + 0.63 * np.log(d.wbc.clip(lower=0.1)) - 2)

    # ---------------- COSSH-ACLF II ----------------
    he_s = np.select([d.he == 0, d.he <= 2], [1, 2], default=3)
    d["COSSH_II"] = (1.649 * np.log(d.inr.clip(lower=0.1)) + 0.457 * he_s
                     + 0.425 * np.log(d.neu.clip(lower=0.1))
                     + 0.396 * np.log(d.tb_umol.clip(lower=1))
                     + 0.576 * np.log(d.bun.clip(lower=0.1)) + 0.033 * d.age)

    # ---------------- CTP ----------------
    d["CTP"] = (np.select([d.tb_mgdl < 2, d.tb_mgdl <= 3], [1, 2], default=3)
                + np.select([(d.alb_gL / 10) > 3.5, (d.alb_gL / 10) >= 2.8], [1, 2], default=3)
                + np.select([d.inr < 1.7, d.inr <= 2.3], [1, 2], default=3)
                + np.where(d.ascites > 0, 2, 1)
                + np.select([d.he == 0, d.he <= 2], [1, 2], default=3))

    # ---------------- 与提供值并列 ----------------
    pmap = {"MELD": "MELD_prov", "MELD-Na": "MELD_Na_prov", "CLIF-c Ofs": "CLIF_OF_prov",
            "CLIF-C Ads": "CLIF_AD_prov", "CLIF-C ACLFs": "CLIF_ACLF_prov",
            "COOSH-ACLF": "COSSH_II_prov", "CTP": "CTP_prov"}
    d = d.merge(prov.rename(columns={"ID": "id", **pmap}), on="id", how="left")

    # ---------------- 判别力对比 ----------------
    scores = {"MELD": "MELD", "MELD-Na": "MELD_Na", "MELD 3.0": "MELD_3",
              "CLIF-C OF": "CLIF_OF", "CLIF-C AD": "CLIF_AD",
              "CLIF-C ACLF": "CLIF_ACLF", "COSSH-ACLF II": "COSSH_II", "CTP": "CTP"}
    prov_col = {"MELD": "MELD_prov", "MELD-Na": "MELD_Na_prov", "CLIF-C OF": "CLIF_OF_prov",
                "CLIF-C AD": "CLIF_AD_prov", "CLIF-C ACLF": "CLIF_ACLF_prov",
                "COSSH-ACLF II": "COSSH_II_prov", "CTP": "CTP_prov"}

    aclf = d["COSSH_II_prov"].notna() & (d["CLIF_OF"] >= 6)  # 占位，稍后按 COSSH 定义改
    cossh_day = num(clin["COSSH_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"])
    d["cossh_day"] = cossh_day
    aclf = d["cossh_day"].isin([1, 2])

    res = {"n": int(len(d)), "events90": int(d.event90.sum()),
           "aclF_n": int(aclf.sum()), "aclF_events": int(d.loc[aclf, "event90"].sum()),
           "all": {}, "aclF": {}}
    print("=" * 84)
    print("判别力：表内既有评分 vs 正确重算（90 天死亡/移植，C-index，越高越好）")
    print("=" * 84)
    print(f"{'评分':<16}{'提供值(全队列)':>16}{'重算(全队列)':>14}{'提供值(ACLF)':>14}{'重算(ACLF)':>12}")
    for lab, col in scores.items():
        row = {}
        for tag, sub, key in (("all", d, "all"), ("aclF", d[aclf], "aclF")):
            y = sub["event90"]
            for src, c in (("prov", prov_col.get(lab)), ("calc", col)):
                if c is None or c not in sub:
                    continue
                x = sub[c]
                ok = x.notna() & y.notna()
                if ok.sum() > 20 and y[ok].sum() > 5:
                    row[f"{src}_{key}"] = round(float(
                        concordance_index(np.ones(int(ok.sum())), -x[ok], y[ok])), 3)
        res["all"][lab] = {k: v for k, v in row.items() if k.endswith("all")}
        res["aclF"][lab] = {k: v for k, v in row.items() if k.endswith("aclF")}
        f = lambda k: ("%.3f" % row[k]) if k in row else "  –  "
        print(f"{lab:<16}{f('prov_all'):>16}{f('calc_all'):>14}"
              f"{f('prov_aclF'):>14}{f('calc_aclF'):>12}")

    # ---------------- 落盘 ----------------
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["id", "center", "age", "sex_male", "cossh_day", "event90",
            "tb_umol", "cr_umol", "inr", "he", "map_mmhg", "pf", "wbc", "neu", "na", "alb_gL", "bun",
            "OF_liver", "OF_kidney", "OF_brain", "OF_coag", "OF_circ", "OF_resp",
            "CLIF_OF", "MELD", "MELD_pipe", "MELD_Na", "MELD_3", "CLIF_AD", "CLIF_ACLF", "COSSH_II", "CTP",
            "MELD_prov", "MELD_Na_prov", "CLIF_OF_prov", "CLIF_AD_prov",
            "CLIF_ACLF_prov", "COSSH_II_prov", "CTP_prov"]
    d[keep].to_csv(OUT / "scores_corrected_1390.csv", index=False, encoding="utf-8-sig")

    # 与主流水线一致的中位数核对
    res["median_compare"] = {
        lab: {"provided": round(float(d[prov_col[lab]].median()), 2),
              "recomputed": round(float(d[col].median()), 2)}
        for lab, col in scores.items() if lab in prov_col}
    (OUT / "score_discrimination.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n中位数对照（提供 vs 重算）：")
    for k, v in res["median_compare"].items():
        print(f"  {k:<16} {v['provided']:>8.2f}  ->  {v['recomputed']:>7.2f}")
    print("\n输出：analysis/output2/scores_corrected_1390.csv, score_discrimination.json")


if __name__ == "__main__":
    main()
