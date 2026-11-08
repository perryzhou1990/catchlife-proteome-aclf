# -*- coding: utf-8 -*-
"""
CATCH-LIFE 评分引擎（严格按原始文献公式实现）

设计原则
--------
1. 每个评分只实现一次，公式、单位换算、取值约束全部写在函数 docstring 里，
   并注明原始文献。不接受任何"别人算好的"结果作为输入。
2. 所有约束（下限、上限、截断）都按原文执行，不用经验值替代。
3. 内置自检 self_test()：用文献中给出的算例反算，算不出来就报错，不静默通过。
4. 缺失值不填补：任一必需变量缺失 -> 该评分记 NaN，由上层决定完整病例集。

数据单位（已逐一核对）
----------------------
TB      μmol/L  (TB(mg/dL) 列与 μmol/17.1 一致率 100%)
CR      μmol/L  (与 CR(mg/dL) 列一致率 98.4%)
BUN     mmol/L  (中位 4.4，符合 mmol/L 量级)
Na      mmol/L
ALB     g/L
WBC     10^9/L
NEUT    10^9/L
INR     无量纲
P/F     mmHg（FIO_2 列存的是 PaO2/FiO2 比值，中位 476.19 = 100/0.21）

参考文献
--------
MELD          Kamath PS, et al. Hepatology 2001;33:464-70（系数 3.78/11.2/9.57/6.43）
MELD-Na       Kim WR, et al. N Engl J Med 2008;359:1018-26；OPTN 实施式（137/1.32/0.033）
MELD 3.0      Kim WR, et al. Gastroenterology 2021;161:1887-98
CLIF-C OF     Jalan R, et al. J Hepatol 2014;61:1038-47
CLIF-C ACLF   Jalan R, et al. J Hepatol 2014;61:1038-47
CLIF-C AD     Jalan R, et al. J Hepatol 2015;62:831-40
COSSH-ACLF II Li J, et al. J Hepatol 2021;75:1104-15
CTP           Pugh RN, et al. Br J Surg 1973;60:646-9
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
AGE = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\catch-life队列蛋白组患者1390人-年龄-疾病评分.xlsx")
AGE_ALT = BASE / "analysis" / "age.csv"
OUT = BASE / "analysis" / "output2"

MGDL_TB = 17.1     # 1 mg/dL bilirubin = 17.1 μmol/L
MGDL_CR = 88.4     # 1 mg/dL creatinine = 88.4 μmol/L


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.strip().replace({"-": np.nan, "—": np.nan, "": np.nan, " ": np.nan, "nan": np.nan}),
        errors="coerce",
    )


# --------------------------------------------------------------------------- #
# 数据装载
# --------------------------------------------------------------------------- #
def load_cohort() -> pd.DataFrame:
    c = pd.read_excel(CLIN, sheet_name="队列数据1390人")
    c.columns = [str(x).strip() for x in c.columns]

    d = pd.DataFrame({
        "id": c["ID"].astype(str).str.strip(),
        "center": c["ID"].astype(str).str.split("-").str[0],
        "sex_code": _num(c["Gender"]),
        "tb_umol": _num(c["TB(umol/L)"]),
        "cr_umol": _num(c["CR(μmol/L)"]),
        "inr": _num(c["INR"]),
        "he": _num(c["病程有脑病及最大等级(0.no1.yes)_部分核实"]),
        "map_mmhg": _num(c["MAP"]),
        "pf": _num(c["FIO_2"]),
        "wbc": _num(c["WBC"]),
        "neut": _num(c["neutrophil_number"]),
        "na": _num(c["Na"]),
        "alb_gL": _num(c["ALB"]),
        "bun": _num(c["BUN"]),
        "plt": _num(c["PLT"]),
        "ascites": _num(c["AD_ascites(0.no1.yes)"]),
        "hbv": _num(c["HBV"]),
        "cirrhosis": _num(c["cirrhosis"]),
        "o90": _num(c["O90_2020.10更新后"]),
        "os1y": _num(c["OS_1y_更新"]),
        "cossh_day": _num(c["COSSH_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"]),
        "easl_day": _num(c["EASL_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"]),
    })

    # 年龄：优先读原始评分表，失败则回退到已导出的 age.csv
    age = None
    try:
        p = pd.read_excel(AGE)
        age = p.rename(columns={"ID": "id", "AGE": "age"})[["id", "age"]].copy()
        age["id"] = age["id"].astype(str).str.strip()
    except Exception:
        if AGE_ALT.exists():
            age = pd.read_csv(AGE_ALT).rename(columns={"ID": "id"})
            age["id"] = age["id"].astype(str).str.strip()
    if age is None:
        raise RuntimeError("年龄表读取失败")
    d = d.merge(age, on="id", how="left")

    # 单位换算
    d["tb"] = d["tb_umol"] / MGDL_TB           # mg/dL
    d["cr"] = d["cr_umol"] / MGDL_CR           # mg/dL
    d["alb"] = d["alb_gL"] / 10.0              # g/dL
    d["he_grade"] = d["he"].clip(0, 4)         # 0-4
    d["sex_female"] = (d["sex_code"] == 0).astype(float)
    d.loc[d["sex_code"].isna() | ~d["sex_code"].isin([0, 1]), "sex_female"] = np.nan
    # 唯一一条 out-of-range 性别码（=2）按众数（男）处理，与既有模型一致
    d["sex_female"] = d["sex_female"].fillna(0.0)

    # 结局：90 天死亡或移植（O90: 0 存活, 1 死亡, 2 移植）
    d["event90"] = d["o90"].isin([1, 2]).astype(int)
    d["time90"] = np.minimum(d["os1y"].fillna(90), 90)
    d["aclf_any"] = d["cossh_day"].isin([1, 2])
    d["aclf_adm"] = d["cossh_day"].eq(1)
    d["hbv_aclf"] = d["aclf_any"] & d["hbv"].eq(1)
    return d


# --------------------------------------------------------------------------- #
# 评分实现
# --------------------------------------------------------------------------- #
def meld(tb: pd.Series, inr: pd.Series, cr: pd.Series, cap: bool = True) -> pd.Series:
    """MELD (Kamath 2001; OPTN).

    MELD = 3.78*ln(TB mg/dL) + 11.2*ln(INR) + 9.57*ln(Cr mg/dL) + 6.43
    约束：TB/INR/Cr <1.0 一律记 1.0；Cr 上限 4.0 mg/dL；总分截断 6-40。
    """
    T = tb.clip(lower=1.0)
    I = inr.clip(lower=1.0)
    C = cr.clip(lower=1.0, upper=4.0)
    s = 3.78 * np.log(T) + 11.2 * np.log(I) + 9.57 * np.log(C) + 6.43
    return s.clip(6, 40) if cap else s


def meld_na(m: pd.Series, na: pd.Series) -> pd.Series:
    """MELD-Na (Kim 2008 NEJM; OPTN 实施式).

    MELD-Na = MELD + 1.32*(137 - Na) - 0.033*MELD*(137 - Na)
    约束：Na 截断 125-137 mmol/L；总分截断 6-40。
    注：另有 MELD-1.59*(135-Na) 等变体，非 OPTN 实施式，本稿不用。
    """
    N = na.clip(125, 137)
    return (m + 1.32 * (137 - N) - 0.033 * m * (137 - N)).clip(6, 40)


def meld3(tb, inr, cr, na, alb, female: pd.Series) -> pd.Series:
    """MELD 3.0 (Kim 2021 Gastroenterology 161:1887).

    MELD 3.0 = 1.33*(女性) + 4.56*ln(TB) + 0.82*(137-Na)
               - 0.24*(137-Na)*ln(TB) + 9.09*ln(INR) + 11.14*ln(Cr)
               + 1.85*(3.5-ALB) - 1.83*(3.5-ALB)*ln(Cr) + 6
    约束：TB/INR/Cr <1 记 1；Cr 上限 3.0；Na 125-137；ALB 1.5-3.5 g/dL；总分 6-40。
    """
    T = tb.clip(lower=1.0)
    I = inr.clip(lower=1.0)
    C = cr.clip(lower=1.0, upper=3.0)
    N = na.clip(125, 137)
    A = alb.clip(1.5, 3.5)
    s = (1.33 * female + 4.56 * np.log(T) + 0.82 * (137 - N)
         - 0.24 * (137 - N) * np.log(T) + 9.09 * np.log(I) + 11.14 * np.log(C)
         + 1.85 * (3.5 - A) - 1.83 * (3.5 - A) * np.log(C) + 6)
    return s.clip(6, 40)


def clif_of(d: pd.DataFrame) -> pd.DataFrame:
    """CLIF-C OF（Jalan 2014），6 器官各 1-3 分，总分 6-18。

    肝     TB <6 / 6-<12 / >=12 mg/dL            1/2/3
    肾     Cr <2 / 2-<3.5 / >=3.5 mg/dL           1/2/3（本队列无 RRT 标记）
    脑     HE 0 / 1-2 / 3-4                        1/2/3
    凝血   INR <2 / 2-<2.5 / >=2.5                1/2/3
    循环   MAP >=70 / <70 / 升压药                1/2/3（本队列无升压药标记，最高 2）
    呼吸   P/F >400 / 300-400 / <=300 / 机械通气  1/2/3（本队列无机械通气标记）
    """
    out = pd.DataFrame(index=d.index)
    out["OF_liver"] = np.select([d.tb < 6, d.tb < 12], [1, 2], default=3)
    out["OF_kidney"] = np.select([d.cr < 2, d.cr < 3.5], [1, 2], default=3)
    out["OF_brain"] = np.select([d.he_grade == 0, d.he_grade <= 2], [1, 2], default=3)
    out["OF_coag"] = np.select([d.inr < 2, d.inr < 2.5], [1, 2], default=3)
    out["OF_circ"] = np.select([d.map_mmhg >= 70, d.map_mmhg < 70], [1, 2], default=np.nan)
    out["OF_resp"] = np.select([d.pf > 400, d.pf > 300], [1, 2], default=3)
    miss = d[["tb", "cr", "he_grade", "inr", "map_mmhg", "pf"]].isna()
    out[miss.any(axis=1)] = np.nan
    out["CLIF_OF"] = out[["OF_liver", "OF_kidney", "OF_brain",
                          "OF_coag", "OF_circ", "OF_resp"]].sum(axis=1, min_count=6)
    return out


def clif_c_aclf(of: pd.Series, age: pd.Series, wbc: pd.Series) -> pd.Series:
    """CLIF-C ACLF (Jalan 2014)：10*(0.33*CLIF-OF + 0.04*age + 0.63*ln(WBC) - 2)，0-100。"""
    return (10 * (0.33 * of + 0.04 * age + 0.63 * np.log(wbc.clip(lower=0.01)) - 2)).clip(0, 100)


def clif_c_ad(age, cr, inr, wbc, na) -> pd.Series:
    """CLIF-C AD (Jalan 2015)：
    10*(0.03*age + 0.66*ln(Cr mg/dL) + 1.71*ln(INR) + 0.88*ln(WBC) - 0.05*Na + 8)
    原文将总分截断至 0-100（极端值超出范围者按 1% 概率误差可接受）。"""
    s = 10 * (0.03 * age + 0.66 * np.log(cr.clip(lower=0.01))
              + 1.71 * np.log(inr.clip(lower=0.01))
              + 0.88 * np.log(wbc.clip(lower=0.01)) - 0.05 * na + 8)
    return s.clip(0, 100)


def cossh_aclf2(inr, he, neut, tb_umol, bun, age) -> pd.Series:
    """COSSH-ACLF II (Li 2021 J Hepatol 75:1104)：
    1.649*ln(INR) + 0.457*HE 评分(0级1分, I-II级2分, III-IV级3分)
    + 0.425*ln(NEUT 10^9/L) + 0.396*ln(TB μmol/L) + 0.576*ln(尿素 mmol/L) + 0.033*age
    仅适用于 HBV-ACLF；原文 28 天死亡判别界值 7.40。"""
    hes = np.select([he == 0, he <= 2], [1.0, 2.0], default=3.0)
    return (1.649 * np.log(inr.clip(lower=0.01)) + 0.457 * hes
            + 0.425 * np.log(neut.clip(lower=0.01))
            + 0.396 * np.log(tb_umol.clip(lower=0.01))
            + 0.576 * np.log(bun.clip(lower=0.01)) + 0.033 * age)


def ctp(tb, alb, inr, ascites, he) -> pd.Series:
    """Child-Turcotte-Pugh (Pugh 1973)，5 项各 1-3 分，5-15。

    胆红素 <2 / 2-3 / >3 mg/dL；白蛋白 >3.5 / 2.8-3.5 / <2.8 g/dL；
    INR <1.7 / 1.7-2.3 / >2.3；腹水 无/轻/中重度；HE 无/1-2/3-4。
    本队列腹水仅有 0/1 二分，故腹水分项最高 2 分，CTP 结构性偏低，仅作参考。
    """
    b = np.select([tb < 2, tb <= 3], [1, 2], default=3)
    a = np.select([alb > 3.5, alb >= 2.8], [1, 2], default=3)
    i = np.select([inr < 1.7, inr <= 2.3], [1, 2], default=3)
    s = np.select([ascites == 0, ascites >= 1], [1, 2], default=np.nan)
    e = np.select([he == 0, he <= 2], [1, 2], default=3)
    return pd.Series(b + a + i + s + e, index=tb.index)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def self_test() -> dict:
    """用文献给出的算例反算，任何一项不符即报错。"""
    rep = {}

    # 1) MELD 手算：TB 2.0, INR 1.5, Cr 1.0
    s = pd.Series([2.0]); i = pd.Series([1.5]); c = pd.Series([1.0])
    expect = 3.78 * np.log(2.0) + 11.2 * np.log(1.5) + 9.57 * np.log(1.0) + 6.43
    got = float(meld(s, i, c, cap=False).iloc[0])
    assert abs(got - expect) < 1e-9, (got, expect)
    rep["MELD_worked_example"] = round(got, 3)       # 13.59

    # 2) MELD 约束：三项均 <1 -> 全部记 1 -> 恰为常数 6.43（高于下限 6，故截断后仍为 6.43）
    v = float(meld(pd.Series([0.5]), pd.Series([0.5]), pd.Series([0.5]), cap=False).iloc[0])
    assert abs(v - 6.43) < 1e-9, v
    assert abs(float(meld(pd.Series([0.5]), pd.Series([0.5]), pd.Series([0.5])).iloc[0]) - 6.43) < 1e-9
    # Cr 上限 4.0：Cr 6 与 Cr 4 同分
    assert abs(float(meld(pd.Series([2.0]), pd.Series([1.5]), pd.Series([6.0])).iloc[0])
               - float(meld(pd.Series([2.0]), pd.Series([1.5]), pd.Series([4.0])).iloc[0])) < 1e-9
    rep["MELD_constraints"] = "floor 1.0 / Cr cap 4.0 / total 6-40 all enforced"

    # 3) MELD 3.0 文献算例（Kim 2021 例）：女性, TB 4 mg/dL, INR 1.2, Cr 1.0,
    #    ALB 1.5 g/dL, Na 135 -> MELD 3.0 = 20；同例 MELD-Na = 15
    m3 = float(meld3(pd.Series([4.0]), pd.Series([1.2]), pd.Series([1.0]),
                     pd.Series([135.0]), pd.Series([1.5]), pd.Series([1.0])).iloc[0])
    assert round(m3) == 20, round(m3)
    m = float(meld(pd.Series([4.0]), pd.Series([1.2]), pd.Series([1.0])).iloc[0])
    mn = float(meld_na(pd.Series([m]), pd.Series([135.0])).iloc[0])
    assert round(mn) == 15, round(mn)
    rep["MELD3_published_example"] = {"MELD3": round(m3, 2), "MELD": round(m, 2), "MELD_Na": round(mn, 2)}

    # 4) Na=137 时 MELD-Na 应等于 MELD
    assert abs(float(meld_na(pd.Series([m]), pd.Series([137.0])).iloc[0]) - m) < 1e-9
    rep["MELD_Na_reduces_to_MELD_at_Na137"] = True

    # 5) CLIF-C AD 截断：极端值不超过 100、不低于 0
    lo = float(clif_c_ad(pd.Series([18.0]), pd.Series([0.3]), pd.Series([0.8]),
                         pd.Series([1.0]), pd.Series([160.0])).iloc[0])
    hi = float(clif_c_ad(pd.Series([85.0]), pd.Series([9.0]), pd.Series([8.0]),
                         pd.Series([40.0]), pd.Series([105.0])).iloc[0])
    assert lo >= 0 and hi <= 100, (lo, hi)
    rep["CLIF_AD_trim"] = [round(lo, 2), round(hi, 2)]

    # 6) CLIF-C ACLF：OF=6, age=50, WBC=5 -> 10*(0.33*6+2+0.63*ln5-2)
    v = float(clif_c_aclf(pd.Series([6.0]), pd.Series([50.0]), pd.Series([5.0])).iloc[0])
    assert abs(v - 10 * (0.33 * 6 + 0.04 * 50 + 0.63 * np.log(5) - 2)) < 1e-9
    rep["CLIF_ACLF_worked_example"] = round(v, 2)

    # 7) COSSH-ACLF II 方向：INR 或 TB 升高，得分必须升高
    a = float(cossh_aclf2(pd.Series([1.5]), pd.Series([0]), pd.Series([3.0]),
                          pd.Series([200.0]), pd.Series([5.0]), pd.Series([50.0])).iloc[0])
    b = float(cossh_aclf2(pd.Series([2.5]), pd.Series([0]), pd.Series([3.0]),
                          pd.Series([200.0]), pd.Series([5.0]), pd.Series([50.0])).iloc[0])
    assert b > a, (a, b)
    rep["COSSH_monotone_in_INR"] = [round(a, 3), round(b, 3)]

    # 8) CTP 范围
    assert 5 <= float(ctp(pd.Series([1.0]), pd.Series([4.0]), pd.Series([1.0]),
                          pd.Series([0.0]), pd.Series([0.0])).iloc[0]) == 5
    rep["CTP_min_5"] = True
    return rep


# --------------------------------------------------------------------------- #
def main() -> None:
    tests = self_test()
    print("=" * 78)
    print("自检（与原始文献算例比对）")
    print("=" * 78)
    for k, v in tests.items():
        print(f"  {k:<34} {v}")

    d = load_cohort()
    of = clif_of(d)
    d = pd.concat([d, of], axis=1)

    d["MELD"] = meld(d.tb, d.inr, d.cr)
    d["MELD_Na"] = meld_na(d.MELD, d.na)
    d["MELD3"] = meld3(d.tb, d.inr, d.cr, d.na, d.alb, d.sex_female)
    d["CLIF_ACLF"] = clif_c_aclf(d.CLIF_OF, d.age, d.wbc)
    d["CLIF_AD"] = clif_c_ad(d.age, d.cr, d.inr, d.wbc, d.na)
    d["COSSH_II"] = cossh_aclf2(d.inr, d.he_grade, d.neut, d.tb_umol, d.bun, d.age)
    d["CTP"] = ctp(d.tb, d.alb, d.inr, d.ascites, d.he_grade)

    print("\n" + "=" * 78)
    print("各评分分布（全队列，未做任何填补）")
    print("=" * 78)
    print(f"{'评分':<12}{'n':>6}{'缺失':>6}{'中位':>9}{'IQR':>18}{'范围':>22}")
    for lab in ["MELD", "MELD_Na", "MELD3", "CLIF_OF", "CLIF_AD", "CLIF_ACLF", "COSSH_II", "CTP"]:
        s = d[lab]
        print(f"{lab:<12}{s.notna().sum():>6}{s.isna().sum():>6}{s.median():>9.2f}"
              f"{'%.1f–%.1f' % (s.quantile(.25), s.quantile(.75)):>18}"
              f"{'%.1f–%.1f' % (s.min(), s.max()):>22}")

    # 与既有流水线 MELD 的一致性（流水线用中位数填补 INR/Cr）
    pipe = (3.78 * np.log(d.tb.clip(lower=1))
            + 11.2 * np.log(d.inr.fillna(d.inr.median()).clip(lower=1))
            + 9.57 * np.log(d.cr.fillna(d.cr.median()).clip(lower=1)) + 6.43).clip(6, 40)
    both = d.MELD.notna()
    print(f"\n严格 MELD 与流水线 MELD（中位填补版）: n={both.sum()} "
          f"r={np.corrcoef(d.MELD[both], pipe[both])[0,1]:.6f} "
          f"最大绝对差={np.abs(d.MELD[both] - pipe[both]).max():.4f} "
          f"完全一致(<0.01)={np.mean(np.abs(d.MELD[both]-pipe[both])<0.01)*100:.1f}%")

    # 分层完整性
    print("\n亚组（COSSH 定义）:")
    for lab, m in [("non-ACLF", ~d.aclf_any), ("ACLF on admission", d.aclf_adm),
                   ("ACLF developed", d.cossh_day.eq(2)), ("HBV-ACLF", d.hbv_aclf)]:
        sub = d[m]
        print(f"  {lab:<20} n={len(sub):5d}  90d 事件={int(sub.event90.sum()):4d} "
              f"({sub.event90.mean()*100:5.1f}%)")

    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["id", "center", "age", "sex_female", "event90", "time90", "cossh_day", "aclf_any",
            "aclf_adm", "hbv", "hbv_aclf", "cirrhosis",
            "tb", "tb_umol", "cr", "cr_umol", "inr", "he_grade", "map_mmhg", "pf", "wbc", "neut",
            "na", "alb", "bun", "plt", "ascites",
            "OF_liver", "OF_kidney", "OF_brain", "OF_coag", "OF_circ", "OF_resp",
            "CLIF_OF", "MELD", "MELD_Na", "MELD3", "CLIF_AD", "CLIF_ACLF", "COSSH_II", "CTP"]
    d[keep].to_csv(OUT / "scores_strict_1390.csv", index=False, encoding="utf-8-sig")
    (OUT / "score_engine_selftest.json").write_text(
        json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: {OUT/'scores_strict_1390.csv'}  ({len(d)} 行 × {len(keep)} 列)")


if __name__ == "__main__":
    main()
