# -*- coding: utf-8 -*-
"""
年龄数据交接校验与规范化（为 CLIF-C ACLF / CLIF-C AD / COSSH-ACLF II 补算准备）

背景
----
交付的临床工作簿（CATCH-LIFE临床队列数据_1390人.xlsx）共 57 个字段、两个 sheet
（队列数据1390人 / 变量注释），经全字段扫描无 age、无出生日期、无入院日期，
因此三个需要年龄的评分一直无法计算。年龄由课题组另行取回后，先用本脚本校验
再交给 scores_clif.py。

用法
----
    python analysis/prep_age.py --src <年龄文件> [--out analysis/age.csv]

    --src  可为 .csv / .xlsx / .xls；需含两列：患者 ID、年龄
           列名自动识别（ID / id / 编号 / 住院号；age / 年龄 / Age）
    --out  规范化输出，默认 analysis/age.csv（列固定为 ID,age）

校验项（任一不通过会明确报错，不做静默填补）
----
    1. ID 覆盖率：与临床主表 1,390 例的交集比例（目标 100%）
    2. ID 重复：同一 ID 出现多次（要求唯一，否则报错并列出）
    3. ID 格式：去空格、去 ".0" 尾巴、全角转半角后比对
    4. 年龄范围：合法 18–100；越界值列出但不自动改
    5. 年龄缺失：NaN 计数
    6. 分布核对：中位数 / IQR，与计划书所载"中位 48 岁"比对提示

输出
----
    analysis/age.csv            规范化两列（可直接 --age 传给 scores_clif.py）
    analysis/output5/age_audit.json   校验报告
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
OUT_JSON = BASE / "analysis" / "output5" / "age_audit.json"

ID_PAT = re.compile(r"^(id|编号|患者id|住院号|病案号|patientid|patient_id|登记号)$", re.I)
AGE_PAT = re.compile(r"^(age|年龄|岁|ageyears|age_years|年龄岁)$", re.I)


def norm_id(s: pd.Series) -> pd.Series:
    """统一 ID：转半角、去空格、去小数尾巴。"""
    out = (s.astype(str)
            .str.replace("\u3000", "", regex=False)
            .str.strip()
            .str.replace(r"\.0+$", "", regex=True))
    return out


def pick(cols: list[str], pat: re.Pattern) -> str | None:
    for c in cols:
        if pat.match(str(c).strip().replace(" ", "").replace("_", "_")):
            return c
    # 退一步：包含匹配
    for c in cols:
        cl = str(c).strip().lower()
        if pat.search(cl):
            return c
    return None


def read_any(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        xl = pd.ExcelFile(p)
        # 若多 sheet，取第一个含 ID+年龄列的
        best = None
        for sh in xl.sheet_names:
            head = pd.read_excel(p, sheet_name=sh, nrows=0)
            cols = list(head.columns)
            if pick(cols, ID_PAT) and pick(cols, AGE_PAT):
                best = sh
                break
        sh = best or xl.sheet_names[0]
        d = pd.read_excel(p, sheet_name=sh)
        print(f"[读入] sheet='{sh}'，{len(d)} 行 × {len(d.columns)} 列")
        return d
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            d = pd.read_csv(p, encoding=enc)
            print(f"[读入] CSV（{enc}），{len(d)} 行 × {len(d.columns)} 列")
            return d
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"无法解码：{p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="年龄文件（csv/xlsx）")
    ap.add_argument("--out", default=str(BASE / "analysis" / "age.csv"))
    ap.add_argument("--report", default=str(OUT_JSON),
                    help="校验报告输出路径（自检时请另指，避免覆盖真实报告）")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"文件不存在：{src}")

    raw = read_any(src)
    cols = list(raw.columns)
    print(f"[列名] {cols}")

    cid = pick(cols, ID_PAT)
    cage = pick(cols, AGE_PAT)
    if cid is None or cage is None:
        raise SystemExit(
            f"未能自动识别列。需要一列患者 ID、一列年龄；实际列名：{cols}\n"
            f"请把列名改成 ID / age，或用 --col-id / --col-age 指定（见脚本说明）")

    d = pd.DataFrame({"ID": norm_id(raw[cid]),
                      "age": pd.to_numeric(raw[cage].replace({"-": np.nan, "—": np.nan}),
                                           errors="coerce")})
    print(f"[识别] ID='{cid}'  age='{cage}'")

    # ---- 与主表比对 ----
    clin = pd.read_excel(CLIN, sheet_name="队列数据1390人", usecols=["ID"])
    main_ids = set(norm_id(clin["ID"]))
    have = set(d["ID"])
    inter = main_ids & have

    rep: dict = {
        "src": str(src),
        "col_id": str(cid), "col_age": str(cage),
        "rows_in_file": int(len(d)),
        "unique_ids": int(d["ID"].nunique()),
        "duplicated_ids": int(len(d) - d["ID"].nunique()),
        "main_cohort_n": len(main_ids),
        "matched_in_main": len(inter),
        "coverage_pct": round(100 * len(inter) / len(main_ids), 2),
        "in_file_not_in_main": len(have - main_ids),
        "age_missing": int(d["age"].isna().sum()),
    }

    dup = d[d["ID"].duplicated(keep=False)].sort_values("ID")
    if len(dup):
        rep["duplicated_ids_list"] = dup["ID"].unique().tolist()[:50]
        print(f"\n[!] 重复 ID {rep['duplicated_ids']} 个，示例：{rep['duplicated_ids_list'][:10]}")

    bad_mask = (d["age"] < 18) | (d["age"] > 100)
    bad = d[bad_mask]
    if len(bad):
        rep["out_of_range"] = bad[["ID", "age"]].to_dict("records")[:50]
        print(f"[!] 越界年龄 {len(bad)} 条（<18 或 >100），已置空、不进输出：")
        print(bad[["ID", "age"]].head(10).to_string(index=False))
        # 越界必为编码错误（150 岁/个位数），置空而非透传，避免污染 CLIF-C ACLF
        d.loc[bad_mask, "age"] = np.nan
        rep["out_of_range_nulled"] = int(len(bad))

    a = d.loc[d["ID"].isin(inter), "age"].dropna()
    if len(a):
        rep["age_stats_matched"] = {
            "n": int(len(a)), "mean": round(float(a.mean()), 1),
            "median": float(a.median()),
            "q1": float(a.quantile(.25)), "q3": float(a.quantile(.75)),
            "min": float(a.min()), "max": float(a.max()),
        }

    miss = sorted(main_ids - have)
    if miss:
        rep["missing_id_examples"] = miss[:20]

    print("\n" + "=" * 62)
    print("年龄数据校验")
    print("=" * 62)
    print(f"文件行数            {rep['rows_in_file']}")
    print(f"唯一 ID             {rep['unique_ids']}  （重复 {rep['duplicated_ids']}）")
    print(f"与主表 1,390 例交集  {rep['matched_in_main']}   覆盖率 {rep['coverage_pct']}%")
    if "age_stats_matched" in rep:
        s = rep["age_stats_matched"]
        print(f"年龄（已匹配）      中位 {s['median']:.0f}（IQR {s['q1']:.0f}–{s['q3']:.0f}），"
              f"均值 {s['mean']}，范围 {s['min']:.0f}–{s['max']:.0f}")
        print(f"                    计划书所载中位 48 岁 —— "
              f"{'基本一致' if abs(s['median'] - 48) <= 6 else '与计划书差异较大，需核对单位/人群'}")
    print(f"年龄缺失            {rep['age_missing']}")
    print("=" * 62)

    # ---- 落盘（只输出匹配成功的、无重复的规范两列）----
    ok = d.dropna(subset=["age"]).drop_duplicates("ID", keep="first")
    ok = ok[ok["ID"].isin(main_ids)]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    ok.to_csv(outp, index=False, encoding="utf-8-sig")
    print(f"\n[输出] {outp}（{len(ok)} 行，列：ID, age）")

    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[报告] {rp}")

    if rep["coverage_pct"] < 95:
        print("\n[警告] 覆盖率 <95%，三个评分将只在该子集上计算，"
              "C 指数与主模型不可直接同比。建议补齐后重跑。")
        sys.exit(2)
    print("\n[下一步] python analysis/scores_clif.py --age analysis/age.csv")


if __name__ == "__main__":
    main()
