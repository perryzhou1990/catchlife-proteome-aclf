# -*- coding: utf-8 -*-
"""Stage runner for reselect_panel under the 8dev split.

Usage:  SPLIT=8dev python run_8dev_stage.py primary
        SPLIT=8dev python run_8dev_stage.py variants
Keeps the heavy variants separate so the primary panel is known first.
"""
import json
import os
import sys

import reselect_panel as R

stage = sys.argv[1] if len(sys.argv) > 1 else "primary"
print("SPLIT=%s dev=%s val=%s" % (R.SPLIT, R.DEV_CENTRES, R.VAL_CENTRES))

res = {}
if stage == "primary":
    r = R.run("primary883", R.Z.columns.tolist(), "(all 883 proteins, no filter)")
    if r:
        res["primary883"] = r
    (R.OUT / "stage_primary.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
elif stage == "variants":
    det_cols = [c for c in R.Z.columns if R.det_rate[c] >= R.DET_CUT]
    r = R.run("filter665", det_cols, "(proteins measurable in >=80%)")
    if r:
        res["filter665"] = r
    OLD = ["TNF.RI", "Renin", "DR6", "SREC.II"]
    if all(p in R.Z.columns for p in OLD):
        met_old, coeff_old, summ_old = R.evaluate(OLD, "oldpanel")
        res["oldpanel"] = dict(tag="oldpanel", panel=OLD, summary=summ_old)
    (R.OUT / "stage_variants.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print("DONE", stage)
