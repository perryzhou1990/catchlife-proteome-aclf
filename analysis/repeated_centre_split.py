# -*- coding: utf-8 -*-
"""Repeated centre-level split validation (zero-cost replacement for an
external cohort).

The manuscript currently reports ONE 5/8 centre split.  A reviewer can always
ask "would the gain survive a different allocation of centres?".  This script
answers that by repeating the split 200 times: centres are randomly assigned
to derivation / validation, the locked 5-protein panel is refitted in the
derivation part, and discrimination is measured in the held-out centres.
No new assay is required -- it uses exactly the data behind Fig 4.

Reported per split: validation AUC90 and C-index for MELD and MELD+5P, the
paired differences, and how often the difference is positive.  Selection is
NOT re-run inside each split (that would need 200 x stability selection);
the panel identity is therefore held fixed, which is the honest question:
"given this panel, is the gain split-invariant?"

Output
    analysis/output2/repeated_centre_split.csv        one row per split
    analysis/output2/repeated_centre_split_summary.json
    figures/figS1/panels/data/figS1_panelG.png        distribution plot
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "figures" / "fig4" / "code" / "fig4_make.py"
SPLIT = "# ---------------- figure ----------------"
OUT = BASE / "analysis" / "output2"

text = SRC.read_text(encoding="utf-8")
ns = {"__name__": "__panel_data__", "__file__": str(SRC)}
exec(compile(text[: text.index(SPLIT)], str(SRC), "exec"), ns)

fit, base, P5 = ns["fit"], ns["base"], ns["P5"]
centres = sorted(base["center"].unique())
print("centres:", len(centres), centres)

RNG = np.random.default_rng(20261106)
BLUE, GREY, AMBER = "#2E5E8C", "#777777", "#B07A2A"


def evaluate(dev, val):
    out = {}
    for name, cols in (("MELD", ["meld"]), ("MELD+5P", ["meld"] + P5)):
        m = fit(dev, cols)
        lp = np.log(m.predict_partial_hazard(val[cols]).to_numpy())
        out[name] = dict(
            auc=roc_auc_score(val.event, lp),
            c=concordance_index(val.time, -lp, val.event))
    return out


rows = []
for b in range(200):
    while True:
        dev_c = list(RNG.choice(centres, 5, replace=False))
        dm = base["center"].isin(dev_c).to_numpy()
        dev, val = base[dm], base[~dm]
        if dev.event.sum() >= 60 and val.event.sum() >= 15:
            break
    r = evaluate(dev.reset_index(drop=True), val.reset_index(drop=True))
    rows.append(dict(split=b + 1, dev_centres="|".join(dev_c),
                     n_dev=len(dev), ev_dev=int(dev.event.sum()),
                     n_val=len(val), ev_val=int(val.event.sum()),
                     auc_meld=r["MELD"]["auc"], auc_panel=r["MELD+5P"]["auc"],
                     d_auc=r["MELD+5P"]["auc"] - r["MELD"]["auc"],
                     c_meld=r["MELD"]["c"], c_panel=r["MELD+5P"]["c"],
                     d_c=r["MELD+5P"]["c"] - r["MELD"]["c"]))
    if (b + 1) % 25 == 0:
        print("  %d/200 done" % (b + 1))

df = pd.DataFrame(rows)
df.to_csv(OUT / "repeated_centre_split.csv", index=False, encoding="utf-8-sig")

q = lambda s, p: float(np.percentile(s, p))
s = dict(
    n_splits=len(df),
    d_auc_median=q(df.d_auc, 50), d_auc_lo=q(df.d_auc, 2.5), d_auc_hi=q(df.d_auc, 97.5),
    d_auc_frac_positive=float((df.d_auc > 0).mean()),
    d_c_median=q(df.d_c, 50), d_c_lo=q(df.d_c, 2.5), d_c_hi=q(df.d_c, 97.5),
    d_c_frac_positive=float((df.d_c > 0).mean()),
    auc_meld_median=q(df.auc_meld, 50), auc_meld_range=[q(df.auc_meld, 2.5), q(df.auc_meld, 97.5)],
    auc_panel_median=q(df.auc_panel, 50), auc_panel_range=[q(df.auc_panel, 2.5), q(df.auc_panel, 97.5)],
    ev_val_median=q(df.ev_val, 50))
(OUT / "repeated_centre_split_summary.json").write_text(
    json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")

print("\ndAUC  median %+.4f  (2.5-97.5%%: %+.4f to %+.4f)  positive in %.1f%% of splits"
      % (s["d_auc_median"], s["d_auc_lo"], s["d_auc_hi"], 100 * s["d_auc_frac_positive"]))
print("dC    median %+.4f  (%.4f to %+.4f)  positive in %.1f%%"
      % (s["d_c_median"], s["d_c_lo"], s["d_c_hi"], 100 * s["d_c_frac_positive"]))
print("validation AUC: MELD %.3f (%.3f-%.3f) | panel %.3f (%.3f-%.3f)"
      % (s["auc_meld_median"], *s["auc_meld_range"], s["auc_panel_median"], *s["auc_panel_range"]))

# ---------------- plot ----------------
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(4.4, 3.2))
d = df.d_auc.to_numpy()
ax.hist(d, bins=22, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5)
ax.axvline(0, color=GREY, ls="--", lw=0.9)
ax.axvline(np.median(d), color=AMBER, lw=1.2)
ax.set_xlabel("$\\Delta$AUC at 90 days (panel − MELD), held-out centres", fontsize=8)
ax.set_ylabel("Number of random centre splits", fontsize=8)
ax.set_title("%d random 5/8 centre splits" % len(df), fontsize=9, loc="left")
ax.text(0.03, 0.96, "median %+.3f\n%.0f%% of splits > 0" % (np.median(d), 100 * (d > 0).mean()),
        transform=ax.transAxes, va="top", fontsize=7, color="#333333")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
p = BASE / "figures" / "figS1" / "panels" / "data"
p.mkdir(parents=True, exist_ok=True)
fig.savefig(p / "figS1_panelG_repeated_split.png", dpi=600, bbox_inches="tight")
fig.savefig(p / "figS1_panelG_repeated_split.pdf", bbox_inches="tight")
plt.close(fig)
print("plot ->", (p / "figS1_panelG_repeated_split.png").name)
