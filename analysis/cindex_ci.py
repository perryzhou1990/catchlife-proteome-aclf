# -*- coding: utf-8 -*-
"""Bootstrap CIs for the C-index contrasts quoted in Results 5.4.

The AUC contrasts already carry percentile CIs from fig4_make.py (2,000
paired resamples).  The C-index contrasts did not, and 5.4 therefore said
"95%CI not reported".  This script fills that gap without re-deriving
anything: it executes the data section of fig4_make.py verbatim (so the
cohort, split, panel and model fitting are identical to the figure), then
bootstraps the validation set and reports percentile CIs for

    dC(MELD+5P vs MELD), dC(MELD+3P vs MELD), dC(5P vs 3P)

plus the 46-event power note (what delta C a 46-event set can resolve).

Output: analysis/output2/panel_3v5_cindex.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "figures" / "fig4" / "code" / "fig4_make.py"
SPLIT = "# ---------------- figure ----------------"

text = SRC.read_text(encoding="utf-8")
data_part = text[: text.index(SPLIT)]
ns = {"__name__": "__panel_data__", "__file__": str(SRC)}
exec(compile(data_part, str(SRC), "exec"), ns)

fit, val, dev, P5, P3 = ns["fit"], ns["val"], ns["dev"], ns["P5"], ns["P3"]
RNG = np.random.default_rng(20261106)

models = {"MELD": ["meld"], "MELD+3P": ["meld"] + P3, "MELD+5P": ["meld"] + P5}
lp = {}
for name, cols in models.items():
    m = fit(dev, cols)
    lp[name] = np.log(m.predict_partial_hazard(val[cols]).to_numpy())

t = val.time.to_numpy()
e = val.event.to_numpy()


def cidx(idx, key):
    return concordance_index(t[idx], -lp[key][idx], e[idx])


point = {k: cidx(np.arange(len(t)), k) for k in lp}
print("validation C-index:", {k: round(v, 4) for k, v in point.items()})

pairs = [("MELD+3P", "MELD"), ("MELD+5P", "MELD"), ("MELD+5P", "MELD+3P")]
draws = {f"{a} vs {b}": [] for a, b in pairs}
B = 2000
n = len(t)
for _ in range(B):
    s = RNG.choice(n, n, replace=True)
    if not (5 < e[s].sum() < n - 5):
        continue
    for a, b in pairs:
        draws[f"{a} vs {b}"].append(cidx(s, a) - cidx(s, b))

rows = []
for a, b in pairs:
    v = np.array(draws[f"{a} vs {b}"])
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    rows.append(dict(contrast=f"{a} vs {b}", dC=point[a] - point[b],
                     lo=lo, hi=hi, p=p, n_boot=len(v)))
    print("%-22s dC %+.4f (%.4f, %.4f) P=%.3f" % (f"{a} vs {b}", point[a] - point[b], lo, hi, p))

# how much of a C-index difference 46 events can resolve, for the Discussion
se_approx = np.sqrt(0.25 / max(1, int(e.sum())))
rows.append(dict(contrast="resolution of a 46-event validation set (approx. SE of C)",
                 dC=float("nan"), lo=float("nan"), hi=float("nan"),
                 p=float("nan"), n_boot=int(e.sum())))
print("approximate SE of C with 46 events ~ %.3f" % se_approx)

out = pd.DataFrame(rows)
out.to_csv(BASE / "analysis" / "output2" / "panel_3v5_cindex.csv",
           index=False, encoding="utf-8-sig")
print("wrote analysis/output2/panel_3v5_cindex.csv")
