"""Check whether the CV optimum sits on the lower edge of the previous grid
(0.01) and, if so, whether lambda_1se - and hence the stability-selection
environment and the locked panel - move.

Only the CV curve is recomputed (no bootstrap), grid extended to 0.002.
"""
import numpy as np
import pandas as pd
import reselect_panel as R

scr = pd.read_csv("output2/screen_dev_primary883.csv")
pool = scr.loc[scr.FDR < 0.05, "protein"].tolist()
print("pool", len(pool), "| dev n=%d events=%d"
      % (len(R.dev_all), int(R.dev_all["event"].sum())))

lam_grid = np.logspace(np.log10(0.002), np.log10(0.5), 34)
ev = R.dev_all["event"].to_numpy()
folds = np.zeros(len(R.dev_all), int)
for e in (0, 1):
    idx = np.where(ev == e)[0]
    R.RNG.shuffle(idx)
    folds[idx] = np.arange(len(idx)) % 10

rows = []
for lam in lam_grid:
    lls = []
    for k in range(10):
        tr = R.dev_all.iloc[folds != k]
        te = R.dev_all.iloc[folds == k]
        if te["event"].sum() < 2:
            continue
        try:
            c = R.fit_cph(tr, pool, lam=lam)
            # lifelines' held-out log partial likelihood (same as reselect_panel)
            lls.append(float(c.score(te, scoring_method="log_likelihood")))
        except Exception:
            continue
    if lls:
        rows.append(dict(lam=lam, ll_mean=float(np.mean(lls)), ll_sd=float(np.std(lls)), n_fold=len(lls)))

cv = pd.DataFrame(rows)
cv.to_csv("output2/cv_curve_primary883_extended.csv", index=False, encoding="utf-8-sig")
print(cv.round(5).to_string(index=False))

best = cv.loc[cv.ll_mean.idxmax()]
se = best.ll_sd / np.sqrt(best.n_fold)
thr = best.ll_mean - se
lam_1se = float(cv[cv.ll_mean >= thr].lam.max())
print("\nlambda_cv = %.5f (grid min %.5f, max %.5f)"
      % (best.lam, cv.lam.min(), cv.lam.max()))
print("1-SE threshold LL %.5f -> lambda_1se = %.5f" % (thr, lam_1se))
print("previous: lambda_cv 0.01000 / lambda_1se 0.01963")
