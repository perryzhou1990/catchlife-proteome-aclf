"""3-protein vs 5-protein panel: paired comparison on the same validation
patients (protocol: same derivative split, same seed, nothing refitted on
validation except the diagnostic refit reported separately).

Paired bootstrap resamples validation patients once per replicate and
evaluates BOTH models on that replicate, which is the correct way to test a
difference in AUC (two independent CIs overlapping says nothing).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
import reselect_panel as R

RNG = np.random.default_rng(20261106)
P3 = ["TNF.RI", "DR6", "Renin"]
P5 = ["TNF.RI", "DR6", "Renin", "SREC.II", "GH"]

dev, val = R.dev_all, R.val_all
y = val["event"].to_numpy()
print("derivation n=%d events=%d | validation n=%d events=%d"
      % (len(dev), int(dev["event"].sum()), len(val), int(y.sum())))

# ---------- fit on derivation, predict 90-day risk on validation ----------
models = {}
for name, cols in (("MELD", ["meld"]),
                   ("3-protein", ["meld"] + P3),
                   ("5-protein", ["meld"] + P5)):
    m = R.fit_cph(dev, cols)
    pr = R.prob90(m, val, cols, R.baseline_h90(m, dev["center"]))
    models[name] = dict(cols=cols, fit=m, risk=pr, ph=m.predict_partial_hazard(val[cols]).to_numpy())
    print("%-10s AUC90 %.4f | Brier %.4f | mean pred %.3f"
          % (name, roc_auc_score(y, pr), brier_score_loss(y, pr), pr.mean()))

# ---------- calibration slope on validation (1.0 = perfect) ----------
print("\ncalibration slope on validation (linear predictor -> outcome):")
for name in ("MELD", "3-protein", "5-protein"):
    lp = np.log(models[name]["ph"])
    vv = val[["time", "event"]].copy()
    vv["lp"] = lp
    try:
        c = R.fit_cph(vv, ["lp"], strata=False)
        print("  %-10s slope %.3f (95%%CI %.3f to %.3f)"
              % (name, c.params_["lp"], c.confidence_intervals_.loc["lp"].iloc[0],
                 c.confidence_intervals_.loc["lp"].iloc[1]))
    except Exception as e:
        print("  %-10s failed %s" % (name, type(e).__name__))

# ---------- paired bootstrap: difference in AUC90 ----------
pairs = [("3-protein", "MELD"), ("5-protein", "MELD"), ("5-protein", "3-protein")]
boot = {k: [] for k in ("3vM", "5vM", "5v3")}
key = {("3-protein", "MELD"): "3vM", ("5-protein", "MELD"): "5vM", ("5-protein", "3-protein"): "5v3"}
for _ in range(2000):
    s = RNG.choice(len(y), len(y), replace=True)
    if not (5 < y[s].sum() < len(s) - 5):
        continue
    a3 = roc_auc_score(y[s], models["3-protein"]["risk"][s])
    a5 = roc_auc_score(y[s], models["5-protein"]["risk"][s])
    am = roc_auc_score(y[s], models["MELD"]["risk"][s])
    boot["3vM"].append(a3 - am)
    boot["5vM"].append(a5 - am)
    boot["5v3"].append(a5 - a3)

rows = []
for a, b in pairs:
    k = key[(a, b)]
    d = np.array(boot[k])
    est = roc_auc_score(y, models[a]["risk"]) - roc_auc_score(y, models[b]["risk"])
    rows.append(dict(comparison=f"{a} vs {b}", dAUC=est,
                     lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                     p=2 * min((d <= 0).mean(), (d >= 0).mean()),
                     P_gain_gt_0=float((d > 0).mean())))
pairs_df = pd.DataFrame(rows)
print("\npaired bootstrap (2,000 resamples) on the same validation patients:")
print(pairs_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------- NRI / IDI: 5-protein vs 3-protein ----------
def nri_idi(old, new, yy):
    up, dn = new > old, new < old
    c, k = yy == 1, yy == 0
    return ((up[c].mean() - dn[c].mean()) + (dn[k].mean() - up[k].mean()),
            (new[c].mean() - old[c].mean()) - (new[k].mean() - old[k].mean()))

r3, r5 = models["3-protein"]["risk"], models["5-protein"]["risk"]
nri, idi = nri_idi(r3, r5, y)
nb = []
for _ in range(2000):
    s = RNG.choice(len(y), len(y), replace=True)
    if not (5 < y[s].sum() < len(s) - 5):
        continue
    nb.append(nri_idi(r3[s], r5[s], y[s]))
nri_b = np.array([x[0] for x in nb]); idi_b = np.array([x[1] for x in nb])
print("\n5-protein vs 3-protein, validation: NRI %.3f (%.3f to %.3f) | IDI %.4f (%.4f to %.4f)"
      % (nri, np.percentile(nri_b, 2.5), np.percentile(nri_b, 97.5),
         idi, np.percentile(idi_b, 2.5), np.percentile(idi_b, 97.5)))

# ---------- decision curve: net benefit difference ----------
def net_benefit(risk, yy, pt):
    pred = risk >= pt
    tp = (pred & (yy == 1)).sum()
    fp = (pred & (yy == 0)).sum()
    return tp / len(yy) - fp / len(yy) * pt / (1 - pt)

dc = []
for pt in np.arange(0.10, 0.46, 0.025):
    b3, b5, bm = (net_benefit(models[k]["risk"], y, pt) for k in ("3-protein", "5-protein", "MELD"))
    dc.append(dict(threshold=round(pt, 3), nb_meld=bm, nb_3=b3, nb_5=b5,
                   diff_5_minus_3=b5 - b3, diff_5_minus_meld=b5 - bm))
dc = pd.DataFrame(dc)
print("\ndecision curve (validation), selected thresholds:")
print(dc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("mean net benefit: MELD %.4f | 3P %.4f | 5P %.4f"
      % (dc.nb_meld.mean(), dc.nb_3.mean(), dc.nb_5.mean()))

# ---------- does the 5-protein model hold up when refitted on validation? ----------
print("\ndiagnostic refit of MELD + 5 proteins WITHIN the validation set "
      "(not a validation of performance; shows whether the added terms carry signal):")
mv = R.fit_cph(val, ["meld"] + P5)
sm = mv.summary[["coef", "exp(coef)", "p"]].copy()
sm["HR"] = sm["exp(coef)"]
print(sm[["coef", "HR", "p"]].to_string(float_format=lambda x: f"{x:.4g}"))
md = R.fit_cph(dev, ["meld"] + P5).summary
sm["HR_derivation"] = md["exp(coef)"]
sm["P_derivation"] = md["p"]
print("\nsame terms, derivation vs validation:")
print(sm[["HR_derivation", "P_derivation", "HR", "p"]].to_string(float_format=lambda x: f"{x:.4g}"))

# ---------- derivation-set nested LRT: does 5 beat 3 in derivation? ----------
m3, m5 = R.fit_cph(dev, ["meld"] + P3), R.fit_cph(dev, ["meld"] + P5)
lr = 2 * (m5.log_likelihood_ - m3.log_likelihood_)
from scipy.stats import chi2
print("\nderivation-set nested LRT (5 vs 3, 2 df): LR %.2f, P=%.3g" % (lr, chi2.sf(lr, 2)))
print("AIC: 3-protein %.2f | 5-protein %.2f"
      % (-2 * m3.log_likelihood_ + 2 * len(m3.params_), -2 * m5.log_likelihood_ + 2 * len(m5.params_)))

pairs_df.to_csv("output2/panel_3v5_paired.csv", index=False, encoding="utf-8-sig")
dc.to_csv("output2/panel_3v5_dca.csv", index=False, encoding="utf-8-sig")
