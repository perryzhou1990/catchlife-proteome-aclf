# -*- coding: utf-8 -*-
"""Fig 2 — Proteomic landscape across the AoCLD disease gradient (CATCH-LIFE, n=1390)
A  PCA before/after center adjustment (colored by ACLF status; center variance annotated)
B  Volcano: ACLF on admission vs No ACLF (linear model, sex + center adjusted, BH-FDR)
C  Gradient heatmap: top proteins across 4 gradient groups (limma-style trend)
D  Co-expression modules (WGCNA-style) x clinical traits + hub proteins
Outputs: output/fig2.pdf/.png, matrices/ (pca.csv, volcano_90d_gradient.csv,
         gradient_means.csv, module_trait.csv, hubs.csv), candidate_proteins.tsv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

BASE = Path(__file__).resolve().parents[3]
CLIN = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\CATCH-LIFE临床队列数据_1390人.xlsx")
PROT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\蛋白质组数据及临床信息\proteinarray_matrix_1390人蛋白质组结果（质控后883蛋白）.xlsx")
OUT = BASE / "figures" / "fig2"
for sub in ["raw_data", "matrices", "code", "output"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# ---------------- load & QC ----------------
clin = pd.read_excel(CLIN, sheet_name="队列数据1390人")
clin.columns = [str(c).strip() for c in clin.columns]
clin["center"] = clin["ID"].astype(str).str.split("-").str[0]
prot = pd.read_excel(PROT, sheet_name="1390已测蛋白质组")
prot.columns = [str(c).strip() for c in prot.columns]
prot = prot.set_index(prot.columns[0])
prot = prot.loc[clin["ID"]].astype(float)          # align rows

n_raw = prot.shape[1]
floor_frac = (prot == prot.min()).mean()
# Protocol v1.2: no detection filter - all 883 quantified proteins enter.
kept = prot.columns
log2 = np.log2(prot.clip(lower=1))
Z = (log2 - log2.mean()) / log2.std(ddof=1)
print(f"QC: {n_raw} -> {Z.shape[1]} proteins (no detection filter, protocol v1.2)")

# ---------------- design matrices ----------------
def design(extra: pd.DataFrame, with_center=True):
    parts = [pd.Series(1.0, index=clin.index, name="int")]
    for c in extra.columns:
        parts.append(extra[c].astype(float).rename(c))
    if with_center:
        parts += [pd.get_dummies(clin["center"], drop_first=True).astype(float)]
    return pd.concat(parts, axis=1)

sex = clin["Gender"].replace(2, np.nan)  # 1 coding artifact -> NaN
sex = sex.fillna(sex.mode()[0])

# center-adjusted matrix: residualize on center (+sex)
Xc = design(pd.DataFrame({"sex": sex}), with_center=True)
Xc_np = Xc.to_numpy(float)
pinvXc = np.linalg.pinv(Xc_np)
Y = Z.to_numpy(float)
Beta = pinvXc @ Y
Resid = Y - Xc_np @ Beta
Zadj = pd.DataFrame(Resid + Beta[0], index=Z.index, columns=Z.columns)
Zadj = (Zadj - Zadj.mean()) / Zadj.std(ddof=1)

# ---------------- PCA ----------------
def pca_scores(M):
    Mc = M - M.mean(0)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    return U[:, :2] * S[:2], (S**2 / (S**2).sum())[:2]

sc_raw, evr_raw = pca_scores(Y)
sc_adj, evr_adj = pca_scores(Zadj.to_numpy())
COL_ACLF = "COSSH_DAY(0.non-ACLF1.ACLFonadmission2.developedACLFduringhospitalization)_张艳"
grp_col = clin[COL_ACLF]
colors = {0: "#7FA6C9", 1: "#C0504D", 2: "#E8A33D"}
def center_r2(scores):
    Xd = pd.get_dummies(clin["center"], drop_first=True).astype(float)
    Xd = np.column_stack([np.ones(len(Xd)), Xd])
    B = np.linalg.pinv(Xd) @ scores
    r = scores - Xd @ B
    return 1 - r.var(0) / scores.var(0)
r2_raw, r2_adj = center_r2(sc_raw), center_r2(sc_adj)
pd.DataFrame({"PC1": sc_adj[:, 0], "PC2": sc_adj[:, 1], "aclf": grp_col,
              "center": clin["center"]}).to_csv(OUT / "matrices" / "pca.csv", index=False)

# ---------------- differential: ACLF on admission vs No ACLF (gradient model) ----------------
gnum = clin[COL_ACLF].astype(float)
Xg = design(pd.DataFrame({"sex": sex, "aclf_adm": (gnum == 1).astype(float)}))
Xg_np = Xg.to_numpy(float)
p = Xg_np.shape[1]
pinvXg = np.linalg.pinv(Xg_np)
Bt = pinvXg @ Y
res = Y - Xg_np @ Bt
dof = len(clin) - np.linalg.matrix_rank(Xg_np)
sigma2 = (res**2).sum(0) / dof
XtXinv = np.linalg.pinv(Xt := Xg_np.T @ Xg_np)
se = np.sqrt(np.outer(np.diag(XtXinv), sigma2))
j = list(Xg.columns).index("aclf_adm")
tv = Bt[j] / se[j]
pv = 2 * stats.t.sf(np.abs(tv), dof)
# proper Benjamini-Hochberg: sort p, scale by m/rank, enforce monotonicity, unsort
_pp = np.clip(pv, 1e-300, 1)
_o = np.argsort(_pp)
_ranked = _pp[_o] * len(_pp) / (np.arange(len(_pp)) + 1)
_q = np.minimum.accumulate(_ranked[::-1])[::-1].clip(max=1)
fdr = np.empty_like(_q); fdr[_o] = _q
log2fc = Bt[j]  # z-scale (already standardized) ~ per-SD difference
vol = pd.DataFrame({"protein": Z.columns, "beta": log2fc, "t": tv, "p": pv, "FDR": fdr})
vol.to_csv(OUT / "matrices" / "volcano_aclfadmission_vs_noaclf.csv", index=False)
n_sig = int((fdr < 0.05).sum())
print(f"Volcano: ACLF on admission vs No ACLF -> {n_sig} proteins FDR<0.05")

# ---------------- gradient groups ----------------
cirr = clin["cirrhosis"].astype(float)
groups = pd.Series(np.select(
    [gnum == 1, gnum == 2, (gnum == 0) & (cirr == 1), (gnum == 0) & (cirr == 0)],
    ["G4 ACLF on admission", "G3 developed ACLF", "G2 cirrhosis AD", "G1 acute liver injury"],
    default=""), index=clin.index)
order = ["G1 acute liver injury", "G2 cirrhosis AD", "G3 developed ACLF", "G4 ACLF on admission"]
gmeans = Zadj.groupby(groups.to_numpy())[Zadj.columns].mean().reindex(order)
gmeans.to_csv(OUT / "matrices" / "gradient_means.csv")

# trend statistic: |corr(protein, ordinal gradient)|  (numpy-based, pandas .corr unreliable here)
ordv = groups.map({o: i for i, o in enumerate(order)}).astype(float).to_numpy()
Yz = Zadj.to_numpy()
trend_r = np.array([np.corrcoef(Yz[:, j], ordv)[0, 1] for j in range(Yz.shape[1])])
trend_s = pd.Series(trend_r, index=Zadj.columns)
top30 = trend_s.reindex(trend_s.abs().sort_values(ascending=False).head(30).index).index
print("trend NaNs:", int(np.isnan(trend_r).sum()))

# candidate list for downstream figures (GR9-c')
cand = vol[vol["FDR"] < 0.05].assign(abs_beta=lambda d: d["beta"].abs()).sort_values("abs_beta", ascending=False)
cand_out = cand[["protein", "beta", "t", "p", "FDR"]]
cand_out.to_csv(BASE / "figures" / "fig2" / "matrices" / "candidate_proteins.tsv", sep="\t", index=False)

# ---------------- WGCNA-style modules ----------------
from sklearn.cluster import KMeans
km = KMeans(n_clusters=8, n_init=20, random_state=42)
labs = km.fit_predict(Zadj.to_numpy().T) + 1
mod = pd.Series(labs, index=Zadj.columns)
sizes = mod.value_counts()
mod[mod.isin(sizes[sizes < 10].index)] = 0          # grey
mod = mod.replace(0, 0).rename("module")
palette = ["#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#B07AA1",
           "#76B7B2", "#EDC948", "#9C755F", "#BAB0AC", "#FF9DA7"]
traits = pd.DataFrame({
    "ACLF on adm": (gnum == 1).astype(float), "Developed ACLF": (gnum == 2).astype(float),
    "Cirrhosis": cirr, "HBV": clin["HBV"].astype(float),
    "Infection": clin["AD_infection(0.no1.yes)"].astype(float),
    "90d death/tx": clin["O90_2020.10更新后"].isin([1, 2]).astype(float),
}, index=clin.index)
rows = []
for m in sorted(mod[mod > 0].unique()):
    mem = Zadj.columns[mod == m]
    eg = np.linalg.svd(Zadj[mem] - Zadj[mem].mean())[0][:, 0]
    def nc(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1])
    r = nc(eg, ordv)
    row = {"module": m, "n": len(mem), "trend_r": r}
    for t in traits.columns:
        row[t] = nc(eg, traits[t].to_numpy())
    # hub proteins: highest |corr to eigengene|
    egS = pd.Series(eg, index=Zadj.index)
    cs = Zadj[mem].apply(lambda c: float(np.corrcoef(c.to_numpy(), eg)[0, 1]))
    row["hubs"] = "; ".join(cs.abs().sort_values(ascending=False).head(3).index)
    rows.append(row)
mt = pd.DataFrame(rows).sort_values("trend_r", key=lambda s: s.abs(), ascending=False)
mt.to_csv(OUT / "matrices" / "module_trait.csv", index=False)
mod.to_csv(OUT / "matrices" / "module_membership.csv")
print("modules:", mt[["module", "n", "trend_r", "hubs"]].to_string(index=False))

# ---------------- figure ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig = plt.figure(figsize=(8.4, 8.8))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.3], hspace=0.50, wspace=1.35)
axA1 = fig.add_subplot(gs[0, 0:3]); axA2 = fig.add_subplot(gs[0, 3:6], sharey=axA1)
axB = fig.add_subplot(gs[1, 0:2]); axC = fig.add_subplot(gs[1, 2:4]); axD = fig.add_subplot(gs[1, 4:6])
plt.setp(axA2.get_yticklabels(), visible=False)

# A: PCA before/after
def draw_pca(ax, sc, evr, r2, title, show_leg=False):
    for g, lab in [(0, "No ACLF"), (1, "ACLF on admission"), (2, "Developed ACLF")]:
        m = (grp_col == g).to_numpy()
        ax.scatter(sc[m, 0], sc[m, 1], s=3, alpha=0.45, color=colors[g], label=lab,
                   edgecolors="none")
    ax.set_title(f"{title}\ncenter explains PC1 {max(r2[0],0)*100:.0f}% / PC2 {max(r2[1],0)*100:.0f}%",
                 fontsize=7)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=7)
    if show_leg:
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=7)
    if show_leg:
        ax.legend(fontsize=5, loc="upper left", frameon=False, handletextpad=0.2)
    ax.tick_params(labelsize=6)
draw_pca(axA1, sc_raw, evr_raw, r2_raw, "Before adjustment", show_leg=True)
draw_pca(axA2, sc_adj, evr_adj, r2_adj, "After adjustment")
axA1.text(-0.22, 1.12, "A", transform=axA1.transAxes, fontsize=12, fontweight="bold")

# B: volcano
axB.text(-0.35, 1.06, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
sig = vol["FDR"] < 0.05
axB.scatter(vol.loc[~sig, "beta"], -np.log10(vol.loc[~sig, "p"]), s=3, color="#BBBBBB", alpha=0.6)
axB.scatter(vol.loc[sig, "beta"], -np.log10(vol.loc[sig, "p"]), s=5, color="#C0504D", alpha=0.8)
lab_set = pd.concat([cand.head(6), vol[vol["protein"].isin(["TNF.R1", "ANG.2", "Renin", "HCC.1", "S100A8", "VEGFR2", "NKG2D"])]]
                    ).drop_duplicates("protein")
for _, r in lab_set.iterrows():
    axB.annotate(r["protein"].replace(".", "-"), (r["beta"], -np.log10(r["p"])),
                 fontsize=5.5, color="#333333")
axB.axhline(-np.log10(0.05), color="#888888", lw=0.6, ls="--")
axB.set_xlabel("β (per-SD difference, ACLF on admission vs No ACLF)", fontsize=7)
axB.set_ylabel("-log10(p)", fontsize=7)
axB.set_title(f"{n_sig} proteins FDR<0.05", fontsize=8)
axB.tick_params(labelsize=6.5)

# C: gradient heatmap
axC.text(-0.45, 1.06, "C", transform=axC.transAxes, fontsize=12, fontweight="bold")
H = gmeans[top30].to_numpy().T  # (4 groups x 30 proteins)
im = axC.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2)
axC.set_xticks(range(4), ["G1\nacute liver\ninjury", "G2\ncirrhosis\nAD", "G3\ndeveloped\nACLF", "G4\nACLF\non adm"], fontsize=5)
axC.set_yticks(range(len(top30)), [p.replace(".", "-") for p in top30], fontsize=4.6)
axC.set_title("Top 30 gradient proteins\n(group means, z)", fontsize=7.5)
plt.colorbar(im, ax=axC, fraction=0.045, pad=0.02)

# D: module-trait heatmap
axD.text(-0.45, 1.06, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")
tcols = ["trend_r"] + list(traits.columns)
M = mt[tcols].to_numpy(float)
im2 = axD.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
axD.set_xticks(range(len(tcols)), ["Grad.\ntrend", "ACLF\nadm", "Dev.\nACLF", "Cirr.", "HBV", "Infect.", "90d\nd/t"], fontsize=4.6)
axD.set_yticks(range(len(mt)), [f"M{m} (n={n})" for m, n in zip(mt["module"], mt["n"])], fontsize=5)
for i in range(len(mt)):
    for jj, c in enumerate(tcols):
        v = M[i, jj]
        if np.isfinite(v) and abs(v) > 0.30:
            axD.text(jj, i, f"{v:.2f}", ha="center", va="center", fontsize=4.2,
                     color="white" if abs(v) > 0.45 else "black")
axD.set_title("Modules vs clinical traits (r)", fontsize=7.5)
plt.colorbar(im2, ax=axD, fraction=0.045, pad=0.02)

fig.suptitle("Figure 2. Plasma proteomic landscape across the AoCLD disease gradient",
             fontsize=10.5, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "output" / "fig2.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "output" / "fig2.png", dpi=300, bbox_inches="tight")
print("Top gradient proteins:", list(top30[:12]))
print("hubs:\n", mt[["module", "n", "hubs"]].to_string(index=False))
print("DONE")
