# -*- coding: utf-8 -*-
"""Fig 6 D: cell-type localisation of panel genes in a public human liver atlas (CELLxGENE).
Dataset: 'Healthy human liver: integrated', 73,295 cells, 4 donors, 12 cell types, normal tissue.
Output: matrices/scrna_expression.csv (cell_type x gene: pct expressing, mean log-normalised)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

BASE = Path(__file__).resolve().parents[3]
FIG6 = BASE / "figures" / "fig6"
RAW, MAT = FIG6 / "raw_data", FIG6 / "matrices"
MAT.mkdir(parents=True, exist_ok=True)

# Rev 2026-11-07: GH1 added. It is a panel protein and was previously omitted from
# the query list, so the fifth panel member had no cellular assignment at all.
GENES = ["TNFRSF1A", "REN", "TNFRSF21", "SCARF2", "GH1", "ANGPT2", "CCL14", "CRP"]

a = ad.read_h5ad(RAW / "liver_integrated.h5ad")
print("shape", a.shape)
sym2id = {s: i for i, s in zip(a.var_names, a.var["feature_name"].astype(str))}
genes = [g for g in GENES if g in sym2id]
missing = [g for g in GENES if g not in sym2id]
print("genes found", genes, "| missing", missing)

# use raw (unfiltered) counts: several panel genes were filtered out of .X as undetected
X = a.raw[:, [sym2id[g] for g in genes]].X
X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
counts = np.asarray(a.raw.X.sum(axis=1)).ravel().astype(float)
norm = np.log1p(X / np.maximum(counts, 1)[:, None] * 1e4)

ct = a.obs["cell_type"].astype(str).to_numpy()
rows = []
for c in pd.unique(ct):
    m = ct == c
    for j, g in enumerate(genes):
        v = norm[m, j]
        rows.append(dict(cell_type=c, n_cells=int(m.sum()), gene=g,
                         pct_expr=float((v > 0).mean() * 100),
                         mean_expr=float(v.mean())))
df = pd.DataFrame(rows)
df.to_csv(MAT / "scrna_expression.csv", index=False)

# per-gene top cell types
print("\n=== top cell types per gene (by mean expression) ===")
for g in genes:
    sub = df[df.gene == g].sort_values("mean_expr", ascending=False).head(4)
    print(g, " | ".join(f"{r.cell_type}:{r.mean_expr:.2f}({r.pct_expr:.0f}%)" for r in sub.itertuples()))

# overall ranking of cell types by summed z-scored expression of the 4 core genes
core = [g for g in ["TNFRSF1A", "REN", "TNFRSF21", "SCARF2"] if g in genes]
piv = df[df.gene.isin(core)].pivot(index="cell_type", columns="gene", values="mean_expr")
z = (piv - piv.mean()) / piv.std(ddof=0)
rank = z.mean(axis=1).sort_values(ascending=False)
print("\n=== core-gene enrichment by cell type (mean z) ===")
print(rank.round(2).to_string())
rank.to_csv(MAT / "scrna_celltype_rank.csv", header=["mean_z_core"])
print("DONE")
