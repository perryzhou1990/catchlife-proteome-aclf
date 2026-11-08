# -*- coding: utf-8 -*-
"""Fig 2 (supplementary panel) — GO/KEGG enrichment of ACLF-associated proteins.
Up: beta>0 & FDR<0.05 (higher in ACLF on admission); Down: beta<0 & FDR<0.05.
Mapping: protein -> geneID from 宋晶 annotation CSVs (883 proteins).
Output: output/fig2_enrichment.pdf/.png, matrices/enrichment_up.csv, enrichment_down.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gseapy as gp

BASE = Path(__file__).resolve().parents[3]
OUT = BASE / "figures" / "fig2"
ANNOT = Path(r"D:\科研\博士课题\蛋白质组多中心数据-to 周沛\宋晶分组分析结果-供参考\cirrhosis_vs_noncirrhosis.csv")

vol = pd.read_csv(OUT / "matrices" / "volcano_aclfadmission_vs_noaclf.csv")
annot = pd.read_csv(ANNOT)[["protein", "geneID"]].drop_duplicates("protein")
v = vol.merge(annot, on="protein", how="left")
print("unmapped genes:", int(v["geneID"].isna().sum()))

up = v.loc[(v["beta"] > 0) & (v["FDR"] < 0.05) & v["geneID"].notna(), "geneID"].dropna().unique().tolist()
dn = v.loc[(v["beta"] < 0) & (v["FDR"] < 0.05) & v["geneID"].notna(), "geneID"].dropna().unique().tolist()
bg = v["geneID"].dropna().unique().tolist()
print(f"up {len(up)}, down {len(dn)}, background {len(bg)}")

res = {}
for name, genes in [("up", up), ("dn", dn)]:
    try:
        e = gp.enrichr(genes, gene_sets=["GO_Biological_Process_2023", "KEGG_2021_Human"],
                       organism="human", outdir=None, no_plot=True)
        r = e.results
        print(f"{name}: enrichr returned {len(r)} terms; cols={list(r.columns)[:3]}")
        if r.empty:
            res[name] = pd.DataFrame(columns=["Term", "Overlap", "Adjusted P-value", "Genes", "Direction"])
            continue
        t = r[["Term", "Overlap", "Adjusted P-value", "Genes"]].copy()
        t["Direction"] = name
        res[name] = t
        t.to_csv(OUT / "matrices" / f"enrichment_{name}.csv", index=False)
    except Exception as ex:
        print(f"{name} enrichment failed: {ex}")
        res[name] = pd.DataFrame()

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
for ax, name, color, title in [(axs[0], "up", "#C0504D", "Up in ACLF on admission"),
                               (axs[1], "dn", "#4E79A7", "Down in ACLF on admission")]:
    t = res[name]
    ax.text(-0.24, 1.06, "E" if name == "up" else "F", transform=ax.transAxes,
            fontsize=12, fontweight="bold")
    if t.empty:
        ax.axis("off"); ax.text(0.5, 0.5, "enrichment unavailable", ha="center")
        continue
    top = t.nsmallest(10, "Adjusted P-value").iloc[::-1]
    top = top.assign(label=[x.split(" (")[0][:52] for x in top["Term"]])
    ax.barh(top["label"], -np.log10(top["Adjusted P-value"]), color=color, alpha=0.85, height=0.62)
    ax.set_xlabel("-log10 (adjusted p)", fontsize=7)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Figure 2 (addendum). Pathway enrichment of ACLF-associated plasma proteins (Enrichr: GO BP + KEGG)",
             fontsize=9.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "output" / "fig2_enrichment.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "output" / "fig2_enrichment.png", dpi=300, bbox_inches="tight")
if not res["up"].empty:
    print(res["up"].nsmallest(8, "Adjusted P-value")[["Term", "Overlap", "Adjusted P-value"]].to_string(index=False))
    print(res["dn"].nsmallest(8, "Adjusted P-value")[["Term", "Overlap", "Adjusted P-value"]].to_string(index=False))
print("DONE")
