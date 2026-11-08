# CATCH-LIFE proteome ACLF — analysis code

Code for: **Five plasma proteins refine 90-day risk stratification in acute decompensation of chronic liver disease** (manuscript under review).

Proteomic profiling (1,390 patients, 883 proteins, Olink platform) and clinical data are from the
CATCH-LIFE cohort. **Individual-level clinical and proteomic data are not part of this repository**;
they are available from the corresponding author under a data-use agreement approved by the lead
ethics committee (see the manuscript's Data availability statement). The scripts here document and
reproduce every analysis and figure from the de-identified analysis tables.

## Repository layout

| Path | Content |
|---|---|
| `analysis/make_v03.py` | Primary pipeline: matrix assembly, residualisation, per-protein Cox models with BH-FDR, k-means modules, LASSO-Cox with 300-resample stability selection, locking rules, derivation/validation fits |
| `analysis/reselect_panel.py` | Candidate-pool re-selection across floor-frequency thresholds; Jaccard stability |
| `analysis/robustness2.py`, `repeated_centre_split.py` | 200 random centre allocations and summaries |
| `analysis/run_8dev_stage.py` | Reversed split sensitivity analysis (8 smaller centres derive, 5 larger test) |
| `analysis/cindex_ci.py` | Paired bootstrap (2,000 resamples) CIs for C-index/AUC differences |
| `analysis/compare_3v5.py`, `panel_5v3.py`, `compare_clif_variants.py` | 3- vs 5-protein comparisons; CLIF-C OF comparator sensitivity |
| `analysis/07_coloc_bug_check.py`, `08_steiger_presso.py` | cis-pQTL MR: colocalisation audit, Steiger directionality, MR-PRESSO |
| `analysis/09_druggability_assessment.py` | ChEMBL / Open Targets / UniProt / ClinicalTrials.gov annotation |
| `analysis/10_review_response_checks.py`, `11_second_review_checks.py`, `12_ph_recheck.py` | Sensitivity and robustness checks (CRP models, win ratio, DCA, leave-one-centre-out, calibration, PH residuals) |
| `analysis/probe_incident_aclf.py` et al. | Incident-ACLF and composite-endpoint screens |
| `analysis/score_engine.py`, `scores_clif.py`, `build_scores_corrected.py` | Reference implementations of all clinical scores (MELD, MELD-Na, MELD 3.0, CLIF-C OF, COSSH-ACLF II, ...) with published bounds and self-tests |
| `analysis/table2_new.py`, `panel_increment.py`, `age_check.py`, `discrimination_strict.py` | Table 2 and increment analyses |
| `figures/` | Panel-level figure generation (matplotlib; vector PDF plus 300 dpi PNG export) |
| `analysis/genesets/` | Public gene-set libraries (GO BP 2023, KEGG 2021, Reactome 2022) used for enrichment |

A script-level manifest mapping each script to the manuscript table/figure it produces is given in
Supplementary Table S8 of the manuscript.

## Requirements

Python 3.10+ with:

```
pandas  numpy  scipy  scikit-learn  statsmodels  lifelines  matplotlib  openpyxl
```

Install with:

```
pip install -r requirements.txt
```

## Usage

Each script is standalone and expects the de-identified analysis tables (cohort clinical table,
normalised protein matrix, age/score table) at the paths recorded in the script header. Outputs are
written to `analysis/output2/` and `figures/fig*/output/`. Running the full pipeline
(`analysis/make_v03.py`) reproduces the locked five-protein panel and all model fits; individual
figure scripts regenerate each main and supplementary figure.

## License

MIT (code). Gene-set libraries in `analysis/genesets/` are from Enrichr and remain under their
respective original licenses.

## Citation

Manuscript under review. Upon acceptance the code will be archived with a DOI (Zenodo); please cite
the paper and the archived version.
