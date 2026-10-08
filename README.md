# CNN vs Random Forest species distribution models — code

Code accompanying the M.Sc. thesis comparing a convolutional neural network with a
Random Forest for species distribution modelling under controlled, simulated
data-generating processes. Study area Germany, 100 m resolution, EPSG:32632 (UTM 32N).

**There are no occurrence records in this study.** Every species the models are fitted
to is simulated: a niche is defined on the first two principal components of a real
covariate stack, occurrence probability follows from it, and presences and absences are
drawn from that probability. The covariates are real; the species are not. This is what
makes the evaluation possible — the true suitability surface is known, so model
predictions are scored against it rather than against observed records.

## What is here

```
notebooks/   the two analysis notebooks
scripts/     the figure, export and diagnostic scripts (run after the notebooks)
exports/     the result tables reported in the thesis, as unrounded CSVs
hyperparameters/  the tuned CNN and Random Forest settings, so the tuner need not be re-run
environment.yml   the conda environment the analysis ran in
LICENSE      MIT
```

## The two notebooks

**`notebooks/data_processing.ipynb`** builds the covariate stack. It acquires and
harmonises fourteen candidate layers on a common 100 m grid, screens them for
collinearity (|r| > 0.7 and VIF > 5), and writes the retained layers as one array.

The screening removes Elevation (VIF 6.13, r = −0.82 with winter temperature), TRI and
Roughness as a pair (VIF 39.5 and 43.3, r = 0.99 — Roughness is kept), Population,
Railway density, Water distance, PA density and Slope. What is written out is
`env_stack_log.npy`, 8655 × 6410 × 7 float32, unscaled, with `log1p` applied to
Roughness, PA\_Distance and SummerPrecip, plus a `meta.json` describing it.

Seven layers, six of which are model predictors:

| # | covariate | source | native resolution |
|---|---|---|---|
| 0 | Roughness (3×3 focal SD of elevation) | Copernicus DEM GLO-30 | 30 m |
| 1 | Landuse (habitat-quality score 0–5) | ESA WorldCover 2021 v200 | 10 m |
| 2 | NDVI (median composite) | Landsat 8/9 Collection 2 Surface Reflectance | 30 m |
| 3 | Road density | OpenStreetMap, Geofabrik regional extracts | vector |
| 4 | Distance to protected areas | WDPA / WDOECM, protectedplanet.net | vector |
| 5 | Summer precipitation (BIO18) | CHELSA v2.1 | ~1 km |
| 6 | Winter temperature (BIO11) | CHELSA v2.1 | ~1 km |

Layer 1 is **not** a predictor. The experiment notebook takes it out of the stack and
uses only its tree-cover class as the binary woody field *W* of the configurational
data-generating process; the models never see it as an input. The six remaining layers
are what `env_scaled_100m` contains after min–max scaling.

Continuous rasters are brought onto the 100 m grid by bilinear resampling, the DEM by
`Resampling.average`, and ESA WorldCover by **majority (modal) resampling** — each 100 m
cell takes the class that covers the largest share of it, with the nodata sentinel
declared in a VRT so that it cannot win a majority.

`exports/covariates.csv` gives the full provenance per channel: variable definition,
period, resampling, transform, target grid, CRS and scaling.

**`notebooks/synthetic_experiment.ipynb`** is the experiment. It loads
`env_stack_log.npy`, does its own scaling, and runs:

| section | what it does |
|---|---|
| 1–4 | configuration, raster helpers, virtual-species generator, orthogonality check |
| 5–7 | CNN architecture, experimental framework, hyperparameter tuning |
| 8.1–8.4 | pointwise sensitivity axes: sample size (`8a`), prevalence (`8b`), resolution with a fixed 100 m truth (`8c-fixed`), patch size (`8e`) |
| 8.6–8.7 | the configurational data-generating process: shared machinery (`8f`) and run-level function (`8g`) |
| 8.8–8.9 | the same four axes under the configurational process (`8i`) |
| 8.10–8.11 | generalisation across a species ensemble along the blend parameter α (`8j`) |

It writes four result files in long format, one row per model fit:
`section8_two_regime_v6_bio11.csv`, `section8c_fixed_truth_v6_bio11.csv`,
`section8i_alpha_axes_v6_bio11.csv`, `section8j_species_ensemble_v6_bio11.csv`.

Both notebooks are stored **without outputs**, so what you see is the code that runs,
not a record of one past execution. Run them to regenerate the figures and tables.

## The scripts

They all read the four result CSVs; none of them refits a model. `_vs_env.py` is the
shared loader: it pulls the helper functions out of `synthetic_experiment.ipynb` by
marker string, so the scripts use the same definitions as the experiment rather than
copies of them.

**Figures.** `thesis_figures.py` holds the shared style — page width, fonts, colour and
marker conventions, legend construction — and draws the three introductory figures. The
rest draw one thesis figure each:

| script | figure |
|---|---|
| `thesis_fig_paired_contrast.py` | paired CNN − RF contrast, pointwise process |
| `thesis_fig_panels.py` | the four pointwise sensitivity axes |
| `thesis_fig_panels_8c_coarse.py` | resolution axis with the coarse-pixel AUC ceiling |
| `thesis_fig_panels_8i.py` | the four axes under the configurational process |
| `thesis_fig_paired_contrast_8i.py` | paired CNN − RF-patch contrast, configurational process |
| `thesis_fig_regime_comparison.py` | both processes side by side, extrapolative arm |
| `thesis_fig_8j.py` | the α blend, correlation-length sweep, oracle validation, W-source comparison |
| `thesis_fig_partitioning.py` | how the data are partitioned, in PC space and on the ground |

**Ceilings.** `coarse_auc_ceiling_8c.py` and `coarse_auc_ceiling_8i.py` compute the AUC
a model could reach knowing its coarse pixel perfectly. They regenerate the run's points
from its seeds and verify them against the stored fingerprint before computing anything.

**Dissimilarity (EDS).** `eds_band_drivers.py` reports the driver fields and their
per-band moments, `eds_band_robustness.py` sweeps K over every feature space on band
pixels, `eds_metric_properties.py` does the same on the drawn sample.

**Other diagnostics.** `configurational_orthogonality.py` measures how far the
configurational fields sit from the environmental axes, `pw_spread.py` the spread of the
cover term on each woody field, `rf_seed_variance.py` how much the Random Forest moves
when only its seed changes.

**Exports and number dumps.** `export_results_revision.py`, `export_appendix_extras.py`
and `export_species_table.py` write the CSVs in `exports/`. The four `results_dump_*.py`
print every number behind a figure as text.

## Running it

```bash
conda env create -f environment.yml
conda activate wolf_sdm
export PROJECT_ROOT=/path/to/the/folder/holding/code.nosync/and/data.nosync
```

Python 3.11.14, TensorFlow 2.12.0, scikit-learn 1.7.2, NumPy 1.23.5, pandas 2.3.3,
rasterio 1.4.4, GeoPandas 1.0.1.

Section 7 tunes the CNN and the Random Forest once and writes the result to JSON; both
files are in `hyperparameters/`, so the tuning run can be skipped. One fixed CNN
architecture and one fixed Random Forest configuration are then used for every
condition.

Order: `data_processing.ipynb`, then `synthetic_experiment.ipynb`, then the two
`coarse_auc_ceiling_*` scripts, then the figure scripts, then the exports.

Paths are resolved from `PROJECT_ROOT`, which defaults to the parent of this checkout.
The folder names `code.nosync` and `data.nosync` carry an iCloud exclusion suffix from
the machine the analysis ran on; rename them in the configuration cell if you prefer.

The input rasters are not in this repository; the notebook expects them under
`$PROJECT_ROOT/data.nosync/` with the file names given in its acquisition cells.

## Data

The source rasters are not redistributed here; the table above names each product and
the thesis gives versions and access dates. Administrative boundaries are GADM 4.1. The
preprocessing notebook also builds the layers that the collinearity screening then
removed, so the screening can be reproduced rather than taken on trust.

## Reproducibility

All seeds derive from one master seed. The CNN is fitted from three initialisations
everywhere and every replicate is recorded, because a single initialisation is not a
result; the Random Forest uses a fixed seed. Both models use 1:1 class weights, and one
fixed architecture and one fixed Random Forest configuration are used throughout — no
per-condition tuning. Scalers and padding values are computed on the training fold only.

Fitting the full set of experiments takes a long time on a CPU. Every long-running
section prints its total model-fit count before launching and has a smoke-test flag that
runs a reduced grid.

## Citation

Please cite the tagged release rather than the branch:

> Mende, J. (2026). *Code for: CNN vs Random Forest species distribution models*
> (version v1.0). GitHub.
> https://github.com/jonahmende/MasterThesis_GlobalChangeEcology/tree/v1.0

Released under the MIT License.
