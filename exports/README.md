# exports/results_revision

Raw, **unrounded** tables behind the revised Results section. Written by
`code.nosync/scripts_synthetic/export_results_revision.py` (CSV tables) and
`export_species_table.py` (`species.csv`). No analysis was changed; every
definition is imported from the figure modules so the tables and the figures
cannot drift apart.

## Shared definitions

- **rho** — Spearman correlation of the model's predicted probability with the
  *known true suitability* at the test points. The thesis' primary metric.
- **auc** — ROC AUC of the same predictions against the *drawn* presence/absence
  labels. Secondary.
- **arm** — `random` = interpolative (one point pool, stratified random split);
  `extrap` = extrapolative (train/val/test from three disjoint PC1 bands).
- **auc_chance_floor** — `0.5 + 2 SE`, `SE = sqrt((n+1)/(12 n1 n0))`, the
  Mann-Whitney null SE on the set in question. Not Hanley-McNeil.
- **no_skill** — the AUC is below that floor, i.e. not two standard errors
  above chance. An outcome statement; it applies to every model.
- **learned / unlearned** — a single CNN fit counts as *unlearned* when its best
  validation AUC is below the floor computed on the validation set. A condition
  is flagged `unlearned` when more than half of its CNN fits are.
- **pooled_init_sd** — the noise band of fig4/fig7:
  `sqrt(mean over species of Var across that species' three CNN inits)`.
- **contrast** — formed **per species** and then averaged, never as a difference
  of two means: the shared data draw cancels only in the paired form. Baseline
  is RF for the pointwise process and RF-patch for the configurational one.
- **rho_ceiling** — rank ceiling. Pointwise: `oracle_spearman_infoloss`, the rank
  information left after coarsening (resolution axis only; empty elsewhere,
  where the ceiling is simply 1). Configurational: `ceiling_config_patch`,
  `E[S_config | patch-mean cover, patch-mean edge density]`, which bounds
  patch-limited models only — not RF-oracle.
- **auc_ceiling** — label-noise ceiling `auc_oracle` (AUC of the true suitability
  against the drawn labels), replaced on **both resolution axes** by the
  coarse-cell ceiling: AUC of the block-mean truth over each point's coarse cell
  against the drawn labels.
- **species_no** — the pipeline's `species_idx`, the position in the admitted
  species list, **starting at 0**. Sensitivity species 0/1/2 are blend species
  0/1/2.
- Model names: `RF` (pointwise centre-pixel RF), `RF-centre`, `RF-patch`
  (+ patch mean and SD of the W channel), `RF-oracle` (+ the raw drivers at the
  truth's own radius; *not* patch-limited), `CNN-Ensemble` (mean of 3 inits).

Filters match the figures: headline `res_mode` only (`constant_footprint` /
`baseline_100m`) on both resolution axes.

## Files

### `sens_species.csv` — one row per process x axis x level x arm x species x model
`process`, `experiment`, `level`, `arm`, `species_no`, `seed`, `model`, `rho`,
`auc`, `rho_ceiling`, `auc_ceiling`, `no_skill`, `auc_chance_floor`, `n_train`,
`n_val`, `n_test`.

### `sens_cnn_inits.csv` — one row per CNN initialisation
Same keys plus `init`, `rho`, `auc`, `best_val_auc`, `learned`.

### `sens_conditions.csv` — one row per process x axis x level x arm
Per model `rho_<model>`, `auc_<model>`, `no_skill_<model>` (computed on the
species mean, as fig6 draws it); `rho_ceiling`, `auc_ceiling`; `contrast`,
`contrast_baseline`, `pooled_init_sd`, `abs_contrast_over_sd`, `within_sd`
(|contrast| <= SD, i.e. inconclusive); `cnn_fits_unlearned` / `cnn_fits_total`
and the derived `unlearned`; `n_species`, `auc_chance_floor`.

### `blend_species.csv` — alpha x W source x arm x species x model
`rho`, `auc`, `rho_ceiling_patch` (NaN at alpha = 0: those runs never compute
it), `auc_ceiling_labelnoise`, `cnn_inits_unlearned` of `cnn_inits_total` (3).

### `blend_conditions.csv` — alpha x W source x arm x model
Mean **and** median over the eight species of `rho`, `auc` and both ceilings;
`share_of_ceiling_mean` / `_median` (formed per species as rho / ceiling, then
aggregated); for the CNN row also `contrast_mean`, `_median`, `_min`, `_max` and
`n_species_cnn_above_rfpatch`; `cnn_fits_unlearned` of `cnn_fits_total` (24).

### `blend_baseline.csv` — alpha = 0 with the W channel vs the no-W baseline
Per arm and model: `rho_no_W_*`, `rho_with_W_*`, `difference_*`. The two runs
share the truth; only the uninformative W channel is added.

### `blend_corrlen.csv` — correlation length 150/200/400 m plus 300 m
Per arm: CNN and RF-patch rho, and the contrast (mean, median, min, max), plus
unlearned fits. `from_alpha_sweep` marks 300 m, which is the alpha sweep's own
default condition spliced in rather than refitted.

### `blend_oracleval.csv` — primary vs oracle validation
Per alpha and arm: CNN rho under both validation designs (mean, median, min,
max) and the change, with its range over species. An upper bound: oracle
validation assumes labels in the target domain.

### `species.csv` — every species the thesis reports on
`species_no`, `seed`, `experiments`, `in_sensitivity`, `in_blend`, `mu1`, `mu2`,
`sigma1`, `sigma2`, `power`, `sigma1_effective` (= sigma1 / sqrt(power), the
width the response actually has and the one the band masses use),
`mu1_pc1_percentile` (percentile of mu1 in the PC1 distribution of all
35,574,987 valid pixels), `mu1_band`, `mean_suitability_{train,val,test,
beyond_test}` (population means of the pointwise suitability over all valid
pixels of each band), `mu_E`, `sigma_E`.

---

## Appendix tables

Written by `export_appendix_extras.py`; the EDS band-pixel sweep comes from
`eds_band_robustness.py`. Same rule as above: unrounded, nothing refitted.

### `eds_robustness.csv` — the dissimilarity metric under K and under whitening
One long table with two sources in the `source` column.

- `source = "drawn sample"` — from `eds_metric_properties.csv`: five cells
  (three extrapolative, two interpolative; reference = that cell's training
  points, query = its test points, seeds 100–104), with `check` in
  {`rotation`, `K`, `whitening`, `scaling`} and `variant` naming the computation.
  The `rotation` rows show that the PCA step inside `_s8_eds_decorr` is a
  full-rank rotation and changes nothing (difference <= 3.5e-8); the
  `whitening` rows show what `PCA(whiten=True)` would change instead.
- `source = "band pixels"` — from `eds_band_robustness.csv`: 1000 pixels per
  PC1 band, three draws (seeds 1000/1001/1002), K in {1, 3, 5, 10, 20}, for
  every feature space (`covariates`, `drivers_PC1PC2`, `drivers_PwE_*`,
  `drivers_PwEperp_*`).

ALL AXES ARE MIN-MAX SCALED over the pool they are defined on, because the
distance is Euclidean and an unscaled axis dominates it. The covariate stack is
already scaled; `(P_w, E)` and `(P_w, E_perp)` are scaled here.

**Discarded variant — do not use `(P_w, E_perp)` numbers computed before
2026-10-07.** An earlier run scaled P_w but let E_perp enter raw. E_perp is a
conditional z-score (SD 1, span about +/-5) while scaled P_w has SD 0.127
(random field) or 0.255 (forest map) on [0, 1], so the Euclidean kNN distance
was driven by E_perp alone, roughly 8:1 and 4:1. Effect on D(test,train) at
K = 5: random field 1.0108 -> 1.0246, forest map 1.2836 -> 1.3151. The ordering
never changed, and the pipeline is untouched (it calls `_s8_eds_decorr` only on
the covariates), but the thesis quotes the `(P_w, E_perp)` values, so the
corrected ones are the ones to use.

`pixel_pool` is the column that explains the two published covariate numbers:
`all_valid` = all 35,574,987 pixels finite in the six covariates, the pool the
notebook's own setup diagnostic uses; `inner` = the same intersected with the
interior-buffer mask (the 33 px window in which P_w and E are defined),
33,624,026 pixels, the pool `eds_band_drivers.py` uses because the driver
spaces do not exist outside it. Same seeds, same n, same K — only the pool
differs, which is why covariates at K = 5 give 1.8446 on `all_valid` and
1.8263 on `inner`. `n_pool_pixels` records the pool size per row.

### `rf_seed_variance_sd.csv` — how far the RF moves on its seed alone
The pipeline fits one RF per cell (`random_state = 42`) and gives the CNN three
initialisations. This refits the RF with `random_state` in {0, 1, 2, 3, 42} on
the default cells (pointwise, N = 2000, both arms, the three sensitivity
species) and reports mean, SD (ddof = 1), min and max of rho and AUC over those
five seeds. `thesis_species` = `species_no + 1`, the 1-based numbering the text
uses. The `random_state = 42` fit reproduces the stored CSV row exactly, which
is what certifies that the other four differ in the seed alone.

### `blend_fit_health.csv` — one row per alpha x W source x arm
`pooled_init_sd_rho` is the fig4/fig7 noise band of the rank correlation for
that condition; `init_sd_rho_min` / `_max` bracket the per-species init SDs it
pools. `cnn_fits_unlearned` of `cnn_fits_total` (24 = 8 species x 3 inits) and
`unlearned_condition` apply the > 50 % rule. `val_auc_floor` is the threshold a
single fit is judged against: `0.5 + 2 SE` on the validation set, which is
**0.56677769** throughout (n_val = 300 at prevalence 0.5, i.e. 150/150).

### `covariates.csv` — the six channels of `env_scaled_100m` (v6_bio11)
One row per channel in stack order: `product`, `variable` (the definition, not
just the name), `native_resolution`, `period`, `resampling`, `transform`
(log1p where applied), plus the shared `grid` (8655 x 6410 px at 100 m), `crs`
(EPSG:32632, UTM zone 32N), `dtype` and `final_scaling` (landscape-wide
per-layer min-max to [0, 1]).

NOTE on `PA_Distance`: the preprocessing notebook's prose says the WDPA/WDOECM
download is the May 2026 release, while the code path reads
`WDPA_WDOECM_Jun2026_Public_DEU_shp`. Both are recorded in `period`; which one
actually shipped has not been resolved here.

### `configurational_orthogonality.csv` — how far the driver fields sit from the environment
One row per `w_source` x `quantity`. Spearman rho of the raw edge density E,
the cover term `P_w`, the orthogonalised `E_perp` and the resulting `S_config`
against PC1 and PC2, plus the redundancy of E and of `E_perp` given `P_w`, the
moments of `E_perp`, and the count of pixels inside the interior buffer where
`E_perp` is undefined. `mu_P_derived` and `frac_woody_realized` record what the
woody field actually came out at. Produced by `configurational_orthogonality.py`.

### `eds_band_driver_moments.csv` — the driver fields per PC1 band
One row per `w_source` x draw (`seed` in {1000, 1001, 1002}) x `band`, on 1000
random pixels per band. `P_w_mean` / `P_w_sd` and `E_mean` / `E_sd` are the
moments of the cover term and the edge density in that band. These are the
numbers behind the statement that cover is flat across the bands on the random
field (0.3016 / 0.2976 / 0.2971 for train / val / test) but rises on the forest
map (0.2722 / 0.4064 / 0.4489). Produced by `eds_band_drivers.py`; the EDS
values of the same run are the `drivers_*` rows of `eds_robustness.csv`.

### `eds_metric_properties.csv` — EDS on the DRAWN SAMPLE, not on band pixels
The companion to `eds_robustness.csv`: the same question asked of the points an
experiment actually draws. One row per `cell` x `arm` x `check` x `variant`.
`variant` is either `K=1 … K=20` (the K sweep), `PCA(whiten=True)` (whitening
on, which rescales each component to unit variance and therefore changes the
distance), `PCA line removed` (the rotation skipped) or `abs difference`
(rotation vs no rotation, which is 0 to within float precision because a
rotation without whitening leaves Euclidean distances unchanged). `n_ref` and
`n_query` give the training and test sample sizes of that cell.

### `pw_spread.csv` — the spread of the cover term on each woody field
One row per `w_source` x `quantity`, over the 33,624,026 interior-buffer pixels.
`P_w` is the cover term itself, `patch32_mean_W` and `patch32_sd_W` the patch
summaries RF-patch is handed. Mean, SD, the percentile ladder, IQR and CV. This
is where the difference in woody density between the two fields is read off:
mean `P_w` is 0.3000 on the random field and 0.4107 on the forest map, with a
much wider spread on the latter (SD 0.2547 against 0.1086).
