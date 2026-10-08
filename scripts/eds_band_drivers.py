"""Do the configurational drivers shift between the PC1 bands? (results table)

At alpha = 1 the truth depends only on the windowed drivers P_w (local woody
cover) and E (local edge density), while the extrapolative partition is defined
on PC1 of the COVARIATES. If the drivers are distributed identically across the
bands, the extrapolative arm moves the models away from their training data in
a direction the truth does not care about -- which is what the GRF arm is for.
This script measures that, for both W sources.

Design: the same shape as the setup diagnostic that produces _S8_EDS_* in the
notebook -- 3 independent draws of 1000 pixels per band, seeds 1000/1001/1002,
_s8_eds_decorr with K = 5 -- but on the drivers instead of the covariates, and
species-independent: P_w and E depend only on W and r_m, never on a niche.

  drivers     (P_w, E), each scaled by its landscape-wide min-max over the
              interior-buffer mask (the pixels where they are defined)
  covariates  the 6 model inputs, on the identical pixel draws, for reference
  bands       PC1 < 40th pct / [40,55) / [55,70), intersected with the
              interior buffer (33 px window at r_m = 1600 m, 100 m grid)

Environment: wolf_sdm  (conda activate wolf_sdm && python eds_band_drivers.py)
Writes: figures/sus_scrofa/synthetic/eds_band_drivers.csv
        figures/sus_scrofa/synthetic/eds_band_driver_moments.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np
import pandas as pd

from _vs_env import notebook_env, band_pixels, FIGURES_DIR

N_PER_BAND = 1000
SEEDS = [1000, 1001, 1002]          # same seeds as the notebook's _S8_EDS_* block
K = 5
R_M = 1600.0
CORR_LEN_M = 300.0
FRAC_WOODY = 0.30
GRF_SEED = 9000


def main():
    g = notebook_env(with_config=True)
    np_ = np
    env = g['env_scaled_100m']
    vr, vc, pc1, (t40, t55, t70) = band_pixels(g)
    valid = np_.all(np_.isfinite(env), axis=2)
    sources = {
        'grf': g['_s8_make_W'](env.shape[:2], CORR_LEN_M / 100.0, FRAC_WOODY, seed=GRF_SEED),
        'landuse': np_.isin(g['landuse'], list(g['WOODY_CLASSES'])).astype(np_.float32),
    }
    eds = g['_s8_eds_decorr']
    rows, moments = [], []
    print(f'PC1 thresholds: {t40:.3f} / {t55:.3f} / {t70:.3f}', flush=True)

    for src, W in sources.items():
        Pw, E, win = g['_s8_W_fields'](W, valid, 100.0, R_M)
        inner = g['minimum_filter'](valid.view(np_.uint8), size=win,
                                    mode='constant', cval=0).astype(bool)
        lo = np_.array([Pw[inner].min(), E[inner].min()])
        hi = np_.array([Pw[inner].max(), E[inner].max()])
        ins = inner[vr, vc]
        idx = {'train': np_.where((pc1 < t40) & ins)[0],
               'val': np_.where((pc1 >= t40) & (pc1 < t55) & ins)[0],
               'test': np_.where((pc1 >= t55) & (pc1 < t70) & ins)[0]}
        print(f'\n[{src}] window={win}px  band px (inner): '
              + ' '.join(f'{b}={len(v):,}' for b, v in idx.items()), flush=True)

        per_trial = {k: [] for k in ('drivers', 'covariates')}
        for seed in SEEDS:
            r = np_.random.default_rng(seed)
            sel = {b: idx[b][r.choice(len(idx[b]), min(N_PER_BAND, len(idx[b])), replace=False)]
                   for b in idx}
            drv = {b: ((np_.stack([Pw[vr[s], vc[s]], E[vr[s], vc[s]]], 1) - lo) / (hi - lo)
                       ).astype(np_.float32) for b, s in sel.items()}
            cov = {b: env[vr[s], vc[s], :].astype(np_.float32) for b, s in sel.items()}
            for space, D in (('drivers', drv), ('covariates', cov)):
                per_trial[space].append(dict(
                    te_tr=eds(D['test'], D['train'], K),
                    vl_tr=eds(D['val'], D['train'], K),
                    vl_te=eds(D['val'], D['test'], K)))
            for b, s in sel.items():
                moments.append(dict(w_source=src, seed=seed, band=b, n=len(s),
                                    P_w_mean=float(Pw[vr[s], vc[s]].mean()),
                                    P_w_sd=float(Pw[vr[s], vc[s]].std()),
                                    E_mean=float(E[vr[s], vc[s]].mean()),
                                    E_sd=float(E[vr[s], vc[s]].std())))
        for space, trials in per_trial.items():
            for pair in ('te_tr', 'vl_tr', 'vl_te'):
                v = np_.array([t[pair] for t in trials], float)
                rows.append(dict(w_source=src, feature_space=space, pair=pair,
                                 mean=float(v.mean()), sd=float(v.std()),
                                 n_draws=len(v), n_per_band=N_PER_BAND, K=K,
                                 window_px=int(win), r_m=R_M))
                print(f'  {space:<11}{pair:<7} {v.mean():.3f} +/- {v.std():.3f}', flush=True)

    df = pd.DataFrame(rows)
    dm = pd.DataFrame(moments)
    out = os.path.join(FIGURES_DIR, 'eds_band_drivers.csv')
    out_m = os.path.join(FIGURES_DIR, 'eds_band_driver_moments.csv')
    df.to_csv(out, index=False)
    dm.to_csv(out_m, index=False)
    print('\n-- band moments, mean over the 3 draws --')
    print(dm.groupby(['w_source', 'band'])[['P_w_mean', 'P_w_sd', 'E_mean', 'E_sd']]
            .mean().round(4).to_string())
    print(f'\nSaved: {out}\nSaved: {out_m}')


if __name__ == '__main__':
    main()
