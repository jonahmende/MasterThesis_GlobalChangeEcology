"""EDS on BAND PIXELS, every feature space, K in {1, 3, 5, 10, 20}.

The companion to eds_metric_properties.py, which does the same sweep on the
DRAWN SAMPLE. Together they answer whether the interpolative < extrapolative
ordering, and the "drivers on the random field sit at the interpolative level"
reading, survive the arbitrary choice K = 5.

Protocol copied from the two existing diagnostics so the numbers are
comparable: 1000 pixels per PC1 band, three draws with seeds 1000/1001/1002,
_s8_eds_decorr taken from the notebook rather than reimplemented.

EVERY AXIS IS MIN-MAX SCALED over the pool it is defined on, because the
distance is Euclidean and an unscaled axis simply wins. For the covariates that
is already true of the stack itself; for (P_w, E) and (P_w, E_perp) it is done
here.

DISCARDED VARIANT -- do not use numbers computed before this change. An earlier
run scaled P_w but let E_perp enter RAW. E_perp is a conditional z-score, so it
has SD 1 and spans about +/-5, while scaled P_w has SD 0.127 (random field) or
0.255 (forest map) on [0, 1]. The Euclidean kNN distance was therefore driven
by E_perp alone, roughly 8:1 on the random field and 4:1 on the forest map.
Measured effect on D(test,train) at K = 5: random field 1.0108 -> 1.0246,
forest map 1.2836 -> 1.3151. The ordering never changed, but the two driver
tables were not on one footing. The pipeline itself is untouched: it calls
_s8_eds_decorr only on the covariates.

TWO PIXEL POOLS, which is the whole reason this script exists:
  all_valid  every pixel finite in all six covariates (35,574,987), the pool
             the notebook's own setup diagnostic uses
  inner      the same, intersected with the interior-buffer mask (the 33 px
             window in which P_w and E are defined), 23,479,515 pixels. This
             is the pool eds_band_drivers.py uses, because the driver spaces
             do not exist outside it.
The covariate EDS is computed on BOTH, so the gap between the two published
covariate numbers is attributable rather than mysterious.

Environment: wolf_sdm
  (/opt/anaconda3/envs/wolf_sdm/bin/python eds_band_robustness.py)
Writes: figures/sus_scrofa/synthetic/eds_band_robustness.csv
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
SEEDS = [1000, 1001, 1002]
KS = [1, 3, 5, 10, 20]
R_M, CORR_LEN_M, FRAC_WOODY, GRF_SEED = 1600.0, 300.0, 0.30, 9000
PAIRS = [('te_tr', 'test', 'train'), ('vl_tr', 'val', 'train'),
         ('vl_te', 'val', 'test')]


def main():
    g = notebook_env(with_config=True)
    env = g['env_scaled_100m']
    eds = g['_s8_eds_decorr']
    vr, vc, pc1, (t40, t55, t70) = band_pixels(g)
    valid = np.all(np.isfinite(env), axis=2)
    pc = g['_pca3b'].transform(g['_sc3b'].transform(
        env[vr, vc, :].astype(np.float32)))[:, :2]
    print(f'env channels: {env.shape[2]}   valid pixels: {len(vr):,}')
    print(f'PC1 cuts: {t40:.6f} / {t55:.6f} / {t70:.6f}')

    sources = {
        'grf': g['_s8_make_W'](env.shape[:2], CORR_LEN_M / 100.0, FRAC_WOODY,
                               seed=GRF_SEED),
        'landuse': np.isin(g['landuse'], list(g['WOODY_CLASSES'])).astype(np.float32),
    }
    rows = []

    def add(space, mask_name, n_pool, seed, K, D):
        for pair, q, r in PAIRS:
            rows.append(dict(space=space, pixel_pool=mask_name,
                             n_pool_pixels=n_pool, seed=seed, K=K, pair=pair,
                             n_per_band=N_PER_BAND, value=eds(D[q], D[r], K)))

    for src, W in sources.items():
        Pw, E, win = g['_s8_W_fields'](W, valid, 100.0, R_M)
        inner = g['minimum_filter'](valid.view(np.uint8), size=win,
                                    mode='constant', cval=0).astype(bool)
        Ep = g['_s8_orthogonalize_E'](E, Pw, inner)
        ok = inner & np.isfinite(Ep)
        lo = np.array([Pw[inner].min(), E[inner].min()])
        hi = np.array([Pw[inner].max(), E[inner].max()])
        # E_perp gets the SAME min-max treatment as every other axis, over the
        # pixels on which it is defined. See the DISCARDED VARIANT note in the
        # header: it used to enter raw, in standard deviations, while P_w was
        # already in [0, 1] -- so the Euclidean kNN distance was driven almost
        # entirely by E_perp.
        loE, hiE = float(Ep[ok].min()), float(Ep[ok].max())
        for mask_name, m in (('inner', inner[vr, vc]),
                             ('inner_finiteEperp', ok[vr, vc])):
            idx = {'train': np.where((pc1 < t40) & m)[0],
                   'val': np.where((pc1 >= t40) & (pc1 < t55) & m)[0],
                   'test': np.where((pc1 >= t55) & (pc1 < t70) & m)[0]}
            npool = int(m.sum())
            for seed in SEEDS:
                rg = np.random.default_rng(seed)
                sel = {b: idx[b][rg.choice(len(idx[b]),
                                           min(N_PER_BAND, len(idx[b])),
                                           replace=False)] for b in idx}
                drv = {b: ((np.stack([Pw[vr[s], vc[s]], E[vr[s], vc[s]]], 1)
                            - lo) / (hi - lo)).astype(np.float32)
                       for b, s in sel.items()}
                dvp = {b: np.stack([(Pw[vr[s], vc[s]] - lo[0]) / (hi[0] - lo[0]),
                                    (Ep[vr[s], vc[s]] - loE) / (hiE - loE)], 1
                                   ).astype(np.float32)
                       for b, s in sel.items()}
                for K in KS:
                    if mask_name == 'inner':
                        add(f'drivers_PwE_{src}', mask_name, npool, seed, K, drv)
                    else:
                        add(f'drivers_PwEperp_{src}', mask_name, npool, seed,
                            K, dvp)

    # covariates and PC space, on BOTH pools
    _, _, win = g['_s8_W_fields'](sources['grf'], valid, 100.0, R_M)
    inner = g['minimum_filter'](valid.view(np.uint8), size=win,
                                mode='constant', cval=0).astype(bool)
    for mask_name, m in (('all_valid', np.ones(len(vr), bool)),
                         ('inner', inner[vr, vc])):
        idx = {'train': np.where((pc1 < t40) & m)[0],
               'val': np.where((pc1 >= t40) & (pc1 < t55) & m)[0],
               'test': np.where((pc1 >= t55) & (pc1 < t70) & m)[0]}
        npool = int(m.sum())
        print(f'[covariates] {mask_name}: pool={npool:,}  '
              + ' '.join(f'{b}={len(v):,}' for b, v in idx.items()))
        for seed in SEEDS:
            rg = np.random.default_rng(seed)
            sel = {b: idx[b][rg.choice(len(idx[b]),
                                       min(N_PER_BAND, len(idx[b])),
                                       replace=False)] for b in idx}
            cov = {b: env[vr[s], vc[s], :].astype(np.float32)
                   for b, s in sel.items()}
            pcs = {b: pc[s].astype(np.float32) for b, s in sel.items()}
            for K in KS:
                add('covariates', mask_name, npool, seed, K, cov)
                add('drivers_PC1PC2', mask_name, npool, seed, K, pcs)

    df = pd.DataFrame(rows)
    out = os.path.join(FIGURES_DIR, 'eds_band_robustness.csv')
    df.to_csv(out, index=False)
    pd.set_option('display.width', 250)
    print('\n-- K = 5, mean over the three draws --')
    print(df[df.K == 5].pivot_table(index=['space', 'pixel_pool'],
                                    columns='pair', values='value')
          .round(4).to_string())
    print('\n-- the driver spaces the thesis quotes: (P_w, E_perp), both axes '
          'min-max scaled --')
    print('   D(val,train) and D(test,train), mean over the three draws, '
          'UNROUNDED')
    dd = df[df.space.str.startswith('drivers_PwEperp')]
    for sp_ in sorted(dd.space.unique()):
        for pair in ('vl_tr', 'te_tr'):
            row = (dd[(dd.space == sp_) & (dd.pair == pair)]
                   .groupby('K')['value'].mean())
            cells = '  '.join(f'K={k}: {v!r}' for k, v in row.items())
            print(f'   {sp_:<24} {pair}  {cells}')
    print(f'\nSaved: {out}   ({len(df)} rows)')


if __name__ == '__main__':
    main()
