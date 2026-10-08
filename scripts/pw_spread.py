"""How much does the cover term P_w actually vary, on each W source?

The claim under test: the forest map's composition-only models (RF-patch,
RF-center) reach rho = 0.14-0.34 at alpha = 1 while the random field's sit at
0.01-0.03 partly because the forest map's COVER TERM carries more variation to
rank on. Spearman(P_w, PC1) = 0.553 vs 0.007 already shows the coupling to the
environmental axes; this adds the spread itself, which the coupling argument
assumes but does not measure.

Three quantities, both W sources, on the same interior mask the pipeline uses:

  P_w            the r_m-scale cover term (uniform filter, r_m = 1600 m = 33 px)
                 -- the quantity that enters S_config through P_w_term
  patch-mean W   the 32 x 32 patch mean of the binary W channel -- literally one
                 of RF-patch's two extra features
  patch-SD W     the other one

Nothing is refitted and nothing is written but the CSV: this is a property of
the two maps, not of any model run.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python pw_spread.py)
Writes: figures/sus_scrofa/synthetic/pw_spread.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np
import pandas as pd

from _vs_env import notebook_env, FIGURES_DIR

R_M = 1600.0
CORR_LEN_M = 300.0
FRAC_WOODY = 0.30
GRF_SEED = 9000


def describe(a, mask, name, src, extra):
    v = a[mask]
    v = v[np.isfinite(v)]
    q = np.percentile(v, [5, 10, 25, 50, 75, 90, 95])
    return dict(w_source=src, quantity=name, n=int(v.size),
                mean=float(v.mean()), sd=float(v.std()),
                p5=q[0], p10=q[1], p25=q[2], p50=q[3], p75=q[4], p90=q[5],
                p95=q[6], iqr=float(q[4] - q[2]),
                cv=float(v.std() / v.mean()) if v.mean() else np.nan, **extra)


def main():
    g = notebook_env(with_config=True)
    env = g['env_scaled_100m']
    valid = np.all(np.isfinite(env), axis=2)
    patch = g['PATCH']

    sources = {
        'grf': g['_s8_make_W'](env.shape[:2], CORR_LEN_M / 100.0, FRAC_WOODY,
                               seed=GRF_SEED),
        'landuse': np.isin(g['landuse'], list(g['WOODY_CLASSES'])).astype(np.float32),
    }

    rows = []
    for src, W in sources.items():
        Pw, E, win = g['_s8_W_fields'](W, valid, 100.0, R_M)
        inner = g['minimum_filter'](valid.view(np.uint8), size=win,
                                    mode='constant', cval=0).astype(bool)
        extra = dict(window_px=int(win),
                     frac_woody_realized=float(W[valid].mean()))
        rows.append(describe(Pw, inner, 'P_w', src, extra))

        # RF-patch's own two features: the mean and SD of the binary W channel
        # over a patch-sized window. uniform_filter gives the mean; the SD comes
        # from E[W^2] - E[W]^2, which is exact for a binary field.
        m1 = g['uniform_filter'](W.astype(np.float32), size=patch,
                                 mode='nearest')
        m2 = g['uniform_filter'](W.astype(np.float32) ** 2, size=patch,
                                 mode='nearest')
        sdW = np.sqrt(np.clip(m2 - m1 ** 2, 0.0, None))
        pmask = g['_s8_patch_ok'](env, g['_s8_patch_buffer_px'](100, patch))
        rows.append(describe(m1, pmask, f'patch{patch}_mean_W', src, extra))
        rows.append(describe(sdW, pmask, f'patch{patch}_sd_W', src, extra))

    df = pd.DataFrame(rows)
    out = os.path.join(FIGURES_DIR, 'pw_spread.csv')
    df.to_csv(out, index=False)
    cols = ['quantity', 'w_source', 'mean', 'sd', 'iqr', 'cv',
            'p10', 'p50', 'p90', 'n']
    print('\n' + df[cols].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n-- ratio landuse / grf --')
    for q in df.quantity.unique():
        a = df[(df.quantity == q) & (df.w_source == 'landuse')].iloc[0]
        b = df[(df.quantity == q) & (df.w_source == 'grf')].iloc[0]
        print(f'  {q:<16} SD {a.sd / b.sd:5.2f}x   IQR {a.iqr / b.iqr:5.2f}x   '
              f'CV {a.cv / b.cv:5.2f}x')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
