"""Properties of the dissimilarity metric actually used in v6 (_s8_eds_decorr).

Three checks on the same cells, all on the v6 stack (6 covariates, per-layer
global min-max, Landuse and Elev excluded):

  1 ROTATION      Is the PCA step inside _s8_eds_decorr doing anything? It is a
                  full-rank rotation without whitening, so Euclidean distances
                  should be invariant. Compared against the identical
                  computation with the PCA line removed.
  2 K SENSITIVITY The neighbour count K is fixed at 5 in the pipeline and was
                  never swept (the k in the ksweep_* scripts is the logistic
                  STEEPNESS of the occurrence conversion, not this K).
                  Computed for K in {1, 3, 5, 10, 20}.
  3 WHITENING     What a genuine correlation correction would change:
                  PCA(whiten=True) on the reference set, everything else equal.

Cells: two extrapolative draws at 2000/300, one at 5000/300, two interpolative
draws at 2000/300; reference = train points, query = test points, drawn from
the PC1 bands with fixed seeds (100..104).

Environment: wolf_sdm  (conda activate wolf_sdm && python eds_metric_properties.py)
Writes: figures/sus_scrofa/synthetic/eds_metric_properties.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors as NNB
from sklearn.preprocessing import StandardScaler

from _vs_env import notebook_env, band_pixels, FIGURES_DIR

K_LIST = [1, 3, 5, 10, 20]
K_DEFAULT = 5
CELLS = [('extrap', 2000, 300), ('extrap', 2000, 300),
         ('extrap', 5000, 300), ('random', 2000, 300), ('random', 2000, 300)]


def _ratio(Q, R, K=K_DEFAULT, rotate=True, whiten=False):
    """_s8_eds_decorr with the PCA step switchable, everything else identical."""
    if len(R) < K + 2 or len(Q) < 1:
        return np.nan
    if rotate:
        p = PCA(n_components=R.shape[1], whiten=whiten).fit(R)
        rp, qp = p.transform(R), p.transform(Q)
    else:
        rp, qp = R, Q
    wm = NNB(n_neighbors=K + 1).fit(rp).kneighbors(rp)[0][:, 1:].mean()
    if wm < 1e-10:
        return np.nan
    return float(NNB(n_neighbors=min(K, len(rp))).fit(rp).kneighbors(qp)[0].mean() / wm)


def _eds_norm(Q, R, K=K_DEFAULT):
    """plain EDS of the ksweep_*/benchmark scripts: StandardScaler FIT ON THE
    REFERENCE SET instead of the pipeline's global min-max, no rotation."""
    if len(R) < K + 2 or len(Q) < 1:
        return np.nan
    sc = StandardScaler().fit(R)
    rp, qp = sc.transform(R), sc.transform(Q)
    wm = NNB(n_neighbors=K + 1).fit(rp).kneighbors(rp)[0][:, 1:].mean()
    if wm < 1e-10:
        return np.nan
    return float(NNB(n_neighbors=min(K, len(rp))).fit(rp).kneighbors(qp)[0].mean() / wm)


def main():
    g = notebook_env(with_config=False)
    env = g['env_scaled_100m']
    vr, vc, pc1, (t40, t55, t70) = band_pixels(g)
    pools = {'train': np.where(pc1 < t40)[0],
             'test': np.where((pc1 >= t55) & (pc1 < t70))[0],
             'all': np.arange(len(vr))}
    eds_nb = g['_s8_eds_decorr']            # the pipeline's own function
    rows = []
    for i, (arm, n_ref, n_qry) in enumerate(CELLS):
        r = np.random.default_rng(100 + i)
        p_ref = pools['train'] if arm == 'extrap' else pools['all']
        p_qry = pools['test'] if arm == 'extrap' else pools['all']
        ri = r.choice(p_ref, n_ref, replace=False)
        qi = r.choice(p_qry, n_qry, replace=False)
        R = env[vr[ri], vc[ri], :].astype(np.float32)
        Q = env[vr[qi], vc[qi], :].astype(np.float32)
        cm = np.corrcoef(R.astype(np.float64).T); np.fill_diagonal(cm, 0.0)
        base = dict(cell=i, arm=arm, n_ref=n_ref, n_query=n_qry,
                    max_abs_r_reference=float(np.abs(cm).max()))
        v_pipe = eds_nb(Q, R, K_DEFAULT)
        v_norot = _ratio(Q, R, K_DEFAULT, rotate=False)
        rows.append({**base, 'check': 'rotation', 'variant': 'pipeline (with PCA)',
                     'K': K_DEFAULT, 'value': v_pipe})
        rows.append({**base, 'check': 'rotation', 'variant': 'PCA line removed',
                     'K': K_DEFAULT, 'value': v_norot})
        rows.append({**base, 'check': 'rotation', 'variant': 'abs difference',
                     'K': K_DEFAULT, 'value': abs(v_pipe - v_norot)})
        for K in K_LIST:
            rows.append({**base, 'check': 'K', 'variant': f'K={K}', 'K': K,
                         'value': eds_nb(Q, R, K)})
        rows.append({**base, 'check': 'whitening', 'variant': 'current (rotation only)',
                     'K': K_DEFAULT, 'value': v_pipe})
        rows.append({**base, 'check': 'whitening', 'variant': 'PCA(whiten=True)',
                     'K': K_DEFAULT, 'value': _ratio(Q, R, K_DEFAULT, whiten=True)})
        rows.append({**base, 'check': 'scaling', 'variant': 'eds_norm (z on reference)',
                     'K': K_DEFAULT, 'value': _eds_norm(Q, R)})
        print(f'  cell {i} ({arm}) done', flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(FIGURES_DIR, 'eds_metric_properties.csv')
    df.to_csv(out, index=False)

    p = df.pivot_table(index=['cell', 'arm'], columns='variant', values='value')
    print('\n-- rotation: the PCA step leaves the value unchanged --')
    print(p[['pipeline (with PCA)', 'PCA line removed', 'abs difference']].to_string(
        float_format=lambda v: f'{v:.3e}' if v < 1e-3 else f'{v:.6f}'))
    print('\n-- K sensitivity --')
    print(p[[f'K={K}' for K in K_LIST]].round(3).to_string())
    print('\n-- whitening and scaling --')
    print(p[['current (rotation only)', 'PCA(whiten=True)',
             'eds_norm (z on reference)']].round(3).to_string())
    print(f'\nmax |pipeline - no-PCA| over cells: '
          f'{df[df.variant == "abs difference"].value.max():.3e}')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
