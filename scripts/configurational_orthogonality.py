"""How far the configurational fields sit from the environmental axes (v6).

Reports, for both W sources, on the interior-buffer mask of the 100 m grid:
  * realised woody fraction and the DERIVED mu_P (landscape mean of P_w)
  * redundancy of the driver against composition -- |Spearman| of the field
    against the best pointwise function of P_w (_s8_redundancy), for raw E and
    for the orthogonalised E_perp; the pipeline asserts <= _S8_MAX_REDUNDANCY
  * E_perp moments (should be mean 0, sd 1 by construction)
  * Spearman of E, P_w, E_perp and S_config against PC1 and PC2 -- the residual
    coupling of the configurational truth to the axes the extrapolative split
    is defined on
  * how much of the band area the interior buffer and the patch buffer remove,
    and whether the two masks coincide (they do at patch 32 / 100 m, which is
    why alpha = 0 and alpha > 0 sample the same area)

S_config uses the configurational niche of ONE species (default seed 42) since
mu_E/sig_E are drawn per species; every other quantity is species-independent.

Environment: wolf_sdm  (conda activate wolf_sdm && python configurational_orthogonality.py)
Writes: figures/sus_scrofa/synthetic/configurational_orthogonality.csv
        figures/sus_scrofa/synthetic/configurational_band_masks.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _vs_env import notebook_env, band_pixels, FIGURES_DIR

SPECIES_SEED = 42
R_M = 1600.0
CORR_LEN_M = 300.0
FRAC_WOODY = 0.30
GRF_SEED = 9000
SIG_P, W_P = 0.20, 0.1
N_SPEARMAN = 2_000_000          # subsample for the correlations
SEED = 0


def _sp(a, b, mask, n=N_SPEARMAN, seed=SEED):
    ii = np.flatnonzero(mask.ravel())
    ii = np.random.default_rng(seed).choice(ii, min(n, ii.size), replace=False)
    return float(spearmanr(a.ravel()[ii], b.ravel()[ii])[0])


def main():
    g = notebook_env(with_config=True)
    env = g['env_scaled_100m']
    vr, vc, pc1v, (t40, t55, t70) = band_pixels(g)
    valid = np.all(np.isfinite(env), axis=2)

    pc = np.full(env.shape[:2] + (2,), np.nan, np.float32)
    for s in range(0, len(vr), 4_000_000):
        e = min(s + 4_000_000, len(vr))
        pc[vr[s:e], vc[s:e]] = g['_pca3b'].transform(
            g['_sc3b'].transform(env[vr[s:e], vc[s:e], :].astype(np.float32)))
    pc1, pc2 = pc[..., 0], pc[..., 1]

    # the species' configurational niche (mu_E/sig_E are drawn per species)
    r = g['make_vs'](SPECIES_SEED, mode='pca', occ='probabilistic',
                     k=g['_S8_OCC_K'], prevalence=g['_S8_OCC_PREV'])
    niche = {**r[2], **g['_s8_config_niche'](SPECIES_SEED)}
    mu_E, sig_E = niche['mu_E'], niche['sig_E']
    print(f'species seed {SPECIES_SEED}: mu_E={mu_E:.4f} sig_E={sig_E:.4f}', flush=True)

    sources = {
        'grf': g['_s8_make_W'](env.shape[:2], CORR_LEN_M / 100.0, FRAC_WOODY, seed=GRF_SEED),
        'landuse': np.isin(g['landuse'], list(g['WOODY_CLASSES'])).astype(np.float32),
    }
    patch_ok = g['_s8_patch_ok'](env, g['_s8_patch_buffer_px'](100, g['PATCH']))
    tr_b, val_b, te_b = g['_s8_region_masks'](env, g['PATCH'])

    rows, mask_rows = [], []
    for src, W in sources.items():
        Pw, E, win = g['_s8_W_fields'](W, valid, 100.0, R_M)
        inner = g['minimum_filter'](valid.view(np.uint8), size=win,
                                    mode='constant', cval=0).astype(bool)
        mu_P = float(np.nanmean(Pw[inner]))
        Ep = g['_s8_orthogonalize_E'](E, Pw, inner)
        ok = inner & np.isfinite(Ep)
        Sc = g['_s8_suit_config_from_fields'](Pw, Ep, ok, mu_P, SIG_P, mu_E, sig_E, W_P)
        base = dict(w_source=src, window_px=int(win), r_m=R_M, mu_P_derived=mu_P,
                    frac_woody_realized=float(W[valid].mean()),
                    species_seed=SPECIES_SEED, mu_E=mu_E, sig_E=sig_E)
        rows.append({**base, 'quantity': 'redundancy_raw_E_given_Pw',
                     'value': float(g['_s8_redundancy'](E, Pw, inner))})
        rows.append({**base, 'quantity': 'redundancy_Eperp_given_Pw',
                     'value': float(g['_s8_redundancy'](Ep, Pw, ok))})
        rows.append({**base, 'quantity': 'Eperp_mean', 'value': float(np.nanmean(Ep[ok]))})
        rows.append({**base, 'quantity': 'Eperp_sd', 'value': float(np.nanstd(Ep[ok]))})
        rows.append({**base, 'quantity': 'Eperp_nan_inside_inner',
                     'value': float((inner & ~np.isfinite(Ep)).sum())})
        for lab, arr in (('E_raw', E), ('P_w', Pw), ('E_perp', Ep), ('S_config', Sc)):
            m = ok & np.isfinite(arr)
            rows.append({**base, 'quantity': f'spearman_{lab}_vs_PC1', 'value': _sp(arr, pc1, m)})
            rows.append({**base, 'quantity': f'spearman_{lab}_vs_PC2', 'value': _sp(arr, pc2, m)})
        S_ok = np.isfinite(Sc)
        for bname, bmask in (('train', tr_b), ('val', val_b), ('test', te_b)):
            mask_rows.append(dict(
                w_source=src, band=bname,
                raw=int(bmask.sum()),
                alpha0_patch_buffer=int((bmask & patch_ok).sum(),),
                alpha1_patch_and_finite_S=int((bmask & patch_ok & S_ok).sum())))
        mask_rows.append(dict(w_source=src, band='inner==patch_ok',
                              raw=int(np.array_equal(inner, patch_ok)),
                              alpha0_patch_buffer=-1, alpha1_patch_and_finite_S=-1))
        _red = {r['quantity']: r['value'] for r in rows if r['w_source'] == src}
        print(f"[{src}] mu_P={mu_P:.4f} "
              f"redundancy E|P_w={_red['redundancy_raw_E_given_Pw']:.3f} "
              f"E_perp|P_w={_red['redundancy_Eperp_given_Pw']:.4f}", flush=True)

    df = pd.DataFrame(rows); dm = pd.DataFrame(mask_rows)
    out = os.path.join(FIGURES_DIR, 'configurational_orthogonality.csv')
    out_m = os.path.join(FIGURES_DIR, 'configurational_band_masks.csv')
    df.to_csv(out, index=False); dm.to_csv(out_m, index=False)
    print('\n' + df.pivot_table(index='quantity', columns='w_source', values='value')
                    .round(4).to_string())
    print('\n' + dm.to_string(index=False))
    print(f'\nSaved: {out}\nSaved: {out_m}')


if __name__ == '__main__':
    main()
