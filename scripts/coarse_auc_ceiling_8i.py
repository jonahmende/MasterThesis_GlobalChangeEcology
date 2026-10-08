"""The coarse-information AUC ceiling for the 8i resolution axis (alpha = 1).

The alpha = 0 counterpart is coarse_auc_ceiling_8c.py; this is the same
construction on the configurational truth, for the same reason. The AUC panel
of the resolution figure carried auc_oracle, which uses the 100 m truth AT THE
POINT and is therefore flat in resolution by construction (0.918 random / 0.925
extrap at every level). What is wanted instead is

    AUC( drawn labels , block mean of the 100 m truth over the point's
                        coarse cell )

the best AUC any model could reach knowing its coarse cell perfectly, which
does fall as the grain coarsens.

The truth here is the configurational surface: W is the GRF (seed 9000,
corr_len 300 m, 30 % woody), the fields are built by _s8_config_truth_100m at
r_m = 1600 m with the species' own mu_E / sig_E, and alpha = 1 makes
_s8_suit_blend return S_config unchanged. Points come from the run's own seeds
(8i resolution, constant_footprint, seed base 38000 + species_idx*100000).

TWO FATAL FIDELITY CHECKS, as in the 8c script: at 100 m the coarse ceiling
must equal the stored auc_oracle exactly, and the regenerated points must
reproduce points_fingerprint.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python coarse_auc_ceiling_8i.py)
Writes: figures/thesis/coarse_auc_ceiling_8i.csv
"""
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _vs_env import notebook_env, exec_def, FIGURES_DIR
from coarse_auc_ceiling_8c import draw_points

VERSION = 'v6_bio11'
SPECIES = [(0, 42), (1, 55838), (2, 63757)]
SEED_BASE = 38000                    # _S8I_SEED_RES, constant_footprint offset 0
SUB_EXP = '8i_resolution_grf_alpha1'
OUT = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', 'thesis',
                                    'coarse_auc_ceiling_8i.csv'))


def main():
    g = notebook_env(with_config=True)
    # _s8_config_truth_100m lives in the 8g cell, whose remainder is the sweep
    # itself, so only that one definition is pulled in -- from the notebook.
    exec_def(g, '_s8_config_truth_100m', marker='def _s8_run_level_config(')
    env = g['env_scaled_100m']
    cfgc = g['_S8_CFG']
    ladder = [(r, g['_s8_constfoot_patch'](r)) for r in g['_S8_RES_CONSTFOOT']]
    print(f'\nladder (constant footprint): {ladder}', flush=True)

    ref = pd.read_csv(os.path.join(FIGURES_DIR,
                                   f'section8i_alpha_axes_{VERSION}.csv'),
                      float_precision='round_trip')
    ref = ref[(ref.sub_exp == SUB_EXP) & (ref.res_mode == 'constant_footprint')
              & (ref.model == 'rf')]

    W = g['_s8_make_W'](env.shape[:2], cfgc['corr_len_m'] / 100.0,
                        cfgc['frac_woody'], seed=cfgc['grf_seed'])
    mask = g['_s8_fixed_common_mask'](env, ladder)
    tr_b, val_b, te_b = g['_s8_region_masks'](env, 0)
    rows, fails = [], []

    for sp_i, sp_sd in SPECIES:
        r0 = g['make_vs'](sp_sd, mode='pca', occ='probabilistic',
                          k=g['_S8_OCC_K'], prevalence=g['_S8_OCC_PREV'])
        g['_s8_meta'] = {**r0[2], **g['_s8_config_niche'](sp_sd)}
        g['_s8_suit_100m'] = g['_s8_suit_fn'](env)
        cfg_t = g['_s8_config_truth_100m'](
            env, W, r_m=cfgc['r_m'], sig_P=cfgc['sig_P'],
            mu_E=g['_s8_meta']['mu_E'], sig_E=g['_s8_meta']['sig_E'],
            w_P=cfgc['w_P'])
        # alpha = 1: _s8_suit_blend returns S_config unchanged
        truth = g['_s8_suit_blend'](g['_s8_suit_100m'], cfg_t['S_config_100'], 1.0)
        ok_truth = np.isfinite(truth) & np.all(np.isfinite(env), axis=2)
        elig = mask & ok_truth
        bands = dict(ok_truth=ok_truth, tr=tr_b & elig, val=val_b & elig,
                     te=te_b & elig)
        seed = SEED_BASE + sp_i * 100000

        for res, patch in ladder:
            f = int(round(res / 100))
            blk = g['_s8_blockmean_to_coarse_grid'](truth, res)
            for arm in ('random', 'extrap'):
                (r_tr, c_tr, y_tr), (r_v, c_v, y_v), (r_te, c_te, y_te) = \
                    draw_points(g, truth, elig, bands, res, patch, seed, arm)
                fp = hashlib.sha1(np.concatenate(
                    [r_tr, c_tr, y_tr, r_v, c_v, y_v, r_te, c_te, y_te]
                ).astype(np.int64).tobytes()).hexdigest()[:12]

                s_te = truth[r_te, c_te]
                bl = blk[r_te // f, c_te // f]
                ok = np.isfinite(bl) & np.isfinite(s_te)
                auc_point = float(roc_auc_score(y_te[ok], s_te[ok]))
                auc_coarse = float(roc_auc_score(y_te[ok], bl[ok]))

                row = ref[(ref.species_idx == sp_i) & (ref.level == res)
                          & (ref.arm == arm)]
                fp_ref = row.points_fingerprint.iloc[0] if not row.empty else None
                auc_ref = float(row.auc_oracle.iloc[0]) if not row.empty else np.nan
                if fp != fp_ref:
                    fails.append(f'fingerprint: species {sp_i} {arm} {res} m -- '
                                 f'{fp} vs {fp_ref}')
                if res == 100 and auc_coarse != auc_ref:
                    fails.append(f'100 m ceiling != stored auc_oracle: species '
                                 f'{sp_i} {arm} -- {auc_coarse!r} vs {auc_ref!r}')
                rows.append(dict(species_idx=sp_i, species_seed=sp_sd, arm=arm,
                                 level=res, patch=patch, n_te=int(len(y_te)),
                                 auc_oracle_point=auc_point,
                                 auc_oracle_stored=auc_ref,
                                 auc_ceiling_coarse=auc_coarse,
                                 fingerprint=fp, fingerprint_stored=fp_ref))
                print(f'  sp{sp_i} {arm:<7} {res:>4} m  coarse={auc_coarse:.6f}  '
                      f'point={auc_point:.6f}  stored={auc_ref:.6f}  '
                      f'fp={fp}/{fp_ref}', flush=True)

    if fails:
        print('\n*** FIDELITY CHECK FAILED ***')
        for f_ in fails:
            print('   ' + f_)
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print('\nfidelity OK: fingerprints reproduce the run and the 100 m ceiling '
          'equals the stored auc_oracle bit for bit')
    print('\n-- coarse-pixel AUC ceiling, mean over the three species --')
    print(df.groupby(['arm', 'level'])[['auc_ceiling_coarse', 'auc_oracle_point']]
          .mean().round(4).to_string())
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
