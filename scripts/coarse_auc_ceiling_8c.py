"""A coarse-information AUC ceiling for 8c-fixed, without refitting anything.

WHAT IT IS. oracle_spearman_infoloss already answers "how well can the coarse
cell RANK the points against the fine truth". Its AUC counterpart is missing:
the AUC panel of the resolution figure carries auc_oracle, which is the AUC of
the 100 m truth AT THE POINT against the drawn labels -- a label-noise ceiling
that does not move with resolution at all (it is 0.935 / 0.869 at every level,
by construction). This script computes instead

    AUC( drawn labels , block mean of the 100 m truth over the point's
                        coarse cell )

using the same block mean as the resolution-loss ceiling
(_s8_blockmean_to_coarse_grid). It is the best AUC any model could reach if it
knew the coarse cell perfectly, so unlike auc_oracle it falls as the grain
coarsens and is the right reference for the AUC panel.

HOW. Truth, points and labels are regenerated deterministically with the run's
own seeds (8c-fixed constant_footprint, seed base 6500 + species_idx*100000,
one seed per species and arm) through the notebook's own sampler. No model is
fitted and no result file is touched.

TWO FIDELITY CHECKS, BOTH FATAL:
  1 at 100 m the coarse ceiling must equal the stored auc_oracle EXACTLY for
    every species and arm -- at that level the block mean is the identity, so
    any difference means the regenerated points are not the run's points;
  2 the regenerated points must reproduce the stored points_fingerprint.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python coarse_auc_ceiling_8c.py)
Writes: figures/thesis/coarse_auc_ceiling_8c.csv
"""
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _vs_env import notebook_env, FIGURES_DIR

VERSION = 'v6_bio11'
SPECIES = [(0, 42), (1, 55838), (2, 63757)]
SEED_BASE = 6500                      # constant_footprint offset is 0
OUT = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', 'thesis',
                                    'coarse_auc_ceiling_8c.csv'))


def draw_points(g, truth_100, elig, bands, resolution, patch, seed, arm):
    """The test fold of one (level, arm), exactly as _s8_run_level_fixedtruth
    draws it: same masks, same separation, same seeds, same split."""
    sample, tts = g['_s8_sample_points'], g['_tts']
    NW = g['_S8_N_WIN']
    sep100 = patch * resolution / 100.0 * g['_S8_SEP_PATCHES']
    occ_c = g['_s8_solve_c'](truth_100[bands['ok_truth']], g['_S8_OCC_K'],
                             g['_S8_OCC_PREV'])
    pres = g['PRES_RATIO']
    if arm == 'random':
        n_draw = int(g['_S8_N_POINTS']) + 2 * NW
        r, c, lbl = sample(truth_100, elig, n_draw, pres, seed=seed,
                           min_sep_px=sep100, occ_prev=None, occ_c=occ_c)
        lbl = lbl.astype(np.int8)
        hold = 2 * NW if len(lbl) >= n_draw else int(
            round(len(lbl) * (2.0 * NW) / n_draw))
        hold = max(2, min(hold, len(lbl) - 4))
        tr_i, rest = tts(np.arange(len(lbl)), test_size=hold, stratify=lbl,
                         random_state=seed)
        val_i, te_i = tts(rest, test_size=0.5, stratify=lbl[rest],
                          random_state=seed + 1)
        return ((r[tr_i], c[tr_i], lbl[tr_i]), (r[val_i], c[val_i], lbl[val_i]),
                (r[te_i], c[te_i], lbl[te_i]))
    tr_m, val_m, te_m = bands['tr'], bands['val'], bands['te']
    r_tr, c_tr, l_tr = sample(truth_100, tr_m, g['_S8_N_POINTS'], pres, seed=seed,
                              min_sep_px=sep100, occ_prev=None, occ_c=occ_c)
    used = list(zip(r_tr.astype(float), c_tr.astype(float)))
    r_v, c_v, l_v = sample(truth_100, val_m, min(NW, int(val_m.sum())), pres,
                           seed=seed + 100, min_sep_px=sep100, occ_prev=None,
                           occ_c=occ_c, seed_grid=used)
    used += list(zip(r_v.astype(float), c_v.astype(float)))
    r_te, c_te, l_te = sample(truth_100, te_m, min(NW, int(te_m.sum())), pres,
                              seed=seed + 200, min_sep_px=sep100, occ_prev=None,
                              occ_c=occ_c, seed_grid=used)
    return ((r_tr, c_tr, l_tr.astype(np.int8)), (r_v, c_v, l_v.astype(np.int8)),
            (r_te, c_te, l_te.astype(np.int8)))


def main():
    g = notebook_env(with_config=True)       # 8c-fixed definitions included
    env = g['env_scaled_100m']
    levels = g['_S8_RES_CONSTFOOT']
    ladder = [(r, g['_s8_constfoot_patch'](r)) for r in levels]
    print(f'\nladder (constant footprint): {ladder}', flush=True)

    # float_precision='round_trip' is REQUIRED for the 100 m identity check.
    # pandas' default C parser is off by one ULP on some values: the stored
    # auc_oracle of species 1 / random reads back as 0.9456444444444444 from a
    # file that contains 0.9456444444444445, which is the value this script
    # recomputes. Without it the check fails on the reader, not on the data.
    ref = pd.read_csv(os.path.join(FIGURES_DIR,
                                   f'section8c_fixed_truth_{VERSION}.csv'),
                      float_precision='round_trip')
    ref = ref[(ref.res_mode == 'constant_footprint') & (ref.model == 'rf')]

    mask_cf = g['_s8_fixed_common_mask'](env, ladder)
    tr_b, val_b, te_b = g['_s8_region_masks'](env, 0)
    rows, fails = [], []

    for sp_i, sp_sd in SPECIES:
        r0 = g['make_vs'](sp_sd, mode='pca', occ='probabilistic',
                          k=g['_S8_OCC_K'], prevalence=g['_S8_OCC_PREV'])
        g['_s8_meta'] = {**r0[2], **g['_s8_config_niche'](sp_sd)}
        truth_100 = g['_s8_suit_fn'](env)
        g['_s8_suit_100m'] = truth_100
        ok_truth = np.isfinite(truth_100) & np.all(np.isfinite(env), axis=2)
        elig = mask_cf & ok_truth
        bands = dict(ok_truth=ok_truth, tr=tr_b & elig, val=val_b & elig,
                     te=te_b & elig)
        seed = SEED_BASE + sp_i * 100000

        for res, patch in ladder:
            f = int(round(res / 100))
            truth_blk = g['_s8_blockmean_to_coarse_grid'](truth_100, res)
            for arm in ('random', 'extrap'):
                (r_tr, c_tr, y_tr), (r_v, c_v, y_v), (r_te, c_te, y_te) = \
                    draw_points(g, truth_100, elig, bands, res, patch, seed, arm)

                fp = hashlib.sha1(np.concatenate(
                    [r_tr, c_tr, y_tr, r_v, c_v, y_v, r_te, c_te, y_te]
                ).astype(np.int64).tobytes()).hexdigest()[:12]

                s_te = truth_100[r_te, c_te]
                blk = truth_blk[r_te // f, c_te // f]
                ok = np.isfinite(blk) & np.isfinite(s_te)
                auc_point = float(roc_auc_score(y_te[ok], s_te[ok]))
                auc_coarse = float(roc_auc_score(y_te[ok], blk[ok]))

                row = ref[(ref.species_idx == sp_i) & (ref.level == res)
                          & (ref.arm == arm)]
                fp_ref = row.points_fingerprint.iloc[0] if not row.empty else None
                auc_ref = float(row.auc_oracle.iloc[0]) if not row.empty else np.nan
                if fp != fp_ref:
                    fails.append(f'fingerprint mismatch: species {sp_i} {arm} '
                                 f'{res} m -- regenerated {fp}, stored {fp_ref}')
                if res == 100 and not np.isclose(auc_coarse, auc_ref, atol=0, rtol=0):
                    fails.append(f'100 m ceiling != stored auc_oracle: species '
                                 f'{sp_i} {arm} -- {auc_coarse!r} vs {auc_ref!r}')
                rows.append(dict(species_idx=sp_i, species_seed=sp_sd, arm=arm,
                                 level=res, patch=patch, n_te=int(len(y_te)),
                                 auc_oracle_point=auc_point,
                                 auc_oracle_stored=auc_ref,
                                 auc_ceiling_coarse=auc_coarse,
                                 fingerprint=fp, fingerprint_stored=fp_ref))
                print(f'  sp{sp_i} {arm:<7} {res:>4} m  n_te={len(y_te)}  '
                      f'coarse={auc_coarse:.6f}  point={auc_point:.6f}  '
                      f'stored={auc_ref:.6f}  fp={fp}/{fp_ref}', flush=True)

    df = pd.DataFrame(rows)
    if fails:
        print('\n*** FIDELITY CHECK FAILED -- nothing further computed ***')
        for f_ in fails:
            print('   ' + f_)
        raise SystemExit(1)

    print('\nfidelity OK: every fingerprint reproduces the run, and at 100 m the '
          'coarse ceiling equals the stored auc_oracle bit for bit')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print('\n-- coarse-information AUC ceiling, mean over the three species --')
    t = df.groupby(['arm', 'level'])[['auc_ceiling_coarse', 'auc_oracle_point']].mean()
    print(t.round(4).to_string())
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
