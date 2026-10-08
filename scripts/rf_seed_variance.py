"""How much does the Random Forest move when only its random_state changes?

The pipeline fits ONE RF per cell (random_state=42, _s8_rf) and gives the CNN
three initialisations, on the rule that the CNN carries the init-variance
budget (the analysis plan). That rule had never been checked for the RF: no seed sweep
of it exists anywhere in the repo.

This refits the RF with random_state in {42, 0, 1, 2, 3} on the DEFAULT cells
-- 8a at N = 2000, both arms, the three sweep species (seeds 42 / 55838 /
63757) -- and reports the SD of spearman_true and roc_auc across those five
seeds, next to the SD across the three CNN initialisations recorded for the
same cells in section8_two_regime_v6_bio11.csv. No CNN is refitted: those three
values are already in the results file.

The points, labels, split and features are produced by the notebook's own
objects with the v6 seeds (8a seed base 4000 + species_idx*100000, +100 per
level, N = 2000 being the 5th level -> +400), following _s8_run_level branch by
branch, so the random_state=42 fit must reproduce the RF row of the v6 CSV.
That equality is checked and printed per cell; it is what certifies that the
other four seeds differ from it only in random_state.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python rf_seed_variance.py)
Writes: figures/sus_scrofa/synthetic/rf_seed_variance.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np
import pandas as pd

from _vs_env import notebook_env, FIGURES_DIR

RF_SEEDS = [42, 0, 1, 2, 3]
SPECIES = [(0, 42), (1, 55838), (2, 63757)]      # the three admitted in v6
LEVEL_N = 2000
SEED_OFFSET = 400                                # 5th level of N_POINTS_RANGE
V6_CSV = os.path.join(FIGURES_DIR, 'section8_two_regime_v6_bio11.csv')


def main():
    g = notebook_env(with_config=False)
    np.seterr(all='ignore')
    env_arr = g['resample_env'](g['env_scaled_100m'], 100)
    PATCH, PRES = g['PATCH'], g['PRES_RATIO']
    NW, SEP = g['_S8_N_WIN'], PATCH * g['_S8_SEP_PATCHES']
    sample, tts, mets, RFC = (g['_s8_sample_points'], g['_tts'],
                              g['_s8_mets'], g['_RFC'])
    hp = dict(g['RF_BEST_HP'])

    v6 = pd.read_csv(V6_CSV)
    v6 = v6[(v6.sub_exp == '8a') & (v6.level == LEVEL_N)]

    patch_ok = g['_s8_patch_ok'](env_arr, g['_s8_patch_buffer_px'](100, PATCH, None))
    tr_b, val_b, te_b = g['_s8_region_masks'](env_arr, PATCH)
    rows = []

    for sp_i, sp_seed in SPECIES:
        _r = g['make_vs'](sp_seed, mode='pca', occ='probabilistic',
                          k=g['_S8_OCC_K'], prevalence=g['_S8_OCC_PREV'])
        # bound exactly as _s8_species_iter does it
        g['_s8_meta'] = {**_r[2], **g['_s8_config_niche'](sp_seed)}
        suit = g['_s8_suit_fn'](env_arr)
        g['_s8_suit_100m'] = suit
        occ_ok = np.all(np.isfinite(env_arr), axis=2) & np.isfinite(suit)
        occ_c = g['_s8_solve_c'](suit[occ_ok], g['_S8_OCC_K'], g['_S8_OCC_PREV'])
        seed = 4000 + sp_i * 100000 + SEED_OFFSET

        for arm in ('random', 'extrap'):
            if arm == 'random':
                n_draw = LEVEL_N + 2 * NW
                vm = g['make_valid_mask'](env_arr, suit, PATCH) & patch_ok
                r, c, lbl = sample(suit, vm, n_draw, PRES, seed=seed,
                                   min_sep_px=SEP, occ_prev=None, occ_c=occ_c)
                lbl = lbl.astype(np.int8)
                hold = 2 * NW if len(lbl) >= n_draw else int(
                    round(len(lbl) * (2.0 * NW) / n_draw))
                hold = max(2, min(hold, len(lbl) - 4))
                tr_i, rest = tts(np.arange(len(lbl)), test_size=hold,
                                 stratify=lbl, random_state=seed)
                _, te_i = tts(rest, test_size=0.5, stratify=lbl[rest],
                              random_state=seed + 1)
                r_tr, c_tr, y_tr = r[tr_i], c[tr_i], lbl[tr_i]
                r_te, c_te, y_te = r[te_i], c[te_i], lbl[te_i]
            else:
                tm, vmk, em = [m & patch_ok for m in (tr_b, val_b, te_b)]
                r_tr, c_tr, l_tr = sample(suit, tm, LEVEL_N, PRES, seed=seed,
                                          min_sep_px=SEP, occ_prev=None, occ_c=occ_c)
                y_tr = l_tr.astype(np.int8)
                used = list(zip(r_tr.astype(float), c_tr.astype(float)))
                r_v, c_v, _ = sample(suit, vmk, min(NW, int(vmk.sum())), PRES,
                                     seed=seed + 100, min_sep_px=SEP,
                                     occ_prev=None, occ_c=occ_c, seed_grid=used)
                used += list(zip(r_v.astype(float), c_v.astype(float)))
                r_te, c_te, l_te = sample(suit, em, min(NW, int(em.sum())), PRES,
                                          seed=seed + 200, min_sep_px=SEP,
                                          occ_prev=None, occ_c=occ_c, seed_grid=used)
                y_te = l_te.astype(np.int8)

            # features exactly as _s8_run_level builds them for the RF
            E_tr = env_arr[r_tr, c_tr, :].astype(np.float32)
            ok_tr = ~np.isnan(E_tr).any(1)
            E_te = env_arr[r_te, c_te, :].astype(np.float32)
            ok_te = ~np.isnan(E_te).any(1)
            s_te = suit[r_te, c_te]

            for rs in RF_SEEDS:
                clf = RFC(random_state=rs, n_jobs=1, **hp)
                clf.fit(E_tr[ok_tr], y_tr[ok_tr])
                p = np.full(len(r_te), np.nan)
                if ok_te.sum() >= 5:
                    p[ok_te] = clf.predict_proba(E_te[ok_te])[:, 1]
                m = mets(p, y_te, s_te)
                rows.append(dict(species_idx=sp_i, species_seed=sp_seed, arm=arm,
                                 rf_random_state=rs, n_tr=int(len(y_tr)),
                                 n_te=int(len(y_te)), spearman_true=m['spearman_true'],
                                 roc_auc=m['roc_auc']))

            ref = v6[(v6.species_idx == sp_i) & (v6.arm == arm) & (v6.model == 'rf')]
            got = rows[-len(RF_SEEDS)]          # the random_state=42 fit
            ds = abs(float(ref.spearman_true.iloc[0]) - got['spearman_true'])
            da = abs(float(ref.roc_auc.iloc[0]) - got['roc_auc'])
            print(f'  sp{sp_i} {arm:<7} n_tr={got["n_tr"]} n_te={got["n_te"]}  '
                  f'seed42 vs v6 row: d_spearman={ds:.2e} d_auc={da:.2e} '
                  f'{"OK" if max(ds, da) < 1e-9 else "MISMATCH"}', flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(FIGURES_DIR, 'rf_seed_variance.csv')
    df.to_csv(out, index=False)

    rf = df.groupby(['species_idx', 'arm'])[['spearman_true', 'roc_auc']].agg(['mean', 'std', 'min', 'max'])
    cnn = (v6[v6.model == 'cnn'].groupby(['species_idx', 'arm'])
           [['spearman_true', 'roc_auc']].agg(['mean', 'std']))
    print('\n-- RF over 5 random_state values (42, 0, 1, 2, 3) --')
    print(rf.round(4).to_string())
    print('\n-- CNN over its 3 initialisations (v6 CSV, same cells) --')
    print(cnn.round(4).to_string())
    comp = pd.DataFrame({
        'rf_sd_spearman': rf[('spearman_true', 'std')],
        'cnn_sd_spearman': cnn[('spearman_true', 'std')],
        'rf_range_spearman': rf[('spearman_true', 'max')] - rf[('spearman_true', 'min')],
        'rf_sd_auc': rf[('roc_auc', 'std')],
        'cnn_sd_auc': cnn[('roc_auc', 'std')],
        'rf_range_auc': rf[('roc_auc', 'max')] - rf[('roc_auc', 'min')]})
    print('\n-- SD side by side --')
    print(comp.round(4).to_string())
    print('\nmedian SD spearman: RF %.4f  CNN %.4f' %
          (comp.rf_sd_spearman.median(), comp.cnn_sd_spearman.median()))
    print('median SD auc:      RF %.4f  CNN %.4f' %
          (comp.rf_sd_auc.median(), comp.cnn_sd_auc.median()))
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
