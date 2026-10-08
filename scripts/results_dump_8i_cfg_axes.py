"""Sample size and prevalence under the CONFIGURATIONAL DGP (8i, alpha = 1).

A narrow companion to results_dump_8i.py, written for the Results text: only the
two axes asked for, both arms kept apart throughout, and for every level

    CNN / RF-patch / RF-oracle   spearman_true AND roc_auc
    the two ceilings             ceiling_config_patch (rank) and auc_oracle (AUC)
    the share of CNN fits that never learned
    the confound block (n_tr, n_te, n_vl, realised prevalences)

Aggregation matches the figures: the CNN curve is the cnn_mean row, per-species
values are the mean within (level, species), the curve is the mean over species.
The unlearned share is the RAW FRACTION of individual CNN fits below the
prevalence-aware chance floor -- the figures only mark a level when that
fraction exceeds 0.5, so a level can carry a non-zero share here and no flag
there.

Reads only CSVs. Writes figures/thesis/results_8i_cfg_axes.txt and prints it.

Environment: wolf_sdm
  (/opt/anaconda3/envs/wolf_sdm/bin/python results_dump_8i_cfg_axes.py)
"""
import os

import numpy as np
import pandas as pd

from _vs_env import FIGURES_DIR
from thesis_fig_panels_8i import (load, auc_floor, SHORT, ARMS,
                                  N_SE_ABOVE_CHANCE)

OUT = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', 'thesis',
                                    'results_8i_cfg_axes.txt'))
UNIT = 'species_idx'
MODELS = ['cnn_mean', 'rf_patch_summary', 'rf_oracle_metrics', 'rf']
AXES = [('8i_npoints_grf_alpha1', 'sample size',
         'Sample size (training points, n_tr)', 'fig6_8i_sample_size'),
        ('8i_prevalence_grf_alpha1', 'prevalence',
         'Prevalence', 'fig6_8i_prevalence')]
L = []


def w(line=''):
    print(line, flush=True)
    L.append(line)


def fmt(v, nd=3):
    return '  --  ' if v is None or not np.isfinite(v) else f'{v:.{nd}f}'


def curve(s, model, col):
    """Figure aggregation: mean within species, then mean over species."""
    x = s[s.model == model].groupby(UNIT)[col].mean()
    return float(x.mean()) if len(x) else np.nan


def main():
    i_all, _ = load()

    w('=' * 96)
    w('CONFIGURATIONAL DGP (alpha = 1) -- SAMPLE SIZE and PREVALENCE, by arm')
    w('=' * 96)
    w('source  : section8i_alpha_axes_v6_bio11.csv, sub_exp '
      '8i_npoints_grf_alpha1 / 8i_prevalence_grf_alpha1')
    w('figures : fig6_8i_sample_size.pdf, fig6_8i_prevalence.pdf  (panel a of each)')
    w(f'species : species_idx 0/1/2, seeds '
      f'{sorted(i_all.species_seed.dropna().unique().astype(int).tolist())}')
    w('arms    : random    = interpolative, train and test drawn from the same '
      'PC1 range')
    w('          extrap    = extrapolative, test band beyond the training band '
      'on PC1')
    w('')
    w('WHAT "n" MEANS -- total points, not presences.')
    w('  _s8_sample_points draws  n_pres = int(n_points * pres_ratio),  '
      'n_abs = n_points - n_pres,')
    w('  with pres_ratio = 0.5 throughout (balanced case-control). The recorded '
      'n_tr equals n_points')
    w('  exactly at every level of both axes (checked below), so n = 500 means '
      '250 presences + 250')
    w('  absences. The training-set PREVALENCE is therefore 0.5 everywhere and '
      'does NOT vary with the')
    w('  prevalence axis -- that axis moves the LANDSCAPE occupancy of the '
      'virtual species (the share of')
    w('  the map drawn as occupied), not the composition of the sample.')
    w('')
    w('MODELS')
    w(f'  {SHORT["cnn_mean"]:<10} full patch, mean of the 3 initialisations')
    w(f'  {SHORT["rf_patch_summary"]:<10} centre cell + mean and SD of the W '
      'channel over a patch-sized window')
    w(f'  {SHORT["rf_oracle_metrics"]:<10} centre cell + the raw driver fields '
      "P_w and E at the truth's own radius r_m")
    w(f'  {SHORT["rf"]:<10} centre cell of the coarsened covariates plus the W '
      'channel (context, not asked for)')
    w('')
    w('CEILINGS')
    w('  ceiling_config_patch   rank ceiling: E[S_config | patch-mean cover, '
      'patch-mean edge density].')
    w('                         Bounds PATCH-LIMITED models only (CNN, '
      'RF-patch, RF-center). RF-oracle')
    w('                         sees r_m directly and is not bound by it. Drawn '
      'pooled over arms in the')
    w('                         figures; both the pooled value and the per-arm '
      'values are given here.')
    w('  auc_oracle             AUC of the true suitability against the drawn '
      'labels = label-noise ceiling.')
    w('                         Model-general (it is a property of the labels). '
      'Per arm.')
    w('')
    w('UNLEARNED FITS')
    w(f'  A single CNN fit counts as unlearned when its best validation AUC is '
      f'below 0.5 + {N_SE_ABOVE_CHANCE:.0f} SE,')
    w('  SE = sqrt((n+1) / (12 n1 n0)) on the VALIDATION set (Mann-Whitney null; '
      'n_vl = 300, prev 0.5')
    w(f'  throughout, so the floor is {float(auc_floor(300, 0.5)):.4f} in every '
      'cell of these two axes). Reported as the share of')
    w('  the 3 fits per species x 3 species = 9 fits per (arm, level).')
    w('')

    for sexp, short, xlabel, stem in AXES:
        g = i_all[i_all.sub_exp == sexp]
        levels = sorted(g.level.dropna().unique().astype(float))
        w('')
        w('#' * 96)
        w(f'## {short.upper()}   x = "{xlabel}"   levels: '
          f'{[f"{v:g}" for v in levels]}')
        w(f'## file: {stem}.pdf   (sub_exp {sexp})')
        w('#' * 96)
        w('')

        # -- confound check ---------------------------------------------------
        w('-- confound check (RF rows; identical for every model) -------------')
        for arm in ARMS:
            a = g[(g.model == 'rf') & (g.arm == arm)]
            parts = []
            for k in ('n_tr', 'n_te', 'n_vl', 'real_prev_tr', 'real_prev_vl',
                      'real_prev_te'):
                lo, hi = float(a[k].min()), float(a[k].max())
                parts.append(f'{k}={lo:g}' if lo == hi else f'{k}={lo:g}..{hi:g}')
            w(f'  {arm:<7}: ' + '  '.join(parts))
        eq = all(float(g[(g.model == 'rf') & (g.level == lv)].n_tr.min()) == lv
                 and float(g[(g.model == 'rf') & (g.level == lv)].n_tr.max()) == lv
                 for lv in levels) if short == 'sample size' else None
        if eq is not None:
            w(f'  n_tr == level at every level: {eq}')
        w('')

        for arm in ARMS:
            ga = g[g.arm == arm]

            # -- Spearman -----------------------------------------------------
            w(f'-- {arm} arm | spearman_true (primary) -----------------------')
            w(f'  {"level":>8} ' + ' '.join(f'{SHORT[m]:>10}' for m in MODELS)
              + f' {"ceil_rank":>10} {"initSD":>7} {"unfit":>6}')
            for lv in levels:
                s = ga[ga.level == lv]
                vals = [fmt(curve(s, m, 'spearman_true')) for m in MODELS]
                ceil = float(s['ceiling_config_patch'].mean())
                sd = s[s.model == 'cnn'].groupby(UNIT)['spearman_true'].std()
                pooled = float(np.sqrt((sd ** 2).mean())) if len(sd) else np.nan
                c = s[(s.model == 'cnn') & s.val_auc_best.notna()]
                frac = float((c.val_auc_best
                              < auc_floor(c.n_vl, c.real_prev_vl)).mean()) \
                    if len(c) else np.nan
                w(f'  {lv:>8g} ' + ' '.join(f'{v:>10}' for v in vals)
                  + f' {fmt(ceil):>10} {fmt(pooled):>7} '
                    f'{("%.2f" % frac) if np.isfinite(frac) else "  --":>6}')
            w('')

            # -- AUC ----------------------------------------------------------
            w(f'-- {arm} arm | roc_auc (secondary) ---------------------------')
            w(f'  {"level":>8} ' + ' '.join(f'{SHORT[m]:>10}' for m in MODELS)
              + f' {"auc_oracle":>11} {"AUCfloor":>9}')
            for lv in levels:
                s = ga[ga.level == lv]
                vals = [fmt(curve(s, m, 'roc_auc')) for m in MODELS]
                orc = curve(s, 'rf', 'auc_oracle')
                fl = float(auc_floor(s.n_te.mean(), s.real_prev_te.mean()))
                w(f'  {lv:>8g} ' + ' '.join(f'{v:>10}' for v in vals)
                  + f' {fmt(orc):>11} {fmt(fl):>9}')
            w('')

            # -- ceiling detail ----------------------------------------------
            w(f'-- {arm} arm | ceiling_config_patch, per arm and pooled ------')
            row = []
            for lv in levels:
                pa = float(ga[ga.level == lv]['ceiling_config_patch'].mean())
                po = float(g[g.level == lv]['ceiling_config_patch'].mean())
                row.append(f'{lv:g}: arm={fmt(pa)} pooled={fmt(po)}')
            w('  ' + '  |  '.join(row))
            w('')

            # -- unlearned detail --------------------------------------------
            w(f'-- {arm} arm | unlearned CNN fits, raw counts ----------------')
            for lv in levels:
                c = ga[(ga.level == lv) & (ga.model == 'cnn')
                       & ga.val_auc_best.notna()]
                if c.empty:
                    continue
                bad = c.val_auc_best < auc_floor(c.n_vl, c.real_prev_vl)
                per = bad.groupby(c[UNIT]).sum().astype(int).to_dict()
                tot = bad.groupby(c[UNIT]).size().to_dict()
                det = ' '.join(f'sp{k}={per.get(k,0)}/{tot[k]}' for k in sorted(tot))
                w(f'  {lv:>8g}  {int(bad.sum())}/{len(bad)} '
                  f'({bad.mean():.2f})   {det}'
                  + ('   <- FLAGGED in the figure' if bad.mean() > 0.5 else ''))
            w('')

            # -- per species --------------------------------------------------
            w(f'-- {arm} arm | per species, spearman_true (the faint markers) -')
            for m in MODELS:
                t = (ga[ga.model == m].groupby([UNIT, 'level'])['spearman_true']
                     .mean().unstack('level'))
                if t.empty:
                    continue
                t = t[[c for c in sorted(t.columns, key=float)]]
                for idx, r in t.iterrows():
                    w(f'  {SHORT[m]:<10} species {int(idx)}: '
                      + ' '.join(f'{c:g}={fmt(v)}' for c, v in r.items()))
            w('')

            # -- CNN - RF-patch, paired --------------------------------------
            w(f'-- {arm} arm | CNN - RF-patch, paired within species ---------')
            sd = ga[ga.model == 'cnn'].groupby(['level', UNIT])['spearman_true'].std()
            band = np.sqrt((sd ** 2).groupby(level=0).mean())
            hi = ga[ga.model == 'cnn_mean'].groupby(['level', UNIT])['spearman_true'].mean()
            lo = ga[ga.model == 'rf_patch_summary'].groupby(['level', UNIT])['spearman_true'].mean()
            d = (hi - lo).dropna()
            mu = d.groupby(level='level').mean()
            w(f'  {"level":>8} {"delta":>9} {"poolSD":>8}  verdict')
            for lv in levels:
                if lv not in mu.index:
                    continue
                v, s = float(mu[lv]), float(band.get(lv, np.nan))
                verd = ('n/a' if not np.isfinite(s) else
                        'INSIDE band' if abs(v) <= s else
                        f'outside, {abs(v)/s:.2f}x the band')
                w(f'  {lv:>8g} {v:>+9.4f} {fmt(s):>8}  {verd}')
            w('')

    with open(OUT, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'\nwritten: {OUT}')


if __name__ == '__main__':
    main()
