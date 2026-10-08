"""Every number behind the thesis 8i figures, as text.

Covers fig6_8i_{patch,npoints,prevalence,resolution} (two panels each) and
fig7_8i_paired_contrast (four panels). Reproduces those figures' own
aggregation, so what is printed is what is drawn: headline res_mode only, the
CNN curve is the cnn_mean row, per-species values are the mean within
(level, species), the curve is the mean over species, and the noise band is the
pooled within-species SD across the three CNN initialisations, per arm.

Reads only CSVs. Writes figures/thesis/results_8i.txt and prints the same.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python results_dump_8i.py)
"""
import os

import numpy as np
import pandas as pd

from _vs_env import FIGURES_DIR
from thesis_fig_panels_8i import (load, degenerate, auc_floor, AXES, SHORT,
                                  ARMS, N_SE_ABOVE_CHANCE)

OUT = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', 'thesis',
                                    'results_8i.txt'))
MODELS = ['rf', 'rf_patch_summary', 'rf_oracle_metrics', 'cnn_mean']
L = []


def w(line=''):
    print(line, flush=True)
    L.append(line)


def fmt(v, nd=3):
    return '  --  ' if v is None or not np.isfinite(v) else f'{v:.{nd}f}'


def main():
    i_all, a0 = load()
    UNIT = 'species_idx'
    deg1, deg0 = degenerate(i_all, UNIT), degenerate(a0, UNIT)

    w('=' * 100)
    w('RESULTS DUMP -- the 8i figures (configurational DGP, alpha = 1)')
    w('=' * 100)
    w('figures       : fig6_8i_{patch,npoints,prevalence,resolution}  (2 panels each)')
    w('                fig7_8i_paired_contrast                        (4 panels)')
    w('source CSVs   : section8i_alpha_axes_v6_bio11.csv (alpha = 1)')
    w('                section8_two_regime_v6_bio11.csv + '
      'section8c_fixed_truth_v6_bio11.csv (alpha = 0)')
    w('res_mode      : constant_footprint only. The 8i resolution arm also ran '
      'fixed_patch_px; it is NOT in these figures.')
    w('replication   : species_idx; species seeds '
      f'{sorted(i_all.species_seed.dropna().unique().astype(int).tolist())}')
    w('')
    w('MODELS')
    w('  RF-center  (rf)                centre cell of the coarsened covariates, '
      'plus the W channel')
    w('  RF-patch   (rf_patch_summary)  + mean and SD of the SAME W channel over a '
      'patch-sized window;')
    w('                                 no reference to r_m and none to the truth -- '
      'composition without configuration')
    w('  RF-oracle  (rf_oracle_metrics) + the raw driver fields P_w and E at the '
      "truth's own radius r_m.")
    w('                                 NOT patch-limited, and dropped from the '
      'patch figure because its two')
    w('                                 features do not depend on the patch at all '
      '(cell 37: Pw_oracle_2d = Pw_2d)')
    w('  CNN        (cnn_mean)          the full patch, mean of the 3 initialisations')
    w('')
    w('CEILINGS -- two different things, do not conflate them')
    w('  ceiling_config_patch  E[S_config | patch-mean cover, patch-mean edge '
      'density], scored at each row\'s')
    w('                        own test points. It bounds every PATCH-LIMITED model. '
      'Measured over all 114 rows')
    w('                        per model: CNN 0 above it, RF-patch 0, RF-center 0, '
      'RF-oracle 34 (29.8 %,')
    w('                        max excess 0.740) -- which is what "RF-oracle is not '
      'patch-limited" means.')
    w('  auc_oracle            AUC of the true suitability against the drawn labels '
      '= the label-noise ceiling.')
    w('')

    cfg = i_all.iloc[0]
    w(f'DGP: r_m={cfg.r_m:.0f} m, corr_len={cfg.corr_len_m:.0f} m, '
      f'frac_woody={cfg.frac_woody:.2f} (realised {cfg.frac_woody_realized:.3f}), '
      f'w_P={cfg.w_P}, sig_P={cfg.sig_P}, mu_P={cfg.mu_P:.4f}')
    w('')

    for sexp, xlabel, a0se, short in AXES:
        g = i_all[i_all.sub_exp == sexp]
        if g.empty:
            continue
        levels = sorted(g.level.dropna().unique().astype(float))
        w('')
        w('#' * 100)
        w(f'## {short}   file: fig6_8i_{short}.pdf   (alpha = 1 sub_exp: {sexp})')
        w('#' * 100)
        w(f'panel (a): absolute skill at alpha = 1, both arms, with the '
          f'patch-limited ceiling')
        w(f'panel (b): the same axis in both regimes, EXTRAPOLATIVE arm '
          f'(alpha = 0 from {a0se})')
        w(f'x label  : "{xlabel}"   x scale: log   levels: '
          f'{[f"{v:g}" for v in levels]}')
        w('')

        # -- confound check ---------------------------------------------------
        w('-- confound check (RF rows) ----------------------------------------')
        for arm in ARMS:
            a = g[(g.model == 'rf') & (g.arm == arm)]
            if a.empty:
                continue
            parts = []
            for k in ('n_tr', 'n_te', 'n_vl', 'real_prev_tr', 'real_prev_te'):
                lo, hi = float(a[k].min()), float(a[k].max())
                parts.append(f'{k}={lo:g}' if lo == hi else f'{k}={lo:g}..{hi:g}')
            w(f'  {arm:<7}: ' + '  '.join(parts))
        w('')

        # -- panel (a) --------------------------------------------------------
        for arm in ARMS:
            ga = g[g.arm == arm]
            if ga.empty:
                continue
            w(f'-- panel (a), {arm} arm: spearman_true -------------------------')
            hdr = f'  {"level":>9} ' + ' '.join(f'{SHORT[m]:>10}' for m in MODELS)
            hdr += f' {"ceiling":>9} {"CNNflag":>8} {"nsp":>4}'
            w(hdr)
            for lv in levels:
                s = ga[ga.level == lv]
                vals = []
                for m in MODELS:
                    x = s[s.model == m].groupby(UNIT)['spearman_true'].mean()
                    vals.append(fmt(float(x.mean()) if len(x) else np.nan))
                ceil = s['ceiling_config_patch'].mean()
                fl = 'UNFIT' if (sexp, arm, float(lv)) in deg1 else ''
                nsp = int(s[s.model == 'rf'][UNIT].nunique())
                w(f'  {lv:>9g} ' + ' '.join(f'{v:>10}' for v in vals)
                  + f' {fmt(ceil):>9} {fl:>8} {nsp:>4}')
            w('')

        # roc_auc as a secondary read
        w('-- roc_auc (secondary), mean over species --------------------------')
        for arm in ARMS:
            ga = g[g.arm == arm]
            if ga.empty:
                continue
            row = []
            for lv in levels:
                s = ga[ga.level == lv]
                x = s[s.model == 'cnn_mean'].groupby(UNIT)['roc_auc'].mean()
                y = s[s.model == 'rf_patch_summary'].groupby(UNIT)['roc_auc'].mean()
                o = s[s.model == 'rf'].groupby(UNIT)['auc_oracle'].mean()
                row.append(f'{lv:g}: CNN={fmt(x.mean())} RFpatch={fmt(y.mean())} '
                           f'oracleAUC={fmt(o.mean())}')
            w(f'  {arm:<7}: ' + '  |  '.join(row))
        w('')

        # -- panel (b) --------------------------------------------------------
        w('-- panel (b), EXTRAPOLATIVE arm: alpha = 0 vs alpha = 1 ------------')
        w(f'  {"level":>9} {"RF a=0":>9} {"RF a=1":>9} {"CNN a=0":>9} '
          f'{"CNN a=1":>9}  flags')
        for lv in levels:
            r0 = (a0[(a0.sub_exp == a0se) & (a0.model == 'rf')
                     & (a0.arm == 'extrap') & (a0.level == lv)]['spearman_true'].mean())
            c0 = (a0[(a0.sub_exp == a0se) & (a0.model == 'cnn_mean')
                     & (a0.arm == 'extrap') & (a0.level == lv)]['spearman_true'].mean())
            r1 = (g[(g.model == 'rf') & (g.arm == 'extrap')
                    & (g.level == lv)]['spearman_true'].mean())
            c1 = (g[(g.model == 'cnn_mean') & (g.arm == 'extrap')
                    & (g.level == lv)]['spearman_true'].mean())
            fl = []
            if (a0se, 'extrap', float(lv)) in deg0:
                fl.append('a=0 CNN UNFIT')
            if (sexp, 'extrap', float(lv)) in deg1:
                fl.append('a=1 CNN UNFIT')
            w(f'  {lv:>9g} {fmt(r0):>9} {fmt(r1):>9} {fmt(c0):>9} {fmt(c1):>9}'
              f'  {", ".join(fl)}')
        w('')

        # -- per species ------------------------------------------------------
        w('-- per species (the faint markers), spearman_true, alpha = 1 -------')
        for arm in ARMS:
            ga = g[g.arm == arm]
            for m in MODELS:
                t = (ga[ga.model == m].groupby([UNIT, 'level'])['spearman_true']
                     .mean().unstack('level'))
                if t.empty:
                    continue
                t = t[[c for c in sorted(t.columns, key=float)]]
                for idx, r in t.iterrows():
                    seed = int(ga[ga[UNIT] == idx].species_seed.iloc[0])
                    w(f'  {arm:<7} {SHORT[m]:<10} species {idx} (seed {seed}): '
                      + ' '.join(f'{c:g}={fmt(v)}' for c, v in r.items()))
        w('')

        # -- fig7: the paired contrast on this axis ---------------------------
        w('-- fig7 panel: CNN - RF-patch, paired within species ---------------')
        w(f'  {"level":>9} {"arm":<7} {"delta":>9} {"poolSD":>8} {"verdict":<14} '
          f'flags')
        for arm in ARMS:
            ga = g[g.arm == arm]
            sd = (ga[ga.model == 'cnn'].groupby(['level', UNIT])['spearman_true']
                  .std())
            band = np.sqrt((sd ** 2).groupby(level=0).mean()) if not sd.empty \
                else pd.Series(dtype=float)
            hi = ga[ga.model == 'cnn_mean'].groupby(['level', UNIT])['spearman_true'].mean()
            lo = ga[ga.model == 'rf_patch_summary'].groupby(['level', UNIT])['spearman_true'].mean()
            mu = (hi - lo).dropna().groupby(level='level').mean()
            for lv in levels:
                if lv not in mu.index:
                    continue
                v = float(mu[lv]); s = float(band.get(lv, np.nan))
                verd = ('n/a' if not np.isfinite(s) else
                        'INSIDE band' if abs(v) <= s else
                        f'outside {abs(v)/s:.2f}x')
                fl = 'CNN UNFIT' if (sexp, arm, float(lv)) in deg1 else ''
                w(f'  {lv:>9g} {arm:<7} {v:>+9.4f} {fmt(s):>8} {verd:<14} {fl}')
        w('')

    # -- the flag inventory ---------------------------------------------------
    w('')
    w('#' * 100)
    w('## FIT-HEALTH FLAGS -- where the CNN never learned')
    w('#' * 100)
    w(f'rule: > 50 % of the CNN fits in a cell have val AUC below '
      f'0.5 + {N_SE_ABOVE_CHANCE:.0f} SE (prevalence-aware, Mann-Whitney null SE)')
    w(f'alpha = 1: {len(deg1)} cells flagged')
    for k in sorted(deg1, key=lambda t: (t[0], t[1], float(t[2]))):
        w(f'    {k[0]:<28} {k[1]:<7} level={float(k[2]):g}')
    w(f'alpha = 0: {len(deg0)} cells flagged')
    if not deg0:
        c = a0[(a0.model == 'cnn') & a0.val_auc_best.notna()].copy()
        c['degen'] = c.val_auc_best < auc_floor(c.n_vl, c.real_prev_vl)
        h = c.groupby(['sub_exp', 'arm'])['degen'].max().groupby(level=[0, 1]).max()
        hh = c.groupby(['sub_exp', 'arm', 'level'])['degen'].mean().groupby(level=[0, 1]).max()
        w('    none. The same rule was applied; the highest fraction of degenerate '
          'fits in any alpha = 0 cell is')
        for k, v in hh.items():
            w(f'      {k[0]:<10} {k[1]:<7} max frac_degen = {v:.3f}')
        w('    so nothing crosses 0.5. This is a result, not an omission: in the '
          'pointwise regime the CNN')
        w('    almost always learns something, in the configurational regime it '
          'often does not.')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'\nwritten: {OUT}')


if __name__ == '__main__':
    main()
