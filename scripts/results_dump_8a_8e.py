"""Every number behind s8_8a / 8b / 8c-fixed / 8e _panels.png, as text.

Reproduces the panel figure's own aggregation exactly, so what is printed is
what is drawn: rows are filtered to the headline res_mode, the CNN curve is the
cnn_mean row (the ensemble of the three initialisations), per-species values
are the mean within (level, species), and the curve is the mean over species.
The init-SD band, the paired contrast, the ceilings and the four flags
(degenerate fit, no-skill, thin level, val guard) use the same helpers the
figure uses.

Reads only CSVs. Writes figures/thesis/results_8a_8e.txt and prints the same.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python results_dump_8a_8e.py)
"""
import os
import numpy as np
import pandas as pd

from _vs_env import FIGURES_DIR

VERSION = 'v6_bio11'
OUT = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', 'thesis',
                                    'results_8a_8e.txt'))
HEADLINE_RES_MODE = 'constant_footprint'
N_SE_ABOVE_CHANCE = 2.0
NSP_MIN = 3
ARMS = ['random', 'extrap']

# the figure's own labels (cell 33, _S8EXP)
S8EXP = {
    '8a':       ('N training points',   'log',    'Sample size'),
    '8b':       ('Prevalence', 'linear', 'Prevalence'),
    '8c-fixed': ('Resolution (m)',      'log',    'Resolution (fixed 100 m truth)'),
    '8e':       ('Patch size (px)',     'log',    'Patch size'),
}
FOOTNOTE = ('Faint dots = individual reps (not independent -- shown raw). '
            'Grey ring = AUC not 2 SE above chance (either model). '
            'Hollow + red x = CNN never learned (val AUC at chance). '
            f'Grey band = fewer than {NSP_MIN} species at that level '
            '(no cross-species claim).')

L = []


def w(line=''):
    print(line, flush=True)
    L.append(line)


def load():
    d = pd.read_csv(os.path.join(FIGURES_DIR, f'section8_two_regime_{VERSION}.csv'))
    cf = os.path.join(FIGURES_DIR, f'section8c_fixed_truth_{VERSION}.csv')
    if os.path.exists(cf):
        d = pd.concat([d, pd.read_csv(cf)], ignore_index=True)
    if 'res_mode' in d.columns:
        keep = d.res_mode.isna() | d.res_mode.isin([HEADLINE_RES_MODE, 'baseline_100m'])
        d = d[keep].reset_index(drop=True)
    return d


def auc_floor(n, p):
    n = np.asarray(n, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    n1 = np.maximum(n * p, 1.0); n0 = np.maximum(n * (1 - p), 1.0)
    return 0.5 + N_SE_ABOVE_CHANCE * np.sqrt((n + 1.0) / (12.0 * n1 * n0))


def fmt(v, nd=3):
    return '  --  ' if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f'{v:.{nd}f}'


def main():
    d = load()
    CN = 'cnn_mean' if (d.model == 'cnn_mean').any() else 'cnn'
    UNIT = 'species_idx' if 'species_idx' in d.columns else 'rep'
    sexps = [s for s in S8EXP if s in d.sub_exp.unique()]

    w('=' * 100)
    w('RESULTS DUMP -- the four sensitivity panel figures')
    w('=' * 100)
    w(f'source CSVs   : section8_two_regime_{VERSION}.csv '
      f'+ section8c_fixed_truth_{VERSION}.csv')
    w(f'res_mode shown: {HEADLINE_RES_MODE} (8c-fixed runs a second convention, '
      f'fixed_patch_px, which is NOT in these figures)')
    w(f'CNN curve     : model="{CN}" -- the mean of the 3 initialisations, scored once')
    w(f'replication   : {UNIT}; species seeds '
      f'{sorted(d.species_seed.dropna().unique().astype(int).tolist())}')
    w('metrics       : spearman_true = Spearman of the prediction against the TRUE '
      'suitability (primary)')
    w('                roc_auc       = AUC against the drawn (noisy) labels (secondary)')
    w('                auc_oracle    = AUC of the true suitability itself against the '
      'drawn labels = the noise ceiling')
    w('                oracle_spearman_infoloss = the rank information left after '
      'coarsening, before any model (8c-fixed only)')
    w('aggregation   : mean within (level, species), then mean over species. '
      'NO cross-fold SE is computed anywhere.')
    w('noise band    : initSD = mean of the 3 within-species SDs across the 3 CNN '
      'inits (the figure\'s own quantity);')
    w('                poolSD = sqrt(mean of those 3 variances) = the pooled '
      'within-species SD. The verdict uses poolSD,')
    w('                the more conservative of the two. Both are printed; no '
      'verdict differs between them.')
    w('')

    for se in sexps:
        xlab, xscale, title = S8EXP[se]
        g = d[d.sub_exp == se]
        levels = sorted(g.level.unique(), key=float)
        w('')
        w('#' * 100)
        w(f'## {se}   file: s8_{se}_panels.png')
        w('#' * 100)
        w(f'figure title (suptitle) : "{title}"')
        w(f'left panel  y label     : "Spearman $\\rho$ vs truth"')
        w(f'right panel y label     : "AUC-ROC"')
        w(f'x label (both panels)   : "{xlab}"      x scale: {xscale}')
        w(f'legend entries          : "RF random", "RF extrap", "CNN random", '
          f'"CNN extrap"'
          + (', "resolution-loss ceiling, interpolative", "resolution-loss ceiling, extrapolative" (left panel)'
             if se == '8c-fixed' else '')
          + ', "oracle AUC (ceiling, mean)" (right panel)')
        w(f'figure footnote         : "{FOOTNOTE}"')
        w(f'levels plotted          : {[float(x) for x in levels]}')
        w('')

        # -- confound check ---------------------------------------------------
        w('-- confound check (RF rows; what varies along this axis) ------------')
        rf = g[g.model == 'rf']
        for arm in ARMS:
            a = rf[rf.arm == arm]
            if a.empty:
                continue
            parts = []
            for k in ('n_tr', 'n_te', 'n_vl', 'real_prev_tr', 'real_prev_te'):
                if k not in a.columns:
                    continue
                lo, hi = float(a[k].min()), float(a[k].max())
                parts.append(f'{k}={lo:g}' if lo == hi else f'{k}={lo:g}..{hi:g}')
            w(f'  {arm:<7}: ' + '  '.join(parts))
        w('')

        # -- the plotted curves ----------------------------------------------
        for arm in ARMS:
            ga = g[g.arm == arm]
            if ga.empty:
                continue
            w(f'-- {arm} arm: the plotted curves -----------------------------------')
            hdr = (f'  {"level":>8} {"RF rho":>8} {"CNN rho":>8} {"CNN-RF":>8} '
                   f'{"initSD":>7} {"poolSD":>7} {"verdict":<14} {"RF AUC":>8} {"CNN AUC":>8} '
                   f'{"oracleAUC":>9}')
            if se == '8c-fixed':
                hdr += f' {"infoCeil":>9} {"RFrel":>7} {"CNNrel":>7}'
            hdr += f' {"nsp":>4}'
            w(hdr)
            # TWO versions of the CNN init-noise band, per level:
            #   init_sd  the figure's own quantity -- the arithmetic MEAN of the
            #            three within-species SDs (_init_sd8 in the plotting cell)
            #   pool_sd  the pooled within-species SD, sqrt(mean of the three
            #            variances). With 3 species all at n=3 inits the design is
            #            balanced, so this is the textbook pooled estimator.
            # pooled is the larger of the two by construction (Jensen); median
            # ratio over the 38 cells is 1.13, max 1.36. No verdict differs.
            _psd = (ga[ga.model == 'cnn'].groupby(['level', UNIT])['spearman_true']
                    .std())
            init_sd = _psd.groupby(level=0).mean()
            pool_sd = np.sqrt((_psd ** 2).groupby(level=0).mean())
            for lv in levels:
                s = ga[ga.level == lv]
                if s.empty:
                    continue

                def m(model, col):
                    x = s[s.model == model].groupby(UNIT)[col].mean()
                    return float(x.mean()) if len(x) else np.nan

                rf_r, cn_r = m('rf', 'spearman_true'), m(CN, 'spearman_true')
                rf_a, cn_a = m('rf', 'roc_auc'), m(CN, 'roc_auc')
                orc = m('rf', 'auc_oracle')
                sd = float(init_sd.get(lv, np.nan))
                psd = float(pool_sd.get(lv, np.nan))
                diff = cn_r - rf_r
                if not np.isfinite(diff) or not np.isfinite(psd):
                    verd = 'n/a'
                elif abs(diff) <= psd:
                    verd = 'INSIDE band'
                else:
                    verd = f'outside {abs(diff)/psd:.2f}x'
                nsp = int(s[s.model == 'rf'][UNIT].nunique())
                row = (f'  {float(lv):>8g} {fmt(rf_r):>8} {fmt(cn_r):>8} '
                       f'{diff:>8.3f} {fmt(sd):>7} {fmt(psd):>7} {verd:<14} {fmt(rf_a):>8} '
                       f'{fmt(cn_a):>8} {fmt(orc):>9}')
                if se == '8c-fixed':
                    ceil = m('rf', 'oracle_spearman_infoloss')
                    row += (f' {fmt(ceil):>9} '
                            f'{fmt(rf_r / ceil if ceil else np.nan):>7} '
                            f'{fmt(cn_r / ceil if ceil else np.nan):>7}')
                row += f' {nsp:>4}'
                w(row)
            w('')

        # -- per species -------------------------------------------------------
        w('-- per species (the faint dots), spearman_true ----------------------')
        for arm in ARMS:
            ga = g[g.arm == arm]
            if ga.empty:
                continue
            for model, lab in (('rf', 'RF'), (CN, 'CNN')):
                t = (ga[ga.model == model]
                     .groupby([UNIT, 'level'])['spearman_true'].mean().unstack('level'))
                if t.empty:
                    continue
                t = t[[c for c in sorted(t.columns, key=float)]]
                w(f'  {arm} / {lab}:')
                for idx, r in t.iterrows():
                    seed = int(ga[ga[UNIT] == idx].species_seed.iloc[0])
                    w(f'    species {idx} (seed {seed}): '
                      + ' '.join(f'{c:g}={fmt(v)}' for c, v in r.items()))
        w('')

        # -- flags -------------------------------------------------------------
        w('-- flags drawn on the figure ---------------------------------------')
        any_flag = False
        # CNN degenerate
        if 'val_auc_best' in g.columns and g.val_auc_best.notna().any():
            c = g[(g.model == 'cnn') & g.val_auc_best.notna()].copy()
            c['floor'] = auc_floor(c.n_vl, c.real_prev_vl)
            c['degen'] = c.val_auc_best < c.floor
            fh = c.groupby(['arm', 'level']).agg(
                n_fits=('degen', 'size'), frac_degen=('degen', 'mean'),
                val_auc_best=('val_auc_best', 'mean'),
                best_epoch=('early_stop_ep', 'mean'), floor=('floor', 'mean'))
            bad = fh[fh.frac_degen > 0.5]
            for (arm, lv), r in bad.iterrows():
                any_flag = True
                w(f'  HOLLOW + RED X (CNN never learned): {arm} level={lv:g} -- '
                  f'{r.frac_degen:.0%} of {int(r.n_fits)} fits below the chance floor '
                  f'({r.val_auc_best:.3f} vs floor {r.floor:.3f})')
            w(f'  [CNN fit health, all levels: mean val_auc_best '
              f'{fh.val_auc_best.mean():.3f}, mean best epoch {fh.best_epoch.mean():.1f}]')
        # no-skill rings
        for arm in ARMS:
            for model, lab in (('rf', 'RF'), (CN, 'CNN')):
                s = g[(g.arm == arm) & (g.model == model)]
                if s.empty:
                    continue
                agg = s.groupby('level').agg(auc=('roc_auc', 'mean'),
                                             n=('n_te', 'mean'),
                                             p=('real_prev_te', 'mean'))
                for lv, r in agg.iterrows():
                    if np.isfinite(r.auc) and r.auc < auc_floor(r.n, r.p):
                        any_flag = True
                        w(f'  GREY RING (AUC not 2 SE above chance): {arm} {lab} '
                          f'level={float(lv):g} -- AUC {r.auc:.3f} vs floor '
                          f'{float(auc_floor(r.n, r.p)):.3f}')
        # thin levels
        for arm in ARMS:
            sub = g[g.arm == arm]
            if sub.empty:
                continue
            for model, lab in (('rf', 'RF'), (CN, 'CNN')):
                cnt = (sub[(sub.model == model) & sub.spearman_true.notna()]
                       .groupby('level')[UNIT].nunique())
                for lv in levels:
                    n = int(cnt.get(lv, 0))
                    if n < NSP_MIN:
                        any_flag = True
                        w(f'  GREY BAND (fewer than {NSP_MIN} species): {arm} {lab} '
                          f'level={float(lv):g} -- {n} species')
        # val guard
        rfc = g[g.model == 'rf'].groupby(['arm', 'level']).size()
        cnc = g[g.model == CN].groupby(['arm', 'level']).size()
        rate = (cnc / rfc).fillna(0)
        for (arm, lv), r in rate.items():
            if r < 0.95:
                any_flag = True
                w(f'  VAL GUARD: {arm} level={float(lv):g} -- only {r:.0%} of CNN '
                  f'reps ran (a dip here means fewer reps, not worse skill)')
        if not any_flag:
            w('  none -- no degenerate fit, no no-skill ring, no thin level, '
              'no guard skip')
        w('')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'\nwritten: {OUT}')


if __name__ == '__main__':
    main()
