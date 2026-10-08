"""Every number behind the thesis 8j figures, as text.

Covers fig9_8j_blend (main text), fig13_8j_blend_auc, fig10_8j_cfggain,
fig11_8j_oracle_val and fig12_8j_wsource (appendix). Reproduces those figures'
own aggregation, so what is printed is what is drawn.

Reads only the 8j CSV. Writes figures/thesis/results_8j.txt and prints it.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python results_dump_8j.py)
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from _vs_env import FIGURES_DIR
from thesis_fig_panels_8i import auc_floor, SHORT, ARMS
from thesis_fig_8j import (load, degenerate_8j, boot_ci, cfg_table, med_ci,
                           corr_frame, MAIN, WSRC, CORR_KEYS, CORR_MODELS)

OUT = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', 'thesis',
                                    'results_8j.txt'))
MODELS = ['rf', 'rf_patch_summary', 'rf_oracle_metrics', 'cnn_mean']
PRIMARY_ARM, PRIMARY_ALPHA = 'extrap', 1.0
L = []


def w(line=''):
    print(line, flush=True)
    L.append(line)


def fmt(v, nd=3):
    return '  --  ' if v is None or not np.isfinite(v) else f'{v:.{nd}f}'


def main():
    d = load()
    CN = 'cnn_mean'
    UNIT = 'species_idx'
    deg = degenerate_8j(d)
    g = d[d.sub_exp == MAIN]
    alphas = sorted(g.alpha.dropna().unique())

    w('=' * 100)
    w('RESULTS DUMP -- the 8j figures (species ensemble across the alpha continuum)')
    w('=' * 100)
    w('figures : fig9_8j_blend        MAIN TEXT -- Spearman, 2x2 = W source x arm')
    w('          fig13_8j_blend_auc    appendix  -- the same grid for AUC-ROC')
    w('          fig10_8j_corrlen      Spearman of CNN and RF-patch vs correlation length')
    w('          fig11_8j_oracle_val   primary vs oracle validation, one panel per arm')
    w('          fig12_8j_wsource      cfgGain vs alpha, 2x2 = W source x arm (fig9\'s grid)')
    w('source  : section8j_species_ensemble_v6_bio11.csv')
    w(f'species : {d.species_idx.nunique()} '
      f'(seeds {sorted(d.species_seed.unique().astype(int).tolist())})')
    w('')
    w('THE CONTRAST')
    w('  cfgGain(alpha) = Delta(alpha) - Delta(0),  Delta = CNN - RF-patch')
    w('  formed PER SPECIES against that species\' own alpha = 0 run, so differences in')
    w('  intrinsic difficulty cancel. Delta(0) is the architecture penalty: what the CNN')
    w('  gives up to the patch-summary RF when there is no configuration to find.')
    w('  The shaded areas in fig10 and fig12 are the 95 % BOOTSTRAP CI OF THE MEDIAN over')
    w('  species -- NOT the init-SD noise band of fig4/fig7, which is a different quantity.')
    w('')
    w('MODELS  RF-center = centre cell + W channel; RF-patch = + patch mean and SD of the')
    w('        SAME W channel (composition without configuration); RF-oracle = + the raw')
    w('        drivers P_w and E at the truth\'s own radius (NOT patch-limited); CNN = the')
    w('        full patch, mean of 3 initialisations.')
    w('')

    # -- confound check -------------------------------------------------------
    w('-- confound check (RF rows, alpha sweep) ---------------------------------')
    for arm in ARMS:
        a = g[(g.model == 'rf') & (g.arm == arm)]
        parts = []
        for k in ('n_tr', 'n_te', 'n_vl', 'real_prev_tr', 'real_prev_te'):
            lo, hi = float(a[k].min()), float(a[k].max())
            parts.append(f'{k}={lo:g}' if lo == hi else f'{k}={lo:g}..{hi:g}')
        w(f'  {arm:<7}: ' + '  '.join(parts))
    w('')

    # -- fig9 / fig13 ---------------------------------------------------------
    w('#' * 100)
    w('## fig9_8j_blend  (MAIN TEXT, Spearman)  and  fig13_8j_blend_auc  (APPENDIX, AUC)')
    w('#' * 100)
    w('grid   : (a) random field, interpolative   (b) random field, extrapolative')
    w('         (c) forest map,   interpolative   (d) forest map,   extrapolative')
    w('         The two figures share this grid; one carries spearman_true, the other roc_auc.')
    w('         The y axis is shared across all four panels of each figure.')
    w('x label: "$\\alpha$ (weight of the configurational DGP)"   levels: '
      f'{[f"{a:g}" for a in alphas]}')
    w('legend : RF-center, RF-patch, RF-oracle, CNN, + the ceiling. The fit-health cross is')
    w('         NOT in the legend: no 8j cell meets the > 50 % rule (per-cell shares below).')
    w('NOTE   : the alpha = 0 points come from the SAME sweep (at alpha = 0),')
    w('         not from 8j_pointwise_baseline -- so only alpha changes along the axis, and')
    w('         RF-patch / RF-oracle exist at every level (they do not in the baseline run).')
    w('NOTE   : the ceiling line starts at alpha = 0.25. ceiling_config_patch is NaN at')
    w('         alpha = 0 (those runs go through _s8_run_level, which never computes it), and')
    w('         below alpha = 1 it bounds one FACTOR of S_point^(1-alpha) * S_config^alpha,')
    w('         not the whole scoring target -- a model may sit above it legitimately there.')
    w('NOTE   : the two W fields differ in DENSITY as well as in realism: mu_P = 0.300 for the')
    w('         random field against 0.411 for the forest map (configurational_orthogonality).')
    w('')
    for sexp, wlab in WSRC:
        gw = d[d.sub_exp == sexp]
        w('=' * 90)
        w(f'== W = {wlab}   (sub_exp {sexp}, {len(gw)} rows)')
        w('=' * 90)
        for col, nm in (('spearman_true', 'spearman_true (fig9, MAIN)'),
                        ('roc_auc', 'roc_auc (fig13, APPENDIX)')):
            for arm in ARMS:
                w(f'-- {nm}, {arm} arm ----------------------------------------')
                head = f'  {"alpha":>6} ' + ' '.join(f'{SHORT[m]:>10}' for m in MODELS)
                head += (f' {"oracleAUC":>10}' if col == 'roc_auc'
                         else f' {"ceiling":>10}')
                w(head + '  flags')
                for a in alphas:
                    s = gw[(gw.arm == arm) & np.isclose(gw.alpha, a)]
                    vals = []
                    for m in MODELS:
                        x = s[s.model == m].groupby(UNIT)[col].mean()
                        vals.append(fmt(float(x.mean()) if len(x) else np.nan))
                    if col == 'roc_auc':
                        o = s[s.model == 'rf'].groupby(UNIT)['auc_oracle'].mean()
                        extra = f' {fmt(float(o.mean())):>10}'
                    else:
                        extra = f' {fmt(float(s["ceiling_config_patch"].mean())):>10}'
                    fl = []
                    if (sexp, arm, float(a)) in deg:
                        fl.append('CNN UNFIT')
                    if col == 'roc_auc':
                        for m in MODELS:
                            sm = s[s.model == m]
                            if sm.empty:
                                continue
                            au = sm['roc_auc'].mean()
                            if np.isfinite(au) and au < auc_floor(
                                    sm['n_te'].mean(), sm['real_prev_te'].mean()):
                                fl.append(f'{SHORT[m]} no-skill')
                    w(f'  {a:>6g} ' + ' '.join(f'{v:>10}' for v in vals) + extra
                      + '  ' + ', '.join(fl))
                w('')

        a1 = gw[np.isclose(gw.alpha, 1.0)]
        w('-- at alpha = 1: what each model reaches, as a share of the ceiling ------')
        for arm in ARMS:
            s = a1[a1.arm == arm]
            cl = float(s['ceiling_config_patch'].median())
            for m in MODELS:
                v = s[s.model == m].groupby(UNIT)['spearman_true'].mean().median()
                w(f'  {arm:<7} {SHORT[m]:<10} rho={v:+.3f}  ceiling={cl:.3f}  '
                  f'share={v / cl:+.3f}')
        w('')
        w('-- the CNN deficit at alpha = 0 (Delta(0) = CNN - RF-patch, per species) --')
        for arm in ARMS:
            s = gw[(gw.arm == arm) & np.isclose(gw.alpha, 0.0)]
            cn = s[s.model == CN].groupby(UNIT)['spearman_true'].mean()
            rp = s[s.model == 'rf_patch_summary'].groupby(UNIT)['spearman_true'].mean()
            dd = (cn - rp).dropna()
            w(f'  {arm:<7} CNN={cn.mean():.3f}  RF-patch={rp.mean():.3f}  '
              f'mean Delta(0)={dd.mean():+.4f}  median={dd.median():+.4f}  '
              f'range [{dd.min():+.3f}, {dd.max():+.3f}]')
        w('')
        w('-- per species, spearman_true --------------------------------------------')
        for arm in ARMS:
            for m in MODELS:
                t = (gw[(gw.arm == arm) & (gw.model == m)]
                     .groupby([UNIT, 'alpha'])['spearman_true'].mean().unstack('alpha'))
                for idx, r in t.iterrows():
                    seed = int(gw[gw[UNIT] == idx].species_seed.iloc[0])
                    w(f'  {arm:<7} {SHORT[m]:<10} sp{idx} (seed {seed}): '
                      + ' '.join(f'{c:g}={fmt(v)}' for c, v in r.items()))
        w('')

    # -- the appendix fit-health table ----------------------------------------
    w('#' * 100)
    w('## APPENDIX TABLE -- fit health, per cell')
    w('#' * 100)
    w('A single CNN fit counts as unlearned when its best validation AUC is below')
    w('0.5 + 2 SE, SE = sqrt((n+1) / (12 n1 n0)) on the VALIDATION set (Mann-Whitney null).')
    w(f'n_vl = 300 at prevalence 0.5 throughout, so the floor is '
      f'{float(auc_floor(300, 0.5)):.4f} in every 8j cell.')
    w('A cell is FLAGGED in a figure only when the share exceeds 0.5; that never happens')
    w('here, so the flag symbol appears in no 8j figure and in no legend.')
    w('')
    w(f'  {"W source":<14} {"arm":<7} ' + ' '.join(f'{a:>12g}' for a in alphas))
    for sexp, wlab in WSRC:
        gw = d[d.sub_exp == sexp]
        for arm in ARMS:
            cells = []
            for a in alphas:
                c = gw[(gw.arm == arm) & np.isclose(gw.alpha, a)
                       & (gw.model == 'cnn') & gw.val_auc_best.notna()]
                bad = c.val_auc_best < auc_floor(c.n_vl, c.real_prev_vl)
                cells.append(f'{int(bad.sum())}/{len(bad)}')
            short = 'random field' if sexp == MAIN else 'forest map'
            w(f'  {short:<14} {arm:<7} ' + ' '.join(f'{c:>12}' for c in cells))
    w('')
    w(f'  {"W source":<14} {"arm":<7} ' + ' '.join(f'{a:>12g}' for a in alphas)
      + '   (CNN init SD, pooled within species)')
    for sexp, wlab in WSRC:
        gw = d[d.sub_exp == sexp]
        for arm in ARMS:
            cells = []
            for a in alphas:
                sd = (gw[(gw.arm == arm) & np.isclose(gw.alpha, a)
                         & (gw.model == 'cnn')]
                      .groupby(UNIT)['spearman_true'].std())
                cells.append(f'{np.sqrt((sd ** 2).mean()):.3f}')
            short = 'random field' if sexp == MAIN else 'forest map'
            w(f'  {short:<14} {arm:<7} ' + ' '.join(f'{c:>12}' for c in cells))
    w('')
    w('READ: the CNN is unstable at alpha = 1 on the RANDOM FIELD (9 and 10 of 24 fits')
    w('below chance, init SD 0.280 / 0.204) and stable there on the FOREST MAP (0 of 24,')
    w('init SD 0.148 / 0.200). One sentence in the main text; the table in the appendix.')
    w('')

    # -- fig10 ----------------------------------------------------------------
    T_grf = cfg_table(d, MAIN, 'alpha')
    T_lu = cfg_table(d, '8j_alpha_landuse', 'alpha', gap0_from='8j_alpha_landuse')
    T_cl = cfg_table(d, '8j_corrlen_grf_alpha1', 'corr_len_m')
    T300 = T_grf[T_grf.key == 1.0].copy(); T300['key'] = 300.0
    T_cl = pd.concat([T_cl, T300], ignore_index=True)

    w('#' * 100)
    w('## fig10_8j_corrlen')
    w('#' * 100)
    w('panels : (a) interpolative arm   (b) extrapolative arm; x = correlation length,')
    w('         all at alpha = 1, W = random field. Shared y axis.')
    w('y label: "Spearman $\\rho$ vs truth"   drawn: CNN and RF-patch, mean over species,')
    w('         plus the eight species faintly (x-dodged). Both models are patch-limited')
    w('         and both sit at alpha = 1, so the two curves are directly comparable.')
    w('NOTE   : THIS FIGURE NO LONGER DRAWS cfgGain. A contrast cannot say WHICH of the')
    w('         two curves moved, which is the question on this axis. The cfgGain numbers')
    w('         for the same axis are kept below, for the text.')
    w('NOTE   : THE ALPHA PANEL IS GONE from this figure. It drew cfg_table(d, MAIN,')
    w('         "alpha") -- byte for byte the "random field" half of fig12, so two appendix')
    w('         figures carried the same curves. The alpha story is now fig12 alone, for')
    w('         both W sources; those numbers are in the fig12 section below.')
    w('NOTE   : the 300 m correlation length is the alpha sweep\'s own default, so its')
    w('         alpha = 1 point IS that condition (same species, same W, same seed). It is')
    w('         spliced in and ringed in the figure rather than refitted.')
    w('NOTE   : per-species strands carry a small constant x offset (5.5 % of the x range,')
    w('         spread over the 8 species) so their markers do not stack. The offset is in')
    w('         x only; every y value and the median line are untouched.')
    w('')
    w('-- spearman_true at alpha = 1 by correlation length (WHAT IS DRAWN) -------')
    gc = corr_frame(d)
    w(f'  {"arm":<7} {"model":<10} ' + ' '.join(f'{c:>10g}' for c in CORR_KEYS))
    for arm in ARMS:
        for m in CORR_MODELS + ['rf_oracle_metrics', 'rf']:
            per = (gc[(gc.model == m) & (gc.arm == arm)]
                   .groupby(['corr_len_m', UNIT])['spearman_true'].mean()
                   .unstack(UNIT).reindex(CORR_KEYS))
            w(f'  {arm:<7} {SHORT[m]:<10} '
              + ' '.join(f'{per.loc[c].mean():>10.4f}' for c in CORR_KEYS)
              + ('' if m in CORR_MODELS else '   (not drawn)'))
        w(f'  {arm:<7} {"ceiling":<10} ' + ' '.join(
            f'{gc[(gc.arm == arm) & np.isclose(gc.corr_len_m, c)].ceiling_config_patch.mean():>10.4f}'
            for c in CORR_KEYS) + '   (not drawn)')
        w(f'  {arm:<7} {"initSD":<10} ' + ' '.join(
            f'{np.sqrt((gc[(gc.arm == arm) & np.isclose(gc.corr_len_m, c) & (gc.model == "cnn")].groupby(UNIT)["spearman_true"].std() ** 2).mean()):>10.4f}'
            for c in CORR_KEYS) + '   (not drawn)')
    w('')
    w('-- per species, spearman_true, at alpha = 1 by correlation length ---------')
    for arm in ARMS:
        for m in CORR_MODELS:
            per = (gc[(gc.model == m) & (gc.arm == arm)]
                   .groupby([UNIT, 'corr_len_m'])['spearman_true'].mean()
                   .unstack('corr_len_m').reindex(columns=CORR_KEYS))
            for idx, r in per.iterrows():
                w(f'  {arm:<7} {SHORT[m]:<10} sp{int(idx)}: '
                  + ' '.join(f'{c:g}={fmt(v)}' for c, v in r.items()))
    w('')
    w('-- cfgGain at alpha = 1 by correlation length (kept for the text) ---------')
    ks = sorted(T_cl.key.unique())
    for arm in ARMS:
        m, lo, hi, n = med_ci(T_cl, arm, ks)
        for k, mm, l, h in zip(ks, m, lo, hi):
            w(f'  {arm:<7} corr_len={k:<6g} median={mm:+.4f}  '
              f'CI [{l:+.4f}, {h:+.4f}]')
    w('  (spans = 2*r_m/corr_len = 21.3 / 16.0 / 10.7 / 8.0 at 150/200/300/400 m)')
    w('')

    # -- the declared primary endpoint ----------------------------------------
    w('-- THE PRIMARY ENDPOINT, declared before the numbers were read -----------')
    w(f'  cfgGain at alpha = {PRIMARY_ALPHA:g}, {PRIMARY_ARM} arm. One test, m = 1.')
    v = T_grf.loc[(T_grf.arm == PRIMARY_ARM)
                  & (T_grf.key == PRIMARY_ALPHA), 'cfg_gain'].dropna().values
    K = len(v)
    p = wilcoxon(v)[1] if K >= 2 else np.nan
    floor = 2.0 ** -(K - 1)
    lo, hi = boot_ci(v)
    w(f'  K={K} species, median={np.median(v):+.4f}, CI [{lo:+.4f}, {hi:+.4f}], '
      f'Wilcoxon p={p:.4f}')
    w(f'  attainable floor at K={K} is p={floor:.4f} -> the test '
      + ('CAN' if floor < 0.05 else 'CANNOT') + ' reach 0.05; verdict: '
      + ('REJECT the null' if np.isfinite(p) and p < 0.05 else 'do NOT reject'))
    w('  The other 7 alpha x arm cells are DESCRIPTIVE: medians and CIs, no significance')
    w('  claim. A Holm-corrected family of all 8 would need the smallest p below 0.00625,')
    w(f'  and the floor at K={K} is {floor:.5f} -- above it, so a corrected family of 8 is')
    w('  unreachable at this K. That is why one endpoint is declared instead.')
    w('  alpha = 0 is the reference by construction (cfgGain = 0), not a result.')
    w('')

    # -- fig11 ----------------------------------------------------------------
    w('#' * 100)
    w('## fig11_8j_oracle_val')
    w('#' * 100)
    w('what   : CNN Spearman under the PRIMARY validation (val band between train and')
    w('         test) against ORACLE validation (val drawn from the test band), paired')
    w('         within species. An UPPER BOUND: it assumes labels in the target domain,')
    w('         which real extrapolative transfer does not have.')
    w('why    : in the primary design D(val,train)=1.30 against D(test,train)=1.99, so the')
    w('         CNN is model-selected under a smaller shift than it is scored at, while the')
    w('         RF does no model selection at all.')
    w('')
    pairs = [('alpha=1 (configurational)', MAIN, 1.0, '8j_oracleval_alpha1', 1.0),
             ('alpha=0 (pointwise)', '8j_pointwise_baseline', 0.0,
              '8j_oracleval_pointwise', 0.0)]
    for arm in ARMS:
        for nm, se_p, a_p, se_o, a_o in pairs:
            def by_sp(se, al):
                x = d[(d.sub_exp == se) & (d.arm == arm) & (d.model == CN)]
                if x.alpha.notna().any():
                    x = x[np.isclose(x.alpha.fillna(-999), al)]
                return x.groupby(UNIT)['spearman_true'].mean()
            pri, ora = by_sp(se_p, a_p), by_sp(se_o, a_o)
            sh = sorted(set(pri.index) & set(ora.index))
            diff = np.array([ora[s] - pri[s] for s in sh])
            lo, hi = boot_ci(diff)
            w(f'  {arm:<7} {nm:<26} primary={np.median([pri[s] for s in sh]):+.3f}  '
              f'oracle={np.median([ora[s] for s in sh]):+.3f}  '
              f'shift={np.median(diff):+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  n={len(sh)}')
    w('  ALL FOUR CIs CONTAIN ZERO: the shift is not detectable in any condition, so the')
    w('  CNN transfer deficit cannot be attributed to the validation mismatch.')
    w('')

    # -- fig12 ----------------------------------------------------------------
    w('#' * 100)
    w('## fig12_8j_wsource')
    w('#' * 100)
    w('what   : cfgGain across alpha for BOTH W sources, on fig9\'s grid -- W source by')
    w('         row, arm by column, y shared across all four. Panel (c) here sits under')
    w('         panel (c) of fig9: the same cell, absolute curves above, contrast below.')
    w('         The forest-map sweep is what keeps the result from being an artefact of')
    w('         the Gaussian random field: same species, same alphas, same machinery, W')
    w('         taken from the ESA tree-cover class instead.')
    w('NOTE   : this figure now carries the whole alpha story; fig10 kept only the')
    w('         correlation-length sweep (see there).')
    w('NOTE   : each source is referenced to ITS OWN alpha = 0 run, so the pairing does not')
    w('         carry a seed change.')
    w('')
    for nm, tab in (('random field', T_grf), ('forest map', T_lu)):
        for arm in ARMS:
            m, lo, hi, n = med_ci(tab, arm, alphas)
            for a, mm, l, h in zip(alphas, m, lo, hi):
                w(f'  {nm:<13} {arm:<7} alpha={a:<5g} median={mm:+.4f}  '
                  f'CI [{l:+.4f}, {h:+.4f}]')
    w('')

    # -- data inventory -------------------------------------------------------
    w('#' * 100)
    w('## DATA INVENTORY -- what is used where')
    w('#' * 100)
    w(f'rows: {len(d)};  sub_exp present: {sorted(d.sub_exp.unique())}')
    w('  8j_alpha_grf            fig9 + fig13 (top row), fig10 (a), fig12, fig11 (alpha=1)')
    w('  8j_alpha_landuse        fig9 + fig13 (bottom row), fig12')
    w('  8j_corrlen_grf_alpha1   fig10 (b), with the 300 m point spliced from the alpha sweep')
    w('  8j_oracleval_alpha1     fig11')
    w('  8j_oracleval_pointwise  fig11')
    w('  8j_pointwise_baseline   fig11 (alpha=0 side) and the control below')
    w('')
    w('-- control: the two alpha = 0 references (same truth, W channel present vs absent) --')
    for arm in ARMS:
        for m in ['rf', CN]:
            a = (d[(d.sub_exp == '8j_pointwise_baseline') & (d.arm == arm)
                   & (d.model == m)].groupby(UNIT)['spearman_true'].mean())
            b = (d[(d.sub_exp == MAIN) & np.isclose(d.alpha, 0.0) & (d.arm == arm)
                   & (d.model == m)].groupby(UNIT)['spearman_true'].mean())
            sh = sorted(set(a.index) & set(b.index))
            diff = np.array([b[s] - a[s] for s in sh])
            w(f'  {arm:<7} {SHORT[m]:<10} no-W={a.mean():.3f}  with-W={b.mean():.3f}  '
              f'median diff={np.median(diff):+.4f}  max|diff|={np.abs(diff).max():.4f}')
    w('  An extra, uninformative channel moves the median by at most 0.008 (RF) and 0.068')
    w('  (CNN); the two alpha = 0 references are interchangeable.')
    w('')
    w(f'-- fit-health flags: {len(deg)} cells where > 50 % of CNN fits never cleared chance --')
    if not deg:
        w('  none anywhere in 8j.')
    for k in sorted(deg, key=lambda t: (t[0], t[1], float(t[2]))):
        w(f'    {k[0]:<24} {k[1]:<7} level={float(k[2]):g}')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'\nwritten: {OUT}')


if __name__ == '__main__':
    main()
