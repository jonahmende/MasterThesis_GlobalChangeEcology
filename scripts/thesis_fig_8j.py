"""Thesis figures for 8j -- the species ensemble across the alpha continuum.

MAIN TEXT
  fig9_8j_blend      Spearman vs alpha, 2 x 2: W source (row) x arm (column).
                     All four models plus the patch-limited ceiling. This is the
                     figure the discussion rests on -- the RFs' collapse against
                     their plateau, the CNN's deficit at alpha = 0, and the share
                     of the ceiling each model reaches are all absolute
                     quantities, and a contrast cannot show them.

APPENDIX
  fig13_8j_blend_auc the same 2 x 2 grid for AUC-ROC, with the oracle-AUC
                     ceiling. AUC orders the models exactly as Spearman does
                     here; the lesson about the two metrics is drawn in the
                     prevalence experiment, so this is support, not argument.
  fig10_8j_corrlen   ABSOLUTE Spearman of CNN and RF-patch at alpha = 1 across
                     correlation length (150-400 m), one arm per panel. It used
                     to draw cfgGain on this axis; a contrast cannot say WHICH
                     of the two curves moved, which is the whole question here.
                     Its old alpha panel was byte for byte fig12's random-field
                     half, so it was dropped rather than duplicated.
  fig11_8j_oracle_val  how much of the CNN deficit is model selection.
  fig12_8j_wsource   cfgGain across alpha for both W sources, on fig9's grid
                     (W source by row, arm by column), so panel (c) here sits
                     under panel (c) of fig9.

cfgGain is the PRE-REGISTERED PRIMARY ENDPOINT and is reported with value and CI
in the text; its figure sits in the appendix because it cancels the very thing
the main figure has to show -- where the gain comes from. Under the forest map
the composition-only models reach rho = 0.14-0.34 at alpha = 1, because that W
field's composition component is coupled to the environmental axes
(Spearman(P_w, PC1) = 0.553 against 0.007 for the random field;
configurational_orthogonality.log). cfgGain = Delta(alpha) - Delta(0) subtracts
that leak away and shows only a smaller number, not its cause.

cfgGain is the contrast this section is built on:

    cfgGain(alpha) = Delta(alpha) - Delta(0),   Delta = CNN - RF-patch

formed PER SPECIES against that species' own alpha = 0 run, so differences in
intrinsic difficulty cancel. Delta(0) is the architecture penalty: what the CNN
gives up to the patch-summary RF when there is no configuration to find.

THE CEILING IS A COMPONENT BOUND BELOW alpha = 1. ceiling_config_patch bounds
recovery of S_config. The scoring target is S = S_point^(1-alpha) * S_config^alpha,
so only at alpha = 1 is the target S_config itself and the ceiling the ceiling of
the whole target. Below that it bounds one factor of the product, a model can
sit above it through the pointwise part without contradiction, and it is drawn
faint to say so. At alpha = 0 it does not exist at all (those runs go through
_s8_run_level, which never computes it).

THE 300 m CORRELATION LENGTH COMES FROM THE ALPHA SWEEP. corr_len = 300 m is the
alpha-sweep default, so its alpha = 1 point IS that condition -- same species,
same W, same seed. The sweep deliberately does not refit it; it is spliced in
here and ringed in the figure.

LEGIBILITY. Both cfgGain figures used to overlay two groups in one frame: 16
pale per-species strands, two translucent bands and two medians, with nothing
but a 2 pt marker to say which group a strand belonged to. Each panel now
carries ONE group, and inside a panel each species gets a small constant x
offset so its markers do not stack on the others'. The offset is in x only --
every y value and every median is untouched.

THE TWO W FIELDS DIFFER IN DENSITY AS WELL AS IN REALISM: mu_P = 0.300 for the
random field against 0.411 for the forest map. Any difference between the rows
is therefore not purely "synthetic vs real". This belongs in the caption.

NO CELL IS FLAGGED. The fit-health rule (> 50 % of a cell's CNN fits below the
prevalence-aware chance floor) fires nowhere in 8j -- the worst cell is 10 of 24
on the random field at alpha = 1. The flag symbols are therefore absent from
these figures and their legends; the raw per-cell shares are printed and go in
the appendix table, and the caption should say the criterion was applied and met
nowhere.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python thesis_fig_8j.py)
Writes: figures/thesis/fig9_8j_blend.{pdf,png}
        figures/thesis/fig13_8j_blend_auc.{pdf,png}
        figures/thesis/fig10_8j_corrlen.{pdf,png}
        figures/thesis/fig11_8j_oracle_val.{pdf,png}
        figures/thesis/fig12_8j_wsource.{pdf,png}
Every number behind them is printed by results_dump_8j.py -> results_8j.txt.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from _vs_env import FIGURES_DIR
from thesis_figures import (setup_style, save, canvas_width, frame,
                            panel_legend, shared_legend, grouped_legend,
                            chance_line, chance_handle, chance_level,
                            golden, GAP, W_FIG, BASE_PT, INK,
                            INK2, ARM_LABEL, ARM_LS, ARM_FILL, CEIL_COLOR,
                            CEIL_MARKER)
from thesis_fig_panels_8i import (auc_floor, COL, MK, SHORT, LS, FILL, ARMS,
                                  no_skill, no_skill_levels, flag)

VERSION = 'v6_bio11'
MAIN = '8j_alpha_grf'
CEIL_C = '#8a8a84'
# the contrast is CNN - RF-patch, so it takes the CNN's vermillion, as in
# fig4/fig7; green is RF-patch's own colour and would read as that model
C_CONTRAST = '#D55E00'
# The shaded area in fig12 is the 95 % bootstrap CI of the MEDIAN over
# species -- not the init-SD noise band of fig4/fig7, which is a different
# quantity. One band per panel now (see LEGIBILITY above), so the two-band
# convention that told two overlaid arms apart is gone; see BAND below.


def load():
    d = pd.read_csv(os.path.join(FIGURES_DIR,
                                 f'section8j_species_ensemble_{VERSION}.csv'))
    return d


def degenerate_8j(d):
    c = d[(d.model == 'cnn') & d.val_auc_best.notna()].copy()
    if c.empty:
        return set()
    c['degen'] = c.val_auc_best < auc_floor(c.n_vl, c.real_prev_vl)
    h = c.groupby(['sub_exp', 'arm', 'level'])['degen'].mean()
    return set(h[h > 0.5].index)


def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    r = np.random.default_rng(seed)
    med = np.median(x[r.integers(0, len(x), (n_boot, len(x)))], axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def cfg_table(d, sub_exp, keycol, gap0_from=MAIN):
    """Per-species cfgGain = (CNN - RF-patch)(key) - (CNN - RF-patch)(alpha=0)."""
    CN = 'cnn_mean' if (d.model == 'cnn_mean').any() else 'cnn'
    out = []
    for sp in sorted(d.species_idx.unique()):
        for arm in ARMS:
            ref = d[(d.species_idx == sp) & (d.arm == arm)
                    & (d.sub_exp == gap0_from) & np.isclose(d.alpha, 0.0)]
            cn0 = ref[ref.model == CN]['spearman_true'].mean()
            rp0 = ref[ref.model == 'rf_patch_summary']['spearman_true'].mean()
            if not np.isfinite(cn0) or not np.isfinite(rp0):
                continue
            gap0 = cn0 - rp0
            g = d[(d.species_idx == sp) & (d.arm == arm)
                  & (d.sub_exp == sub_exp)]
            cn = g[g.model == CN].groupby(keycol)['spearman_true'].mean()
            rp = g[g.model == 'rf_patch_summary'].groupby(keycol)['spearman_true'].mean()
            for k in cn.index:
                if k not in rp.index:
                    continue
                out.append(dict(species_idx=sp, arm=arm, key=float(k),
                                cfg_gain=float((cn[k] - rp[k]) - gap0),
                                cnn=float(cn[k]), rf_patch=float(rp[k])))
    return pd.DataFrame(out)


def med_ci(tab, arm, keys):
    m, lo, hi, n = [], [], [], []
    for k in keys:
        v = tab.loc[(tab.arm == arm) & (tab.key == k), 'cfg_gain'].dropna().values
        m.append(np.median(v) if len(v) else np.nan)
        a, b = boot_ci(v)
        lo.append(a); hi.append(b); n.append(len(v))
    return np.array(m), np.array(lo), np.array(hi), n


# =============================================================================
# W source (row) x arm (column). One grid, one metric, so all four panels are
# directly comparable; the y axis is shared across all four for the same reason.
WSRC = [(MAIN, 'Gaussian random field'),
        ('8j_alpha_landuse', 'forest map (ESA tree cover)')]


def _alpha_grid(d, col, ylab, stem, ceiling):
    """The 2 x 2 absolute-skill grid, for one metric.

    ceiling: 'patch'  -> ceiling_config_patch, the patch-limited rank ceiling
             'oracle' -> auc_oracle, the label-noise ceiling
    """
    CN = 'cnn_mean' if (d.model == 'cnn_mean').any() else 'cnn'
    UNIT = 'species_idx'
    order = ['rf', 'rf_patch_summary', 'rf_oracle_metrics', CN]

    fig, axes = plt.subplots(2, 2, figsize=(canvas_width(stem), W_FIG * 0.74),
                             gridspec_kw=dict(wspace=GAP['s'], hspace=GAP['m']),
                             sharey=True)
    drew_ring = False
    ceil_label = None
    for ri, (sexp, wlab) in enumerate(WSRC):
        g = d[d.sub_exp == sexp]
        alphas = sorted(g.alpha.dropna().unique())
        for cj, arm in enumerate(ARMS):
            ax = axes[ri, cj]
            for m in order:
                s_ = g[(g.model == m) & (g.arm == arm)]
                if s_.empty:
                    continue
                per = s_.groupby(['alpha', UNIT])[col].mean()
                mu = per.groupby(level='alpha').mean()
                ax.plot(per.index.get_level_values('alpha').astype(float),
                        per.values, MK[m], color=COL[m], ms=1.6, alpha=0.20,
                        ls='none', fillstyle=FILL[arm],
                        mew=0.0 if FILL[arm] == 'full' else 0.4, zorder=1)
                ax.plot(mu.index.astype(float), mu.values, ls=LS[arm],
                        marker=MK[m], color=COL[m], ms=3.4, lw=1.2,
                        fillstyle=FILL[arm], mew=0.8, zorder=3,
                        label='_nolegend_')
            # THE CEILING. ceiling_config_patch bounds recovery of S_config, and
            # the scoring target is S_point^(1-alpha) * S_config^alpha -- so only
            # at alpha = 1 does it bound the whole target. Below that it bounds
            # one FACTOR, and a model may sit above it through the pointwise part
            # without contradiction. It is drawn dotted and unmarkered to read as
            # a reference rather than as a fifth curve, and it starts at
            # alpha = 0.25 because the alpha = 0 runs go through _s8_run_level,
            # which never computes it (NaN there, not zero). The caption carries
            # the caveat; the alpha = 1 values are printed below.
            # The chance level, as ONE line per panel rather than a ring on
            # each no-skill point: n_te and the test prevalence are fixed
            # across the blend, so it is a single number per cell.
            if col == 'roc_auc':
                chance_line(ax, chance_level(
                    auc_floor(g[g.arm == arm].n_te.values,
                              g[g.arm == arm].real_prev_te.values)))
                drew_ring = True

            if ceiling == 'patch':
                cc = (g[g.arm == arm].groupby('alpha')['ceiling_config_patch']
                      .mean().dropna())
                lab = 'patch-limited ceiling'
            else:
                cc = (g[(g.arm == arm) & (g.model == 'rf')]
                      .groupby('alpha')['auc_oracle'].mean().dropna())
                lab = 'label-noise ceiling'
            if not cc.empty:
                ceil_label = lab
                ax.plot(cc.index.astype(float), cc.values, ls=LS[arm], lw=1.0,
                        marker=CEIL_MARKER[lab], ms=4.2, fillstyle=FILL[arm],
                        mew=0.7, color=CEIL_COLOR, zorder=2,
                        label='_nolegend_')

            ax.set_xticks(alphas)
            ax.grid(axis='y', alpha=0.25, lw=0.4)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
            if ri == 1:
                ax.set_xlabel('$\\alpha$ (weight of the configurational DGP)')
            if cj == 0:
                ax.set_ylabel(ylab)
            frame(ax); golden(ax)
            lab_arm = ARM_LABEL[arm]
            ax.set_title(f'({"abcd"[ri * 2 + cj]}) {wlab}, {lab_arm} arm',
                         loc='left', fontsize=BASE_PT, color=INK, pad=3)

    # The fit-health cross is NOT legended: no 8j cell meets the > 50 % rule,
    # so the symbol is nowhere on the page.
    from matplotlib.lines import Line2D
    models = [Line2D([], [], color=COL[m], marker=MK[m], ls='-', ms=4.0,
                     lw=1.2, label=SHORT[m]) for m in order]
    arms = [Line2D([], [], color='0.45', marker='o', ms=4.0, lw=1.2,
                   ls=LS[a_], fillstyle=FILL[a_], mew=0.8,
                   label=ARM_LABEL[a_])
            for a_ in ARMS]
    refs = []
    if ceil_label:
        refs.append(Line2D([], [], color=CEIL_COLOR, ms=4.6, lw=1.0, ls='-',
                           marker=CEIL_MARKER[ceil_label], label=ceil_label))
    if drew_ring:
        refs.append(chance_handle())
    grouped_legend(fig, axes, [('Model', models), ('Arm', arms),
                               ('Reference and flags', refs)])
    save(fig, stem)


def fig_blend(d):
    _alpha_grid(d, 'spearman_true', 'Spearman $\\rho$',
                'fig9_8j_blend', ceiling='patch')


def fig_blend_auc(d):
    _alpha_grid(d, 'roc_auc', 'AUC-ROC', 'fig13_8j_blend_auc',
                ceiling='oracle')


# -----------------------------------------------------------------------------
# TWO DEVICES AGAINST THE HAIRBALL, used by both cfgGain figures.
#
# 1 ONE GROUP PER PANEL. Overlaying two groups (two arms, or two W sources) put
#   16 pale strands, two translucent bands and two medians in one frame; the
#   strands crossed constantly and nothing but a 2 pt marker said which group a
#   strand belonged to. Each panel now carries ONE group: 8 strands, one band,
#   one median. The comparison moves from within a panel to between panels, and
#   both figures use the same grid as fig9 (W source x arm) where they can, so
#   the reader learns one layout.
#
# 2 A PER-SPECIES X DODGE. Even inside one group the eight strands piled up on
#   the same x positions, so at every node eight markers sat on one another.
#   Each species now gets a small constant horizontal offset, which separates
#   the markers without moving any value: the offset is in x only, and the
#   y readings and the median line are untouched.
DODGE_FRAC = 0.055        # of the full x range, spread across all species


def _dodge(keys, i, n):
    """x positions for species i of n: a constant small offset, in x units."""
    keys = np.asarray(keys, float)
    span = keys.max() - keys.min()
    if n < 2 or span <= 0:
        return keys
    off = (i - (n - 1) / 2) / (n - 1) * DODGE_FRAC * span
    return keys + off


def _cfg_panel(ax, tab, keys, arm, colour, ls, marker, band_kw, legend=True):
    """One group in one panel: CI band, dodged species strands, median.

    legend=False labels nothing, for figures whose legend is built by hand
    because shared_legend de-duplicates by label and would collapse two rows
    that carry the same wording into one entry.
    """
    m, lo, hi, n = med_ci(tab, arm, keys)
    ax.fill_between(keys, lo, hi, zorder=0, **band_kw,
                    label='95\\,\\% CI of the median' if legend else '_nolegend_')
    sps = sorted(tab.species_idx.unique())
    for i, sp in enumerate(sps):
        v = (tab[(tab.arm == arm) & (tab.species_idx == sp)]
             .set_index('key')['cfg_gain'].reindex(keys))
        ax.plot(_dodge(keys, i, len(sps)), v.values, color=colour, lw=0.6,
                alpha=0.45, ls=ls, marker=marker, ms=2.2, mew=0.5, zorder=1,
                label='individual species' if (legend and i == 0)
                else '_nolegend_')
    ax.plot(keys, m, ls=ls, marker=marker, ms=4.0, lw=1.4, color=colour,
            mew=0.9, zorder=3,
            label=f'median over {max(n)} species' if legend else '_nolegend_')
    return m, lo, hi, n


def _cfg_axis(ax, keys, xlab, ylab):
    ax.axhline(0, color=INK, lw=0.7, alpha=0.6, zorder=2)
    ax.set_xlabel(xlab)
    ax.set_xticks(keys)
    if ylab:
        ax.set_ylabel(ylab)
    ax.grid(axis='y', alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
    frame(ax); golden(ax)


YLAB_CFG = 'cfgGain $=\\Delta(\\alpha)-\\Delta(0)$'
# One band style, because there is now one band per panel. The wide grey of the
# old two-band convention is no longer needed to tell two bands apart.
BAND = dict(fc='0.55', alpha=0.20, ec='none')


CORR_KEYS = [150.0, 200.0, 300.0, 400.0]
CORR_MODELS = ['cnn_mean', 'rf_patch_summary']


def corr_frame(d):
    """The correlation-length sweep at alpha = 1, with the 300 m point spliced in.

    The sweep itself ran 150 / 200 / 400 m. 300 m is the alpha sweep's own
    default, so 8j_alpha_grf at alpha = 1 IS that condition -- same species,
    same W construction, same seed -- and it is not refitted. It is spliced in
    here and ringed in the figure, so the reader can see which point came from
    a different sub-experiment.
    """
    sw = d[d.sub_exp == '8j_corrlen_grf_alpha1'].copy()
    a1 = d[(d.sub_exp == MAIN) & np.isclose(d.alpha, 1.0)].copy()
    assert sorted(sw.corr_len_m.unique()) == [150.0, 200.0, 400.0], \
        f'unexpected sweep levels: {sorted(sw.corr_len_m.unique())}'
    assert sorted(a1.corr_len_m.unique()) == [300.0], \
        f'the alpha sweep is not at 300 m: {sorted(a1.corr_len_m.unique())}'
    return pd.concat([sw, a1], ignore_index=True)


def fig_corrlen(d):
    """ABSOLUTE skill against correlation length, at alpha = 1, one arm per panel.

    This replaces the cfgGain version of the same axis. cfgGain answers whether
    the CNN gained on RF-patch relative to its own alpha = 0; on this axis the
    question is the plainer one -- does a coarser or finer W change how much
    either model recovers -- and a contrast cannot answer it, because it cannot
    say which of the two curves moved. Both models here are patch-limited and
    both sit at alpha = 1, so the two curves are directly comparable.

    ITS FORMER ALPHA PANEL IS GONE. It drew cfg_table(d, MAIN, 'alpha') -- byte
    for byte the 'random field' half of fig12 -- so the alpha story lives in
    fig12 alone, for both W sources.
    """
    stem = 'fig10_8j_corrlen'
    UNIT = 'species_idx'
    g = corr_frame(d)
    keys = CORR_KEYS

    fig, axes = plt.subplots(1, 2, figsize=(canvas_width(stem), W_FIG * 0.40),
                             gridspec_kw=dict(wspace=GAP['s']), sharey=True)
    for k, (ax, arm) in enumerate(zip(axes, ARMS)):
        for m in CORR_MODELS:
            s_ = g[(g.model == m) & (g.arm == arm)]
            per = (s_.groupby(['corr_len_m', UNIT])['spearman_true'].mean()
                   .unstack(UNIT).reindex(keys))
            mu = per.mean(axis=1)
            # pale individual species, dodged in x so their markers do not
            # stack -- the offset is in x only, every value is untouched
            sps = list(per.columns)
            for i, sp in enumerate(sps):
                ax.plot(_dodge(keys, i, len(sps)), per[sp].values, color=COL[m],
                        lw=0.6, alpha=0.40, ls=LS[arm], marker=MK[m], ms=2.2,
                        fillstyle=FILL[arm], mew=0.5, zorder=1,
                        label='_nolegend_')
            ax.plot(keys, mu.values, ls=LS[arm], marker=MK[m], ms=4.0, lw=1.4,
                    color=COL[m], fillstyle=FILL[arm], mew=0.9, zorder=3,
                    label=f'{SHORT[m]} (mean over {per.shape[1]} species)')
            # NO RING ON THE 300 m POINT. An open grey ring means "inside the
            # init-SD band, inconclusive" in fig4 and fig7; using it here for
            # "spliced from the alpha sweep" would give one symbol two
            # meanings. That provenance belongs in the caption.
        _cfg_axis(ax, keys, 'Correlation length (m)',
                  'Spearman $\\rho$' if k == 0 else None)
        lab = ARM_LABEL[arm]
        ax.set_title(f'({"ab"[k]}) {lab} arm', loc='left', fontsize=BASE_PT,
                     color=INK, pad=3)
    # The two generic marks go last and in neutral grey: both apply to BOTH
    # models, so neither may wear a model colour. Labelling them inside the loop
    # put them between the two model entries in plot order.
    from matplotlib.lines import Line2D
    marks = [Line2D([], [], color='0.45', lw=0.6, alpha=0.55, marker='^',
                    ms=2.2, mew=0.5, ls='-', label='individual species'),
             ]
    shared_legend(fig, axes, ncol=4, extra=marks)
    save(fig, stem)

    print('\n-- alpha = 1 by correlation length: spearman_true, mean over '
          'species --')
    for arm in ARMS:
        for m in CORR_MODELS:
            s_ = g[(g.model == m) & (g.arm == arm)]
            per = (s_.groupby(['corr_len_m', UNIT])['spearman_true'].mean()
                   .unstack(UNIT).reindex(keys))
            row = '  '.join(f'{c:g}={per.loc[c].mean():+.4f}' for c in keys)
            print(f'  {arm:<7} {SHORT[m]:<10} {row}')
        cc = '  '.join(
            f'{c:g}={g[(g.arm == arm) & np.isclose(g.corr_len_m, c)].ceiling_config_patch.mean():.4f}'
            for c in keys)
        print(f'  {arm:<7} {"ceiling":<10} {cc}')


def fig_cfggain_numbers(d):
    """cfgGain figures are fig12; these are the numbers its axis no longer holds."""
    T_grf = cfg_table(d, MAIN, 'alpha')
    T_lu = cfg_table(d, '8j_alpha_landuse', 'alpha')
    T_cl = cfg_table(d, '8j_corrlen_grf_alpha1', 'corr_len_m')
    T300 = T_grf[T_grf.key == 1.0].copy()
    T300['key'] = 300.0
    T_cl = pd.concat([T_cl, T300], ignore_index=True)

    print('\n-- cfgGain, median over species (alpha sweep, GRF) --')
    for arm in ARMS:
        m, lo, hi, n = med_ci(T_grf, arm, sorted(T_grf.key.unique()))
        for k, a, b, c in zip(sorted(T_grf.key.unique()), m, lo, hi):
            print(f'  {arm:<7} alpha={k:<5g} median={a:+.4f}  CI [{b:+.4f}, {c:+.4f}]')
    print('\n-- cfgGain at alpha = 1 by W source --')
    for T, nm in [(T_grf, 'random field'), (T_lu, 'forest map')]:
        for arm in ARMS:
            v = T.loc[(T.arm == arm) & (T.key == 1.0), 'cfg_gain'].dropna().values
            lo, hi = boot_ci(v)
            print(f'  {nm:<13} {arm:<7} median={np.median(v):+.4f}  '
                  f'CI [{lo:+.4f}, {hi:+.4f}]  n={len(v)}')
    print('\n-- cfgGain at alpha = 1 by correlation length (NOT drawn any more; '
          'fig10 now shows absolute skill) --')
    for arm in ARMS:
        ks = sorted(T_cl.key.unique())
        m, lo, hi, n = med_ci(T_cl, arm, ks)
        for k, a, b, c in zip(ks, m, lo, hi):
            print(f'  {arm:<7} corr_len={k:<6g} median={a:+.4f}  CI [{b:+.4f}, {c:+.4f}]')


def fig_wsource(d):
    """cfgGain across alpha for BOTH W sources, on fig9's grid.

    The forest-map sweep is what keeps the configurational result from being an
    artefact of the Gaussian random field: same species, same alphas, same
    machinery, with W taken from the ESA tree-cover class instead. It ran at
    every alpha, so the comparison is drawn over the whole continuum rather
    than at alpha = 1 alone.

    THE GRID IS fig9's GRID: W source by row, arm by column. Overlaying the two
    sources put 16 strands and two bands in one frame; on this grid each panel
    carries one source in one arm, and panel (c) of this figure sits under panel
    (c) of fig9 -- the same cell, the absolute curves above, their contrast
    below. The y axis is shared across all four so the rows stay comparable.
    """
    stem = 'fig12_8j_wsource'
    T = {'random field': cfg_table(d, MAIN, 'alpha'),
         'forest map': cfg_table(d, '8j_alpha_landuse', 'alpha',
                                 gap0_from='8j_alpha_landuse')}
    # W source by colour, kept from the overlaid version: the colour is no
    # longer needed to separate the sources now that they are in separate rows,
    # but it labels the row at a glance. #D55E00 is the configurational DGP's
    # colour in the pipeline diagram; #882255 (Tol muted wine) appears nowhere
    # else in the set, so neither reads as one of the four model colours.
    style = {'random field': dict(color=C_CONTRAST, marker='^'),
             'forest map': dict(color='#882255', marker='s')}
    keys = sorted(T['random field'].key.unique())

    fig, axes = plt.subplots(2, 2, figsize=(canvas_width(stem), W_FIG * 0.74),
                             gridspec_kw=dict(wspace=GAP['s'], hspace=GAP['m']),
                             sharey=True)
    for ri, (nm, tab) in enumerate(T.items()):
        st = style[nm]
        for cj, arm in enumerate(ARMS):
            ax = axes[ri, cj]
            band = dict(fc=st['color'], alpha=0.16, ec='none')
            _cfg_panel(ax, tab, keys, arm, st['color'], LS[arm], st['marker'],
                       band, legend=False)
            _cfg_axis(ax, keys, '$\\alpha$ (weight of the configurational DGP)'
                      if ri == 1 else '', YLAB_CFG if cj == 0 else None)
            if ri == 0:
                ax.set_xlabel('')
            lab = ARM_LABEL[arm]
            ax.set_title(f'({"abcd"[ri * 2 + cj]}) {nm}, {lab} arm', loc='left',
                         fontsize=BASE_PT, color=INK, pad=3)
    # One entry pair per ROW, built by hand: colour and marker identify the row,
    # and shared_legend de-duplicates by label, so the two rows' identical
    # wording would collapse into one entry if the panels labelled themselves.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = []
    for nm, st in style.items():
        handles += [Line2D([], [], color=st['color'], marker=st['marker'],
                           ms=4.0, lw=1.4, mew=0.9, ls='-',
                           label=f'{nm}: median over 8 species'),
                    Patch(fc=st['color'], alpha=0.16, ec='none',
                          label=f'{nm}: 95\\,\\% CI of the median')]
    handles.append(Line2D([], [], color='0.45', lw=0.6, alpha=0.45, marker='^',
                          ms=2.2, mew=0.5, label='individual species'))
    shared_legend(fig, axes, ncol=2, extra=handles)
    save(fig, stem)

    print('\n-- cfgGain by W source, median over species (95 % CI) --')
    for nm, tab in T.items():
        for arm in ARMS:
            m, lo, hi, n = med_ci(tab, arm, keys)
            for a, mm, l, h in zip(keys, m, lo, hi):
                print(f'  {nm:<13} {arm:<7} alpha={a:<5g} median={mm:+.4f}  '
                      f'CI [{l:+.4f}, {h:+.4f}]')


def fig_oracle_val(d):
    """Is the CNN model-selected on the same problem it is scored on?

    In the primary design the validation fold sits BETWEEN train and test in
    PC1, so it is necessarily closer to train (measured D(val,train) = 1.30
    against D(test,train) = 1.99). Early stopping therefore picks the epoch
    that is best under a shift of 1.30 and the model is then scored at 1.99,
    while the RF does no model selection and carries no such penalty. That
    asymmetry sits in exactly the arm where the CNN does worst.

    The second run draws val from the TEST band IN THE EXTRAPOLATIVE ARM, so
    selection and scoring see the same shift. It assumes labels in the target
    domain, which real transfer
    does not have: it is an UPPER BOUND, never the headline. Paired within
    species against its own primary counterpart.
    """
    CN = 'cnn_mean' if (d.model == 'cnn_mean').any() else 'cnn'
    stem = 'fig11_8j_oracle_val'
    pairs = [('$\\alpha$=1\n(configurational)', MAIN, 1.0,
              '8j_oracleval_alpha1', 1.0),
             ('$\\alpha$=0\n(pointwise)', '8j_pointwise_baseline', 0.0,
              '8j_oracleval_pointwise', 0.0)]

    def cnn_by_species(sub_exp, alpha, arm):
        g = d[(d.sub_exp == sub_exp) & (d.arm == arm) & (d.model == CN)]
        if 'alpha' in g.columns and g.alpha.notna().any():
            g = g[np.isclose(g.alpha.fillna(-999), alpha)]
        return g.groupby('species_idx')['spearman_true'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(canvas_width(stem), W_FIG * 0.34),
                             gridspec_kw=dict(wspace=GAP['m']), sharey=True)
    out = []
    for k, (ax, arm) in enumerate(zip(axes, ARMS)):
        xt, lbl = [], []
        for xi, (nm, se_p, a_p, se_o, a_o) in enumerate(pairs):
            pri, ora = cnn_by_species(se_p, a_p, arm), cnn_by_species(se_o, a_o, arm)
            sh = sorted(set(pri.index) & set(ora.index))
            if not sh:
                continue
            for sp in sh:
                ax.plot([xi - 0.16, xi + 0.16], [pri[sp], ora[sp]], ls='-',
                        color='0.70', lw=0.6, zorder=1)
            ax.plot([xi - 0.16] * len(sh), [pri[s] for s in sh], 'o',
                    color=COL['rf'], ms=3.4, mew=0.8, ls='none', zorder=3,
                    label='primary run' if xi == 0 else None)
            ax.plot([xi + 0.16] * len(sh), [ora[s] for s in sh], '^',
                    color=COL['cnn_mean'], ms=3.4, mew=0.8, ls='none', zorder=3,
                    label=('second run (extrapolative: validation from the '
                           'test band)') if xi == 0 else None)
            # THE MEAN, not the median. With eight species the median lands
            # between two of them and moves in steps; the mean is the quantity
            # the appendix table reports (change_mean in blend_oracleval.csv).
            ch = np.array([ora[s] - pri[s] for s in sh])
            dmed = float(ch.mean())
            lo, hi = boot_ci(ch)
            ax.annotate(f'{dmed:+.2f}', xy=(xi, 0.96),
                        xycoords=('data', 'axes fraction'), ha='center',
                        fontsize=BASE_PT - 1.5, color=INK)
            xt.append(xi); lbl.append(nm)
            out.append((arm, 'alpha=1 (configurational)' if xi == 0 else 'alpha=0 (pointwise)', dmed, lo, hi, len(sh)))
        ax.set_xticks(xt); ax.set_xticklabels(lbl, fontsize=BASE_PT - 1.5)
        ax.set_xlim(-0.55, len(pairs) - 0.45)
        ax.grid(axis='y', alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
        frame(ax); golden(ax)
        ax.set_title(f'({"ab"[k]}) {ARM_LABEL[arm]} arm', loc='left', fontsize=BASE_PT,
                     color=INK, pad=3)
    axes[0].set_ylabel('CNN Spearman $\\rho$')
    shared_legend(fig, axes, ncol=2)
    save(fig, stem)

    print('\n-- how much of the CNN deficit is model selection, not capability? --')
    print('   (MEAN over species of oracle-val minus primary; the CI is a '
          'bootstrap of the median)')
    for arm, nm, dmed, lo, hi, n in out:
        print(f'  {arm:<7} {nm:<26} {dmed:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  n={n}')


def main():
    setup_style()
    d = load()
    deg = degenerate_8j(d)
    print(f'8j: {len(d)} rows, {d.species_idx.nunique()} species, '
          f'sub_exp {sorted(d.sub_exp.unique())}')
    print(f'CNN-unfit cells: {len(deg)}')
    for k in sorted(deg, key=lambda t: (t[0], t[1], float(t[2]))):
        print(f'   {k[0]:<24} {k[1]:<7} level={float(k[2]):g}')
    fig_blend(d)           # main text
    fig_blend_auc(d)       # appendix
    fig_corrlen(d)
    fig_oracle_val(d)
    fig_wsource(d)
    fig_cfggain_numbers(d)

    # 8j_pointwise_baseline is not used by cfgGain (see the module docstring);
    # it is the no-W control. Report how far it sits from the alpha = 0 run of
    # the configurational machinery, which has the same truth but carries the
    # W channel as an extra, uninformative predictor.
    print('\n-- control: the two alpha = 0 references (same truth, W channel '
          'present vs absent) --')
    CN = 'cnn_mean' if (d.model == 'cnn_mean').any() else 'cnn'
    for arm in ARMS:
        for m in ['rf', CN]:
            a = (d[(d.sub_exp == '8j_pointwise_baseline') & (d.arm == arm)
                   & (d.model == m)].groupby('species_idx')['spearman_true'].mean())
            b = (d[(d.sub_exp == MAIN) & np.isclose(d.alpha, 0.0)
                   & (d.arm == arm) & (d.model == m)]
                 .groupby('species_idx')['spearman_true'].mean())
            sh = sorted(set(a.index) & set(b.index))
            diff = np.array([b[s] - a[s] for s in sh])
            print(f'  {arm:<7} {m:<9} no-W={a.mean():.3f}  with-W={b.mean():.3f}  '
                  f'median diff={np.median(diff):+.4f}  max|diff|={np.abs(diff).max():.4f}')


if __name__ == '__main__':
    main()
