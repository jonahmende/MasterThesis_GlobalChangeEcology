"""Thesis version of the paired CNN-RF contrast figure.

The notebook's own version is s8_cnn_vs_rf.png (plotting cell, "Fig 6"). This
script redraws it for the thesis with two changes to the noise band and the
shared thesis style; the notebook figure is left exactly as it is and nothing
in figures/sus_scrofa/ is touched.

  1 POOLED, NOT MEAN-OF-SDS. The notebook band is _init_sd8: the arithmetic
    mean of the three within-species SDs across the three CNN initialisations.
    This one is the pooled within-species SD, sqrt(mean of those three
    variances). With 3 species all at n = 3 inits the design is balanced, so
    this is the textbook pooled estimator, and it is the larger of the two by
    Jensen's inequality (median ratio 1.13 over the 38 cells, max 1.36) --
    the more conservative band to judge a difference against.

  2 ONE BAND PER ARM. The notebook draws the band for the interpolative arm
    only (`if not _sd.empty and _arm == 'random'`) while both arm curves are
    plotted over it. The extrapolative init-SD is much larger (8a at N = 100:
    0.153 against 0.070), so the extrapolative contrast was being read against
    a band that was too narrow for it. Both bands are drawn here, nested.

No verdict changes under either edit: every cell except 8e / random / patch 8
lies outside its own band. The narrowest surviving call is 8c-fixed / random /
400 m at 1.25x the pooled SD, which is marked as marginal.

Contrasts are formed per species on the SAME drawn data (cnn_mean - rf) and
then averaged, so the data draw cancels -- unchanged from the notebook.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python thesis_fig_paired_contrast.py)
Writes: figures/thesis/fig4_paired_contrast.{pdf,png}
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from _vs_env import FIGURES_DIR
from thesis_figures import (ARM_LABEL, ARM_LS, ARM_FILL,
                            setup_style, panel_label, save, canvas_width,
                            frame, grouped_legend, stagger_xticks, golden, GAP, W_FIG,
                            BASE_PT, INK, INK2)

VERSION = 'v6_bio11'
HEADLINE_RES_MODE = 'constant_footprint'
STEM = 'fig4_paired_contrast'

# the figure's own labels, from the plotting cell (_S8EXP)
S8EXP = {
    '8a':       ('Sample size (training points)', 'log', 'Sample size'),
    '8b':       ('Prevalence', 'linear', 'Prevalence'),
    '8c-fixed': ('Resolution (m)',      'log',    'Resolution'),
    '8e':       ('Patch size (px)',     'log',    'Patch size'),
}
ARMS = ['random', 'extrap']
# The shared arm convention: interpolative solid and filled, extrapolative
# dashed and open. This figure used to draw the interpolative arm dotted,
# which is the style every other figure of the set reserves for the POINTWISE
# process (fig8), so one dash pattern meant two different things.
LS = ARM_LS
FILL = ARM_FILL
C_CONTRAST = '#D55E00'          # same vermillion as every Section-8 contrast
BAND = {'random': dict(fc='0.45', alpha=0.20, ec='none'),
        'extrap': dict(fc='0.75', alpha=0.30, ec='0.55', ls='--', lw=0.6)}


def load():
    d = pd.read_csv(os.path.join(FIGURES_DIR, f'section8_two_regime_{VERSION}.csv'))
    cf = os.path.join(FIGURES_DIR, f'section8c_fixed_truth_{VERSION}.csv')
    if os.path.exists(cf):
        d = pd.concat([d, pd.read_csv(cf)], ignore_index=True)
    if 'res_mode' in d.columns:
        d = d[d.res_mode.isna()
              | d.res_mode.isin([HEADLINE_RES_MODE, 'baseline_100m'])]
    return d.reset_index(drop=True)


def main():
    setup_style()
    d = load()
    CN = 'cnn_mean' if (d.model == 'cnn_mean').any() else 'cnn'
    UNIT = 'species_idx' if 'species_idx' in d.columns else 'rep'
    sexps = [s for s in S8EXP if s in d.sub_exp.unique()]

    # 2x2 rather than the notebook's 1x4: at a 15.9 cm text width four panels
    # in a row leave 3.5 cm each, too narrow for a log axis with labels.
    fig, axes = plt.subplots(2, 2, figsize=(canvas_width(STEM), W_FIG * 0.66),
                             gridspec_kw=dict(wspace=0.22, hspace=0.38))
    axes = axes.ravel()
    marginal = []

    for ax, sexp, letter in zip(axes, sexps, 'abcd'):
        xlabel, xscale, title = S8EXP[sexp]
        g = d[d.sub_exp == sexp]

        # bands first, widest underneath
        bands = {}
        for arm in ARMS:
            sd = (g[(g.arm == arm) & (g.model == 'cnn')]
                  .groupby(['level', UNIT])['spearman_true'].std())
            if sd.empty:
                continue
            bands[arm] = np.sqrt((sd ** 2).groupby(level=0).mean())
        for arm in sorted(bands, key=lambda a: -bands[a].mean()):
            b = bands[arm]
            x = b.index.astype(float).values
            o = np.argsort(x)
            ax.fill_between(x[o], -b.values[o], b.values[o], zorder=0,
                            label='_nolegend_', **BAND[arm])

        # the paired contrast, per species then averaged
        for arm in ARMS:
            ga = g[g.arm == arm]
            hi = ga[ga.model == CN].groupby(['level', UNIT])['spearman_true'].mean()
            lo = ga[ga.model == 'rf'].groupby(['level', UNIT])['spearman_true'].mean()
            c = (hi - lo).dropna()
            if c.empty:
                continue
            mu = c.groupby(level='level').mean()
            ax.plot(c.index.get_level_values('level').astype(float), c.values,
                    '^', color=C_CONTRAST, ms=1.8, alpha=0.22, ls='none',
                    fillstyle=FILL[arm], zorder=2)
            ax.plot(mu.index.astype(float), mu.values, ls=LS[arm], marker='^',
                    color=C_CONTRAST, ms=3.4, lw=1.2, fillstyle=FILL[arm],
                    mew=0.8, label='_nolegend_', zorder=3)
            # ring the cells that do NOT clear their own band
            if arm in bands:
                for lv, v in mu.items():
                    s = float(bands[arm].get(lv, np.nan))
                    if np.isfinite(s) and abs(v) <= s:
                        # ms and zorder match fig7, where the ring has to sit
                        # on top of the unlearned flag; one symbol, one size
                        # across both contrast figures.
                        ax.plot([float(lv)], [v], 'o', ms=11, mfc='none',
                                mec='0.25', mew=1.1, zorder=7)
                        marginal.append((sexp, arm, float(lv), v, s, 'INSIDE'))
                    elif np.isfinite(s) and abs(v) <= 1.5 * s:
                        marginal.append((sexp, arm, float(lv), v, s, 'marginal'))

        ax.axhline(0, color=INK, lw=0.7, ls='-', alpha=0.6, zorder=1)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        # a tick at every measured level, linear axis included (see fig5)
        ax.set_xticks(sorted(g.level.unique().astype(float)))
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda v, _: f'{v:g}'))
        if xscale == 'log':
            ax.xaxis.set_minor_formatter(mtick.NullFormatter())
        lv = np.array(sorted(g.level.unique().astype(float)))
        span = lv.max() - lv.min()
        if xscale == 'linear' and len(lv) > 1 and np.diff(lv).min() / span < 0.10:
            ax.tick_params(axis='x', labelsize=BASE_PT - 2.0)
            stagger_xticks(ax)
        ax.grid(axis='y', alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
        frame(ax)
        golden(ax)
        ax.set_title(f'({letter}) {title}', loc='left', fontsize=BASE_PT,
                     color=INK2, pad=3)

    for ax in (axes[0], axes[2]):
        ax.set_ylabel('$\\Delta\\rho$ (CNN $-$ RF)')
    # The ring is explained in the legend, not only in the caption, so that
    # fig4 and fig7 account for the same symbol in the same place. The red
    # cross of fig7 is NOT added: no pointwise cell meets the > 50 % fit-health
    # rule (the worst is 0.444 at 8a / extrapolative / N = 500), so the symbol
    # is nowhere on this page and a legend entry would send the reader hunting.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    contrast = [Line2D([], [], color=C_CONTRAST, marker='^', ls='-', ms=4.0,
                       lw=1.2, label='CNN $-$ RF')]
    arms = [Line2D([], [], color='0.45', marker='o', ms=4.0, lw=1.2,
                   ls=LS[a_], fillstyle=FILL[a_], mew=0.8, label=ARM_LABEL[a_])
            for a_ in ARMS]
    refs = [Patch(label=f'$\\pm$ init-SD, {ARM_LABEL[a_]}',
                  **{k: v for k, v in BAND[a_].items()})
            for a_ in ARMS]
    refs.append(Line2D([], [], ls='none', marker='o', ms=8, mfc='none',
                       mec='0.25', mew=1.1,
                       label='inside the band (inconclusive)'))
    grouped_legend(fig, axes, [('Contrast', contrast), ('Arm', arms),
                               ('Reference and flags', refs)])
    save(fig, STEM)

    print('\n-- cells inside their own band, or within 1.5x of it --')
    for sexp, arm, lv, v, s, kind in marginal:
        print(f'  {kind:<9} {sexp:<9} {arm:<7} level={lv:>8g}  '
              f'delta={v:+.4f}  pooled SD={s:.4f}  ratio={abs(v)/s:.2f}')


if __name__ == '__main__':
    main()
