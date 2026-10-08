"""Paired CNN-minus-baseline contrast for the four 8i axes (alpha = 1).

The counterpart of fig4_paired_contrast, which does this for the pointwise
regime. Same construction: the contrast is formed per species on the SAME drawn
data and then averaged, so the data draw cancels, and it is judged against the
pooled within-species SD across the three CNN initialisations -- one band per
arm, never the interpolative band alone.

THE BASELINE IS RF-PATCH, NOT THE CENTRE-PIXEL RF. In the pointwise regime the
sensible comparison is CNN against the centre-pixel RF. At alpha = 1 that
contrast is close to a restatement of CNN skill, because the centre-pixel RF
cannot see an arrangement at all and sits near rho = 0 on several axes. The
control that matters is RF-patch, which is handed the patch's mean and SD of
the refuge field: composition without configuration. The contrast therefore
asks "does the full patch buy anything beyond a composition summary of it".

TWO MARKINGS, WHICH MEAN DIFFERENT THINGS:
    open ring        |contrast| does not clear its own init-SD band, so the
                     difference is inconclusive
    hollow + red x   more than half the CNN fits at that cell never cleared
                     chance on validation -- the contrast is not a measurement
                     of model skill there, whatever its size

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python thesis_fig_paired_contrast_8i.py)
Writes: figures/thesis/fig7_8i_paired_contrast.{pdf,png}
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from thesis_figures import (ARM_LABEL, ARM_LS, ARM_FILL,
                            setup_style, save, canvas_width, frame,
                            grouped_legend, stagger_xticks, golden, GAP,
                            W_FIG, BASE_PT, INK,
                            INK2)
from thesis_fig_panels_8i import (load, degenerate, SHORT, LS, FILL, ARMS)

STEM = 'fig7_8i_paired_contrast'
# RF-patch only. Against the centre-pixel RF the contrast at alpha = 1 is
# close to a restatement of CNN skill -- that model cannot see an arrangement
# and sits near rho = 0 on several axes -- so the informative control is the
# one that gets composition without configuration.
BASELINE = 'rf_patch_summary'

# -- EVERYTHING BELOW MIRRORS fig4_paired_contrast ----------------------------
# This figure is read against fig4, panel for panel: same four axes, same
# contrast, one regime each. It therefore takes fig4's conventions rather than
# the 8i panel figures':
#   * the contrast wears the CNN's VERMILLION, not RF-patch's green. Green is
#     that model's own colour everywhere else in the set, so a green line here
#     would read as "RF-patch" rather than as "CNN minus RF-patch".
#   * panel order and titles are fig4's -- sample size, prevalence, resolution,
#     patch -- not the order the axes happen to sit in the 8i AXES table.
#   * prevalence is LINEAR: a designed ladder in a bounded quantity. fig4
#     always had this; the 8i panel figures were switched to match.
C_CONTRAST = '#D55E00'          # same vermillion as every Section-8 contrast
BAND = {'random': dict(fc='0.45', alpha=0.20, ec='none'),
        'extrap': dict(fc='0.75', alpha=0.30, ec='0.55', ls='--', lw=0.6)}
# (8i sub_exp, x label, x scale, panel title) -- fig4's wording and order
PANELS = [('8i_npoints_grf_alpha1',    'Sample size (training points)',
           'log',    'Sample size'),
          ('8i_prevalence_grf_alpha1', 'Prevalence',
           'linear', 'Prevalence'),
          ('8i_resolution_grf_alpha1', 'Resolution (m)',
           'log',    'Resolution'),
          ('8i_patch_grf_alpha1',      'Patch size (px)',
           'log',    'Patch size')]


def main():
    setup_style()
    i_all, _ = load()
    CN = 'cnn_mean' if (i_all.model == 'cnn_mean').any() else 'cnn'
    UNIT = 'species_idx' if 'species_idx' in i_all.columns else 'rep'
    deg = degenerate(i_all, UNIT)

    fig, axes = plt.subplots(2, 2, figsize=(canvas_width(STEM), W_FIG * 0.80),
                             gridspec_kw=dict(wspace=GAP['m'], hspace=GAP['l']))
    axes = axes.ravel()
    report = []

    for ax, (sexp, xlabel, xscale, title), letter in zip(axes, PANELS, 'abcd'):
        g = i_all[i_all.sub_exp == sexp]
        if g.empty:
            continue
        levels = sorted(g.level.dropna().unique().astype(float))

        # the noise band, pooled within species, one per arm, widest first
        bands = {}
        for arm in ARMS:
            sd = (g[(g.arm == arm) & (g.model == 'cnn')]
                  .groupby(['level', UNIT])['spearman_true'].std())
            if not sd.empty:
                bands[arm] = np.sqrt((sd ** 2).groupby(level=0).mean())
        for arm in sorted(bands, key=lambda a: -bands[a].mean()):
            b = bands[arm]
            x = b.index.astype(float).values
            o = np.argsort(x)
            ax.fill_between(x[o], -b.values[o], b.values[o], zorder=0,
                            label='_nolegend_', **BAND[arm])

        for arm in ARMS:
            d = g[g.arm == arm]
            hi = d[d.model == CN].groupby(['level', UNIT])['spearman_true'].mean()
            lo = d[d.model == BASELINE].groupby(['level', UNIT])['spearman_true'].mean()
            c = (hi - lo).dropna()
            if c.empty:
                continue
            mu = c.groupby(level='level').mean()
            ax.plot(c.index.get_level_values('level').astype(float), c.values,
                    '^', color=C_CONTRAST, ms=1.8, alpha=0.22, ls='none',
                    fillstyle=FILL[arm], zorder=2)
            ax.plot(mu.index.astype(float), mu.values, ls=LS[arm], marker='^',
                    color=C_CONTRAST, ms=3.4, lw=1.2, fillstyle=FILL[arm],
                    mew=0.8, zorder=3,
                    label='_nolegend_')
            for lv, v in mu.items():
                s = float(bands.get(arm, pd.Series(dtype=float)).get(lv, np.nan))
                inside = np.isfinite(s) and abs(v) <= s
                bad = (sexp, arm, float(lv)) in deg
                if inside:
                    ax.plot([float(lv)], [v], 'o', ms=8, mfc='none',
                            mec='0.25', mew=1.1, zorder=5)
                if bad:
                    ax.plot([float(lv)], [v], 'o', ms=8, mfc='white',
                            mec=C_CONTRAST, mew=1.1, zorder=6)
                    ax.plot([float(lv)], [v], 'x', ms=5, color='crimson',
                            mew=1.4, zorder=7)
                if inside or bad:
                    report.append((title, SHORT[BASELINE], arm, float(lv), v, s,
                                   'inconclusive' if inside else '',
                                   'CNN did not fit' if bad else ''))

        ax.axhline(0, color=INK, lw=0.7, alpha=0.6, zorder=1)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_xticks(levels)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:g}'))
        if xscale == 'log':
            ax.xaxis.set_minor_formatter(mtick.NullFormatter())
        # a linear ladder with uneven spacing crowds its labels -- same guard
        # as fig4 and fig5
        lvv = np.array(levels, float)
        if (xscale == 'linear' and len(lvv) > 1
                and np.diff(lvv).min() / (lvv.max() - lvv.min()) < 0.10):
            ax.tick_params(axis='x', labelsize=BASE_PT - 2.0)
            stagger_xticks(ax)
        ax.grid(axis='y', alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
        frame(ax)
        golden(ax)
        ax.set_title(f'({letter}) {title}', loc='left', fontsize=BASE_PT,
                     color=INK2, pad=3)

    # y label on the left column only, as in fig4
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(f'$\\Delta\\rho$ (CNN $-$ {SHORT[BASELINE]})')

    # the two markings the docstring describes, so the reader never has to
    # guess what a ring or a cross means
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    contrast = [Line2D([], [], color=C_CONTRAST, marker='^', ls='-', ms=4.0,
                       lw=1.2, label=f'CNN $-$ {SHORT[BASELINE]}')]
    arms = [Line2D([], [], color='0.45', marker='o', ms=4.0, lw=1.2,
                   ls=LS[a_], fillstyle=FILL[a_], mew=0.8, label=ARM_LABEL[a_])
            for a_ in ARMS]
    refs = [Patch(label=f'$\\pm$ init-SD, {ARM_LABEL[a_]}',
                  **{k: v for k, v in BAND[a_].items()})
            for a_ in ARMS]
    refs += [Line2D([], [], ls='none', marker='o', ms=8, mfc='none',
                    mec='0.25', mew=1.1,
                    label='inside the band (inconclusive)'),
             Line2D([], [], ls='none', marker='x', ms=5, color='crimson',
                    mew=1.4, label='$>$ half of CNN fits unlearned')]
    grouped_legend(fig, axes, [('Contrast', contrast), ('Arm', arms),
                               ('Reference and flags', refs)])
    save(fig, STEM)

    print('\n-- cells that are inconclusive and/or unfit --')
    for axis, base, arm, lv, v, s, inc, bad in report:
        print(f'  {axis:<11} CNN-{base:<9} {arm:<7} level={lv:>8g}  '
              f'delta={v:+.4f}  band={s:.4f}  {inc} {bad}'.rstrip())


if __name__ == '__main__':
    main()
