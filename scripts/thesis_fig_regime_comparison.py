"""The four axes in both regimes, one panel each (extrapolative arm).

This used to be the right-hand panel of each fig6_8i_* figure. Collected into
one four-panel figure so that fig6 can carry Spearman and AUC side by side, the
same layout as the alpha = 0 figures (fig5_panels_*).

Each panel: the same swept axis at alpha = 0 (pointwise DGP, from 8a / 8b /
8c-fixed / 8e) and at alpha = 1 (configurational DGP, from 8i), for RF-center
and the CNN, in the EXTRAPOLATIVE arm. That arm is the one where the regime is
supposed to matter; in the interpolative arm the two regimes barely separate.

The fit-health flag is drawn on BOTH sides, so a curve that comes from fits
which never cleared chance is marked whichever regime it belongs to.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python thesis_fig_regime_comparison.py)
Writes: figures/thesis/fig8_regime_comparison.{pdf,png}
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from thesis_figures import (setup_style, save, canvas_width, frame,
                            grouped_legend, golden, GAP, W_FIG, BASE_PT, INK,
                            INK2)
from thesis_fig_panels_8i import (load, degenerate, COL, MK, SHORT, flag)

STEM = 'fig8_regime_comparison'
ARM = 'extrap'
# fig4's panel order and wording, so the three four-panel figures of the set
# (fig4, fig7, fig8) can be read against each other panel for panel. The 8i
# AXES table has its own order, which is why this list is spelled out here.
# (alpha = 1 sub_exp, x label, x scale, alpha = 0 sub_exp, panel title)
PANELS = [('8i_npoints_grf_alpha1',    'Sample size (training points)',
           'log',    '8a',       'Sample size'),
          ('8i_prevalence_grf_alpha1', 'Prevalence',
           'linear', '8b',       'Prevalence'),
          ('8i_resolution_grf_alpha1', 'Resolution (m)',
           'log',    '8c-fixed', 'Resolution'),
          ('8i_patch_grf_alpha1',      'Patch size (px)',
           'log',    '8e',       'Patch size')]


def main():
    setup_style()
    i_all, a0 = load()
    CN = 'cnn_mean' if (i_all.model == 'cnn_mean').any() else 'cnn'
    CN0 = 'cnn_mean' if (a0.model == 'cnn_mean').any() else 'cnn'
    UNIT = 'species_idx'
    deg1, deg0 = degenerate(i_all, UNIT), degenerate(a0, UNIT)

    fig, axes = plt.subplots(2, 2, figsize=(canvas_width(STEM), W_FIG * 0.80),
                             gridspec_kw=dict(wspace=GAP['m'], hspace=GAP['l']))
    axes = axes.ravel()

    drew_flag = False
    for k, (ax, (sexp, xlabel, xscale, a0se, short)) in enumerate(
            zip(axes, PANELS)):
        g = i_all[i_all.sub_exp == sexp]
        if g.empty:
            continue
        levels = sorted(g.level.dropna().unique().astype(float))
        for m0, m1 in [('rf', 'rf'), (CN0, CN)]:
            key = 'rf' if m1 == 'rf' else CN
            colour, short_m = COL[key], SHORT[key]
            s0 = (a0[(a0.sub_exp == a0se) & (a0.model == m0) & (a0.arm == ARM)]
                  .groupby('level')['spearman_true'].mean())
            s1 = (g[(g.model == m1) & (g.arm == ARM)]
                  .groupby('level')['spearman_true'].mean())
            if not s0.empty:
                ax.plot(s0.index.astype(float), s0.values, ls=':', marker=MK[key],
                        color=colour, ms=3.4, lw=1.2, fillstyle='none', mew=0.8,
                        zorder=3, label='_nolegend_')
                if m1 == CN:
                    bad = [v for v in s0.index if (a0se, ARM, float(v)) in deg0]
                    drew_flag = drew_flag or bool(bad)
                    if bad:
                        flag(ax, [float(v) for v in bad],
                             s0.reindex(bad).values, colour)
            if not s1.empty:
                ax.plot(s1.index.astype(float), s1.values, ls='--', marker=MK[key],
                        color=colour, ms=3.4, lw=1.2, mew=0.8, zorder=3,
                        label='_nolegend_')
                drew_flag = drew_flag or bool(
                    [v for v in s1.index if (sexp, ARM, float(v)) in deg1])
                if m1 == CN:
                    bad = [v for v in s1.index if (sexp, ARM, float(v)) in deg1]
                    if bad:
                        flag(ax, [float(v) for v in bad],
                             s1.reindex(bad).values, colour)

        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        if k % 2 == 0:
            ax.set_ylabel('Spearman $\\rho$')
        ax.set_xticks(levels)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:g}'))
        if xscale == 'log':
            ax.xaxis.set_minor_formatter(mtick.NullFormatter())
        elif len(levels) > 1:
            span = max(levels) - min(levels)
            if np.diff(levels).min() / span < 0.10:
                ax.tick_params(axis='x', labelsize=BASE_PT - 2.0)
        ax.grid(axis='y', alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
        frame(ax)
        golden(ax)
        ax.set_title(f'({"abcd"[k]}) {short}', loc='left', fontsize=BASE_PT,
                     color=INK2, pad=3)

    # The model block carries the colour and the marker; the process block
    # carries the line style and the fill, once, instead of listing all four
    # model x process combinations.
    from matplotlib.lines import Line2D
    models = [Line2D([], [], color=COL[k_], marker=MK[k_], ls='-', ms=4.0,
                     lw=1.2, label=SHORT[k_]) for k_ in ('rf', CN)]
    procs = [Line2D([], [], color='0.45', marker='o', ms=4.0, lw=1.2, ls=':',
                    fillstyle='none', mew=0.8,
                    label='pointwise ($\\alpha$ = 0)'),
             Line2D([], [], color='0.45', marker='o', ms=4.0, lw=1.2, ls='--',
                    mew=0.8, label='configurational ($\\alpha$ = 1)')]
    refs = []
    if drew_flag:
        refs.append(Line2D([], [], ls='none', marker='x', ms=5,
                           color='crimson', mew=1.4,
                           label='$>$ half of CNN fits unlearned'))
    grouped_legend(fig, axes, [('Model', models), ('Process', procs),
                               ('Flags', refs)])
    save(fig, STEM)

    print(f'\n-- {ARM} arm, alpha = 0 vs alpha = 1, mean over species --')
    for sexp, _x, _sc, a0se, short in PANELS:
        g = i_all[i_all.sub_exp == sexp]
        for m0, m1, nm in [('rf', 'rf', 'RF-center'), (CN0, CN, 'CNN')]:
            s0 = (a0[(a0.sub_exp == a0se) & (a0.model == m0) & (a0.arm == ARM)]
                  .groupby('level')['spearman_true'].mean())
            s1 = (g[(g.model == m1) & (g.arm == ARM)]
                  .groupby('level')['spearman_true'].mean())
            print(f'  {short:<11} {nm:<10} a=0: '
                  + ' '.join(f'{float(k):g}={v:.3f}' for k, v in s0.items()))
            print(f'  {"":<11} {"":<10} a=1: '
                  + ' '.join(f'{float(k):g}={v:.3f}' for k, v in s1.items()))


if __name__ == '__main__':
    main()
