"""Thesis version of the four sensitivity panel figures (8a, 8b, 8c-fixed, 8e).

The notebook's own versions are s8_{sub_exp}_panels.png. These are redrawn in
the thesis style with ONE change of substance, and the notebook figures are
left exactly as they are.

  CEILINGS ARE DRAWN PER ARM. The notebook's AUC panel builds its ceiling as

      _od = _s8p[(_s8p.sub_exp == _sexp) & (_s8p.model == 'rf')]
      _op = _od.groupby(['level', _UNIT])['auc_oracle'].mean()
      _or8 = _op.groupby(level='level').mean()

  with no arm filter, so the single "oracle AUC (ceiling, mean)" line -- and
  the faint per-species squares beside it -- average the interpolative and
  extrapolative arms together. The two arms draw different points and have
  genuinely different ceilings (8c-fixed: 0.935 random against 0.869 extrap;
  8a: 0.908-0.935 against 0.884-0.895), so that line belongs to neither arm.
  Here each arm gets its own ceiling, in that arm's line style, labelled
  "ceiling random" / "ceiling extrap" -- matching the per-arm numbers in the
  results text.

  The same applies to the resolution-loss ceiling on the 8c-fixed Spearman panel,
  which is per arm for the same reason (0.894 random against 0.798 extrap at
  400 m, and the extrapolative arm is missing at the coarse levels).

Everything else follows the notebook: the headline res_mode only, the CNN
curve is the cnn_mean row, faint markers are the per-species values, and the
aggregation is mean within (level, species) then mean over species.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python thesis_fig_panels.py)
Writes: figures/thesis/fig5_panels_{8a,8b,8c_fixed,8e}.{pdf,png}
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from _vs_env import FIGURES_DIR
from thesis_fig_panels_8i import auc_floor, degenerate
from thesis_figures import (ARM_LABEL, ARM_LS, ARM_FILL, CEIL_COLOR,
                            CEIL_MARKER, grouped_legend, chance_line,
                            chance_handle, chance_level,
                            setup_style, panel_label, save, canvas_width,
                            frame, shared_legend, golden, GAP, W_FIG, BASE_PT, INK, INK2)

VERSION = 'v6_bio11'
HEADLINE_RES_MODE = 'constant_footprint'

S8EXP = {
    '8a':       ('Sample size (training points)', 'log', 'Sample size'),
    '8b':       ('Prevalence', 'linear', 'Prevalence'),
    '8c-fixed': ('Resolution (m)',      'log',    'Resolution, fixed 100 m truth'),
    '8e':       ('Patch size (px)',     'log',    'Patch size'),
}
ARMS = ['random', 'extrap']
LS = ARM_LS
FILL = ARM_FILL
MK = {'rf': 'o', 'cnn_mean': '^'}
COL = {'rf': '#0072B2', 'cnn_mean': '#D55E00'}
SHORT = {'rf': 'RF', 'cnn_mean': 'CNN'}
CEIL_C = '#8a8884'


def load():
    d = pd.read_csv(os.path.join(FIGURES_DIR, f'section8_two_regime_{VERSION}.csv'))
    cf = os.path.join(FIGURES_DIR, f'section8c_fixed_truth_{VERSION}.csv')
    if os.path.exists(cf):
        d = pd.concat([d, pd.read_csv(cf)], ignore_index=True)
    if 'res_mode' in d.columns:
        d = d[d.res_mode.isna()
              | d.res_mode.isin([HEADLINE_RES_MODE, 'baseline_100m'])]
    return d.reset_index(drop=True)


def ceiling(g, arm, col, unit):
    """Per-species values and their mean, for ONE arm -- never pooled."""
    s = g[(g.arm == arm) & (g.model == 'rf')]
    per = s.groupby(['level', unit])[col].mean().dropna()
    return per, per.groupby(level='level').mean()


def main():
    setup_style()
    d = load()
    CN = 'cnn_mean' if (d.model == 'cnn_mean').any() else 'cnn'
    UNIT = 'species_idx' if 'species_idx' in d.columns else 'rep'
    # the same fit-health rule as everywhere else, computed not assumed: under
    # the pointwise process it fires in no cell, and the legend reflects that
    deg = degenerate(d, UNIT)
    order = ['rf', CN]

    for sexp in [s for s in S8EXP if s in d.sub_exp.unique()]:
        xlabel, xscale, title = S8EXP[sexp]
        g = d[d.sub_exp == sexp]
        levels = sorted(g.level.unique().astype(float))
        stem = 'fig5_panels_' + sexp.replace('-', '_')

        # panel shape and gaps come from the shared phi scale, not from taste
        fig, axes = plt.subplots(1, 2, figsize=(canvas_width(stem),
                                                W_FIG * 0.34),
                                 gridspec_kw=dict(wspace=GAP['m']))
        # what this figure actually drew, so the legend lists only that
        drew = {}

        for ci, (col, ylab) in enumerate([
                ('spearman_true', 'Spearman $\\rho$'),
                ('roc_auc', 'AUC-ROC')]):
            ax = axes[ci]
            for m in order:
                for arm in ARMS:
                    s = g[(g.model == m) & (g.arm == arm)]
                    if s.empty:
                        continue
                    per = s.groupby(['level', UNIT])[col].mean()
                    mu = per.groupby(level='level').mean()
                    ax.plot(per.index.get_level_values('level').astype(float),
                            per.values, MK[m], color=COL[m], ms=1.8, alpha=0.22,
                            ls='none', fillstyle=FILL[arm], mew=0.0
                            if FILL[arm] == 'full' else 0.4, zorder=1)
                    ax.plot(mu.index.astype(float), mu.values, ls=LS[arm],
                            marker=MK[m], color=COL[m], ms=3.4, lw=1.2,
                            fillstyle=FILL[arm], mew=0.8, zorder=3,
                            label='_nolegend_')
                    # the two flags, drawn from the data rather than assumed:
                    # under the pointwise process neither fires anywhere, and
                    # the legend below reflects whatever actually happened
                    if m == CN:
                        bad = [float(lv) for lv in mu.index
                               if (sexp, arm, float(lv)) in deg]
                        if bad:
                            ax.plot(bad, mu.reindex(bad).values, 'o', ms=7,
                                    mfc='white', mec=COL[m], mew=1.1, zorder=6,
                                    ls='none')
                            ax.plot(bad, mu.reindex(bad).values, 'x', ms=5,
                                    color='crimson', mew=1.4, zorder=7,
                                    ls='none')
                            drew['unlearned'] = True

            # -- the chance level, ONE LINE for the whole AUC panel --------
            # Not a ring per no-skill point: the floor is a single number here
            # (n_te and the test prevalence are fixed by the design), so one
            # line states it for every model at once.
            if col == 'roc_auc':
                chance_line(ax, chance_level(
                    auc_floor(g.n_te.values, g.real_prev_te.values)))
                drew['chance'] = True

            # -- the ceilings, ONE PER ARM ---------------------------------
            ccol = ('oracle_spearman_infoloss' if col == 'spearman_true'
                    else 'auc_oracle')
            if ccol in g.columns and g[ccol].notna().any():
                cname = ('label-noise ceiling' if ccol == 'auc_oracle'
                         else 'resolution-loss ceiling')
                cmk = CEIL_MARKER[cname]
                drew[ccol] = cname
                for arm in ARMS:
                    per, mu = ceiling(g, arm, ccol, UNIT)
                    if mu.empty:
                        continue
                    ax.plot(per.index.get_level_values('level').astype(float),
                            per.values, cmk, color=CEIL_COLOR, ms=2.6,
                            alpha=0.22, ls='none', fillstyle=FILL[arm],
                            mew=0.5, zorder=1)
                    ax.plot(mu.index.astype(float), mu.values, ls=LS[arm],
                            color=CEIL_COLOR, lw=1.0, marker=cmk, ms=4.2,
                            fillstyle=FILL[arm], mew=0.7, zorder=2,
                            label='_nolegend_')

            ax.set_xscale(xscale)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylab)
            # every measured level gets a tick, on the linear axis too: 8b's
            # levels (0.05 .. 0.7) are unevenly spaced, so matplotlib's
            # automatic 0.2/0.4/0.6 labelled positions where nothing was run
            ax.set_xticks(levels)
            ax.xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda v, _: f'{v:g}'))
            if xscale == 'log':
                ax.xaxis.set_minor_formatter(mtick.NullFormatter())
            # 8b's two lowest levels (0.05, 0.1) sit 7.7 % of the range apart,
            # so their labels touch at the base size. Shrink the x labels only
            # where the levels are actually that close.
            span = max(levels) - min(levels)
            tight = min(np.diff(levels)) / span if span and len(levels) > 1 else 1
            if xscale == 'linear' and tight < 0.10:
                ax.tick_params(axis='x', labelsize=BASE_PT - 2.0)
            ax.grid(axis='y', alpha=0.25, lw=0.4)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
            frame(ax)
            golden(ax)
            ax.set_title(f'({"ab"[ci]})', loc='left', fontsize=BASE_PT,
                         color=INK, pad=3)

        # a full phi step below the x labels, so the legend reads as a
        # separate block rather than as part of the axis
        from matplotlib.lines import Line2D
        models = [Line2D([], [], color=COL[m], marker=MK[m], ls='-', ms=4.0,
                         lw=1.2, label=SHORT[m]) for m in order]
        arms = [Line2D([], [], color='0.45', marker='o', ms=4.0, lw=1.2,
                       ls=LS[a_], fillstyle=FILL[a_], mew=0.8,
                       label=ARM_LABEL[a_])
                for a_ in ARMS]
        refs = [Line2D([], [], color=CEIL_COLOR, marker=CEIL_MARKER[nm],
                       ms=4.6, lw=1.0, ls='-', label=nm)
                for k_, nm in drew.items()
                if k_ not in ('chance', 'no_skill', 'unlearned')]
        if drew.get('chance'):
            refs.append(chance_handle())
        if drew.get('unlearned'):
            refs.append(Line2D([], [], ls='none', marker='x', ms=5,
                               color='crimson', mew=1.4,
                               label='$>$ half of CNN fits unlearned'))
        grouped_legend(fig, axes, [('Model', models), ('Arm', arms),
                                   ('Reference and flags', refs)])
        save(fig, stem)

        # the numbers behind the two ceiling lines, for the caption
        for ccol in ('oracle_spearman_infoloss', 'auc_oracle'):
            if ccol not in g.columns or g[ccol].isna().all():
                continue
            for arm in ARMS:
                _, mu = ceiling(g, arm, ccol, UNIT)
                if mu.empty:
                    continue
                print(f'  {sexp:<9} {ccol:<26} {arm:<7}: '
                      + ' '.join(f'{float(k):g}={v:.3f}' for k, v in mu.items()))


if __name__ == '__main__':
    main()
