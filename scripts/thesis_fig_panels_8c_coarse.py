"""8c-fixed panels with the COARSE-PIXEL ceiling in the AUC panel.

Same figure as fig5_panels_8c_fixed, with one substitution in panel (b): the
ceiling is no longer auc_oracle (AUC of the 100 m truth AT THE POINT against
the drawn labels, which is flat in resolution by construction) but the AUC of
the block-mean truth over the point's coarse cell -- the best AUC a model could
reach knowing that cell perfectly. Panel (a) is unchanged and still carries the
rank version of the same idea, oracle_spearman_infoloss.

The ceiling values come from coarse_auc_ceiling_8c.py, which regenerates the
run's points from its seeds and verifies them against points_fingerprint before
computing anything. Run that first.

Both figures are kept: this one writes its own file, fig5_panels_8c_fixed is
left as it is.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python thesis_fig_panels_8c_coarse.py)
Writes: figures/thesis/fig5b_panels_8c_fixed_coarse.{pdf,png}
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from _vs_env import FIGURES_DIR
from thesis_figures import (setup_style, save, canvas_width, frame,
                            grouped_legend, chance_line, chance_handle,
                            chance_level, golden, GAP, W_FIG, BASE_PT, INK,
                            ARM_LABEL, CEIL_COLOR, CEIL_MARKER)
from thesis_fig_panels import (load, ceiling, S8EXP, ARMS, LS, FILL, MK, COL,
                               SHORT)
from thesis_fig_panels_8i import auc_floor

# The two ceilings this figure draws, each with the marker it carries in every
# other figure of the set. The rank panel's ceiling is the RESOLUTION-LOSS one
# (what a perfect model loses by seeing the coarsened covariates); the AUC
# panel's is the COARSE-PIXEL one (the AUC of the block-mean truth over the
# point's own coarse cell). They are different quantities, so they get
# different markers -- the earlier version used a plain square and a diamond,
# which collided with RF-patch's and RF-oracle's markers.
CEILINGS = {'spearman_true': 'resolution-loss ceiling',
            'roc_auc': 'coarse-pixel ceiling'}

SEXP = '8c-fixed'
STEM = 'fig5b_panels_8c_fixed_coarse'
COARSE_CSV = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', 'thesis',
                                           'coarse_auc_ceiling_8c.csv'))


def main():
    setup_style()
    d = load()
    CN = 'cnn_mean' if (d.model == 'cnn_mean').any() else 'cnn'
    UNIT = 'species_idx' if 'species_idx' in d.columns else 'rep'
    g = d[d.sub_exp == SEXP]
    levels = sorted(g.level.unique().astype(float))
    xlabel, xscale, _ = S8EXP[SEXP]

    if not os.path.exists(COARSE_CSV):
        raise SystemExit('run coarse_auc_ceiling_8c.py first -- '
                         f'{os.path.basename(COARSE_CSV)} is missing')
    cc = pd.read_csv(COARSE_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(canvas_width(STEM), W_FIG * 0.34),
                             gridspec_kw=dict(wspace=GAP['m']))

    for ci, (col, ylab) in enumerate([
            ('spearman_true', 'Spearman $\\rho$'),
            ('roc_auc', 'AUC-ROC')]):
        ax = axes[ci]
        for m in ['rf', CN]:
            for arm in ARMS:
                s = g[(g.model == m) & (g.arm == arm)]
                if s.empty:
                    continue
                per = s.groupby(['level', UNIT])[col].mean()
                mu = per.groupby(level='level').mean()
                ax.plot(per.index.get_level_values('level').astype(float),
                        per.values, MK[m], color=COL[m], ms=1.8, alpha=0.22,
                        ls='none', fillstyle=FILL[arm],
                        mew=0.0 if FILL[arm] == 'full' else 0.4, zorder=1)
                ax.plot(mu.index.astype(float), mu.values, ls=LS[arm],
                        marker=MK[m], color=COL[m], ms=3.4, lw=1.2,
                        fillstyle=FILL[arm], mew=0.8, zorder=3,
                        label='_nolegend_')

        cname = CEILINGS[col]
        cmk = CEIL_MARKER[cname]
        for arm in ARMS:
            if col == 'spearman_true':
                # unchanged: the rank ceiling, straight from the results file
                per, mu = ceiling(g, arm, 'oracle_spearman_infoloss', UNIT)
            else:
                # REPLACED: coarse-pixel ceiling instead of auc_oracle, which
                # is flat in resolution by construction
                s = cc[cc.arm == arm]
                if s.empty:
                    continue
                per = s.set_index(['level', 'species_idx'])['auc_ceiling_coarse']
                mu = per.groupby(level='level').mean()
            if mu.empty:
                continue
            ax.plot(per.index.get_level_values('level').astype(float),
                    per.values, cmk, color=CEIL_COLOR, ms=2.6, alpha=0.22,
                    ls='none', fillstyle=FILL[arm], mew=0.5, zorder=1)
            ax.plot(mu.index.astype(float), mu.values, ls=LS[arm],
                    color=CEIL_COLOR, lw=1.0, marker=cmk, ms=4.2,
                    fillstyle=FILL[arm], mew=0.7, zorder=2,
                    label='_nolegend_')
        if col == 'roc_auc':
            chance_line(ax, chance_level(
                auc_floor(g.n_te.values, g.real_prev_te.values)))

        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.set_xticks(levels)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:g}'))
        if xscale == 'log':
            ax.xaxis.set_minor_formatter(mtick.NullFormatter())
        ax.grid(axis='y', alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
        frame(ax)
        golden(ax)
        ax.set_title(f'({"ab"[ci]})', loc='left', fontsize=BASE_PT, color=INK,
                     pad=3)

    from matplotlib.lines import Line2D
    models = [Line2D([], [], color=COL[m], marker=MK[m], ls='-', ms=4.0,
                     lw=1.2, label=SHORT[m]) for m in ['rf', CN]]
    arms = [Line2D([], [], color='0.45', marker='o', ms=4.0, lw=1.2,
                   ls=LS[a_], fillstyle=FILL[a_], mew=0.8, label=ARM_LABEL[a_])
            for a_ in ARMS]
    refs = [Line2D([], [], color=CEIL_COLOR, marker=CEIL_MARKER[nm], ms=4.6,
                   lw=1.0, ls='-', label=nm) for nm in CEILINGS.values()]
    refs.append(chance_handle())
    grouped_legend(fig, axes, [('Model', models), ('Arm', arms),
                               ('Reference and flags', refs)])
    save(fig, STEM)

    print('\n-- coarse-pixel AUC ceiling, mean over the three species --')
    t = cc.groupby(['arm', 'level'])['auc_ceiling_coarse'].mean().unstack('level')
    print(t.round(4).to_string())
    print('\n-- per species --')
    for arm in ARMS:
        for sp in sorted(cc.species_idx.unique()):
            s = cc[(cc.arm == arm) & (cc.species_idx == sp)].sort_values('level')
            print(f'  {arm:<7} species {sp} (seed {int(s.species_seed.iloc[0])}): '
                  + '  '.join(f'{int(r.level)}m={r.auc_ceiling_coarse:.4f}'
                              for r in s.itertuples()))


if __name__ == '__main__':
    main()
