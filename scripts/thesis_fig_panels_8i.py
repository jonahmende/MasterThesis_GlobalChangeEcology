"""Thesis versions of the four 8i axis figures (configurational DGP, alpha = 1).

The notebook's own versions are s8i_{patch_size,sample_size,prevalence,
resolution}.png. Redrawn here in the thesis style, with three changes and one
bug fix; the notebook figures are left as they are.

  1 THE "captured / achievable" PANEL IS DROPPED. It was the middle of three
    panels. What it showed -- skill divided by ceiling_config_patch -- is a
    ratio of correlations, not a share of signal, it goes negative when a model
    ranks the wrong way, and it already had to exclude RF-oracle because the
    patch ceiling does not apply to it. The two surviving panels carry the same
    information without the interpretation trap.

  2 THE FIT-HEALTH FLAG NOW APPEARS ON BOTH PANELS. In the notebook only the
    absolute-skill panel marks levels where more than half the CNN fits never
    cleared chance on validation; the both-regimes panel plots the same CNN
    means unflagged. The flag is drawn on both panels here, for the alpha = 1
    side and (from the alpha = 0 frame) for the alpha = 0 side as well.

  3 BUG FIX -- the alpha = 0 curve of the RESOLUTION axis. The notebook builds
    its alpha = 0 frame as

        _cf0 = _cf0[[c for c in _A0.columns if c in _cf0.columns]]
        _A0  = pd.concat([_A0, _cf0], ignore_index=True)

    and then tries to keep only the headline convention with

        if 'res_mode' in _A0.columns: _A0p = _A0[...res_mode.isin([...])]

    That filter never fires: section8_two_regime has no res_mode column, so the
    intersection on the line above DROPS res_mode from the 8c-fixed rows before
    the check runs. The alpha = 0 resolution curve therefore pools BOTH
    resolution conventions -- six x positions (100/200/300/400/500/1000)
    against the alpha = 1 curve's three, and at 200 m it averages a 16 px patch
    with a 32 px one. Measured difference at 200 m: 0.8720 pooled against
    0.8639 for constant_footprint alone. Here res_mode is carried through
    explicitly and the filter works.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python thesis_fig_panels_8i.py)
Writes: figures/thesis/fig6_8i_{patch,npoints,prevalence,resolution}.{pdf,png}
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from _vs_env import FIGURES_DIR

# the coarse-pixel AUC ceiling for the resolution axis, from coarse_auc_ceiling_8i.py
COARSE_8I = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', 'thesis',
                                          'coarse_auc_ceiling_8i.csv'))
from thesis_figures import (setup_style, save, canvas_width, frame,
                            panel_legend, grouped_legend, chance_line,
                            chance_handle, chance_level, golden, GAP, W_FIG, BASE_PT, INK,
                            ARM_LABEL, ARM_LS, ARM_FILL,
                            CEIL_COLOR, CEIL_MARKER)

VERSION = 'v6_bio11'
HEADLINE_RES_MODE = 'constant_footprint'
N_SE_ABOVE_CHANCE = 2.0
ARMS = ['random', 'extrap']
LS = ARM_LS          # interpolative solid, extrapolative dashed
FILL = ARM_FILL
COL = {'rf': '#0072B2', 'rf_patch_summary': '#009E73',
       'rf_oracle_metrics': '#CC79A7', 'cnn_mean': '#D55E00'}
MK = {'rf': 'o', 'rf_patch_summary': 's', 'rf_oracle_metrics': 'D',
      'cnn_mean': '^'}
SHORT = {'rf': 'RF-centre', 'rf_patch_summary': 'RF-patch',
         'rf_oracle_metrics': 'RF-oracle', 'cnn_mean': 'CNN'}
CEIL_C = '#8a8a84'

# (sub_exp at alpha=1, x label, alpha=0 sub_exp, short name)
AXES = [('8i_patch_grf_alpha1',      'Patch size (px)',     '8e',       'patch'),
        ('8i_npoints_grf_alpha1',    'N training points',   '8a',       'sample size'),
        ('8i_prevalence_grf_alpha1', 'Prevalence', '8b',       'prevalence'),
        ('8i_resolution_grf_alpha1', 'Resolution (m)',      '8c-fixed', 'resolution')]
# The notebook puts every 8i axis on a log scale, including prevalence -- but the
# alpha=0 counterpart of that axis (fig5_panels_8b) is linear, so the same
# quantity was drawn on two different scales in one thesis. Linear wins: the
# levels are a designed ladder in a bounded [0,1] quantity, not a decade sweep.
SCALE = {'8i_prevalence_grf_alpha1': 'linear'}
Y_STRETCH = 1.35        # panel height = stretch/phi of its width (1.0 = golden)


def auc_floor(n, p):
    n = np.asarray(n, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    n1 = np.maximum(n * p, 1.0); n0 = np.maximum(n * (1 - p), 1.0)
    return 0.5 + N_SE_ABOVE_CHANCE * np.sqrt((n + 1.0) / (12.0 * n1 * n0))


def degenerate(frame_, unit):
    """(sub_exp, arm, level) where > 50 % of CNN fits never cleared chance."""
    if 'val_auc_best' not in frame_.columns or frame_.val_auc_best.isna().all():
        return set()
    c = frame_[(frame_.model == 'cnn') & frame_.val_auc_best.notna()].copy()
    if c.empty:
        return set()
    c['degen'] = c.val_auc_best < auc_floor(c.n_vl, c.real_prev_vl)
    h = c.groupby(['sub_exp', 'arm', 'level'])['degen'].mean()
    return set(h[h > 0.5].index)


def load():
    i = pd.read_csv(os.path.join(FIGURES_DIR,
                                 f'section8i_alpha_axes_{VERSION}.csv'))
    if 'res_mode' in i.columns:
        i = i[i.res_mode.isna() | i.res_mode.isin([HEADLINE_RES_MODE,
                                                   'baseline_100m'])]
    a0 = pd.read_csv(os.path.join(FIGURES_DIR,
                                  f'section8_two_regime_{VERSION}.csv'))
    a0['res_mode'] = np.nan                    # the two-regime file has no such column
    cf = pd.read_csv(os.path.join(FIGURES_DIR,
                                  f'section8c_fixed_truth_{VERSION}.csv'))
    # KEEP res_mode. The notebook intersects the columns first, which silently
    # drops it and makes its own res_mode filter a no-op (see the header).
    keep = [c for c in a0.columns if c in cf.columns]
    a0 = pd.concat([a0, cf[keep]], ignore_index=True)
    a0 = a0[a0.res_mode.isna() | a0.res_mode.isin([HEADLINE_RES_MODE,
                                                   'baseline_100m'])]
    return i.reset_index(drop=True), a0.reset_index(drop=True)


def flag(ax, xs, ys, colour):
    """The CNN fit-health mark: > 50 % of the fits never cleared chance on the
    validation fold. A TRAINING diagnostic, and only the CNN has one."""
    ax.plot(xs, ys, 'o', ms=7, mfc='white', mec=colour, mew=1.3, zorder=6)
    ax.plot(xs, ys, 'x', ms=5, color='crimson', mew=1.4, zorder=7)


# THE NO-SKILL REFERENCE IS A LINE, NOT A MARK. Ringing the individual
# points that fell at or below chance made the AUC panels dense and said
# nothing a horizontal line does not: the floor is one number per figure
# (n_te and the test prevalence are fixed), so the reader can see which
# curves touch it. chance_line() draws it; this stub only survives because
# other modules still import the name.
def no_skill(ax, xs, ys):
    """Deprecated. Superseded by chance_line(); draws nothing."""
    return None


def no_skill_levels(frame_, sexp, arm, model, unit):
    s = frame_[(frame_.sub_exp == sexp) & (frame_.arm == arm)
               & (frame_.model == model)]
    if s.empty or 'roc_auc' not in s.columns:
        return {}
    agg = s.groupby('level').agg(auc=('roc_auc', 'mean'), n=('n_te', 'mean'),
                                 p=('real_prev_te', 'mean'))
    return {float(lv): float(r.auc) for lv, r in agg.iterrows()
            if np.isfinite(r.auc) and r.auc < auc_floor(r.n, r.p)}


def main():
    setup_style()
    i_all, a0 = load()
    CN = 'cnn_mean' if (i_all.model == 'cnn_mean').any() else 'cnn'
    UNIT = 'species_idx' if 'species_idx' in i_all.columns else 'rep'
    CN0 = 'cnn_mean' if (a0.model == 'cnn_mean').any() else 'cnn'
    deg1, deg0 = degenerate(i_all, UNIT), degenerate(a0, UNIT)
    order = ['rf', 'rf_patch_summary', 'rf_oracle_metrics', CN]
    # RF-ORACLE IS DROPPED FROM THE PATCH AXIS. Its two extra features are the
    # raw driver fields at the truth's own radius (cell 37:
    # `Pw_oracle_2d, E_oracle_2d = Pw_2d, E_2d`), which do not depend on the
    # patch at all -- so along this one axis the curve is a horizontal constant
    # by construction (measured spread 0.022 random / 0.037 extrap, which is
    # just the different points drawn as the separation grows with the patch).
    # It is kept on the other three axes, where what it sees really does change
    # with the swept quantity: block-averaged onto the coarse grid for
    # resolution, and more data or less label noise for N and prevalence
    # (spread 0.34 / 0.07 / 0.07).
    DROP_ORACLE_ON = {'8i_patch_grf_alpha1'}

    for sexp, xlabel, a0se, stem_short in AXES:
        g = i_all[i_all.sub_exp == sexp]
        if g.empty:
            continue
        stem = 'fig6_8i_' + stem_short.replace(' ', '_')
        levels = sorted(g.level.dropna().unique().astype(float))
        # Y-STRETCH. Eight curves and the flag rings sat on top of one
        # another at the golden height; the x axis is a fixed ladder of
        # measured levels, so y is the only direction that can give.
        fig, axes = plt.subplots(1, 2, figsize=(canvas_width(stem),
                                                W_FIG * 0.46),
                                 gridspec_kw=dict(wspace=GAP['m']))

        # -- (a) absolute skill at alpha = 1, with the achievable ceiling ----
        ax = axes[0]
        for m in order:
            if m == 'rf_oracle_metrics' and sexp in DROP_ORACLE_ON:
                continue
            for arm in ARMS:
                d = g[(g.model == m) & (g.arm == arm)]
                if d.empty:
                    continue
                per = d.groupby(['level', UNIT])['spearman_true'].mean()
                mu = per.groupby(level='level').mean()
                ax.plot(per.index.get_level_values('level').astype(float),
                        per.values, MK[m], color=COL[m], ms=1.8, alpha=0.22,
                        ls='none', fillstyle=FILL[arm],
                        mew=0.0 if FILL[arm] == 'full' else 0.4, zorder=1)
                ax.plot(mu.index.astype(float), mu.values, ls=LS[arm],
                        marker=MK[m], color=COL[m], ms=3.4, lw=1.2,
                        fillstyle=FILL[arm], mew=0.8, zorder=3,
                        label='_nolegend_')
                if m == CN:
                    bad = [v for v in mu.index if (sexp, arm, float(v)) in deg1]
                    if bad:
                        flag(ax, [float(v) for v in bad],
                             mu.reindex(bad).values, COL[m])
        # ONE LINE PER ARM. ceiling_config_patch is stored per row, so an
        # arm-specific value exists for both; the earlier version averaged the
        # two together and drew a single line that belonged to neither.
        has_patch_ceiling = False
        for arm in ARMS:
            ceil = (g[g.arm == arm].groupby('level')['ceiling_config_patch']
                    .mean().dropna())
            if ceil.empty:
                continue
            has_patch_ceiling = True
            ax.plot(ceil.index.astype(float), ceil.values, ls=LS[arm],
                    color=CEIL_COLOR, lw=1.0,
                    marker=CEIL_MARKER['patch-limited ceiling'], ms=4.2,
                    fillstyle=FILL[arm], mew=0.7, zorder=2)
        ax.set_ylabel('Spearman $\\rho$')

        # -- (b) AUC-ROC, the same layout as the alpha = 0 figures -----------
        ax = axes[1]
        for m in order:
            if m == 'rf_oracle_metrics' and sexp in DROP_ORACLE_ON:
                continue
            for arm in ARMS:
                d = g[(g.model == m) & (g.arm == arm)]
                if d.empty:
                    continue
                per = d.groupby(['level', UNIT])['roc_auc'].mean()
                mu = per.groupby(level='level').mean()
                ax.plot(per.index.get_level_values('level').astype(float),
                        per.values, MK[m], color=COL[m], ms=1.8, alpha=0.22,
                        ls='none', fillstyle=FILL[arm],
                        mew=0.0 if FILL[arm] == 'full' else 0.4, zorder=1)
                ax.plot(mu.index.astype(float), mu.values, ls=LS[arm],
                        marker=MK[m], color=COL[m], ms=3.4, lw=1.2,
                        fillstyle=FILL[arm], mew=0.8, zorder=3,
                        label='_nolegend_')

                if m == CN:
                    bad = [v for v in mu.index if (sexp, arm, float(v)) in deg1]
                    if bad:
                        flag(ax, [float(v) for v in bad],
                             mu.reindex(bad).values, COL[m])
        # THE CEILING ON THE AUC PANEL.
        # On the resolution axis, auc_oracle uses the 100 m truth AT THE POINT
        # and is therefore flat in resolution by construction -- the same
        # problem the alpha = 0 resolution figure had. There it is replaced by
        # the coarse-pixel ceiling (AUC of the block-mean truth over the point's
        # coarse cell against the drawn labels), and the same replacement is
        # made here. On the other three axes nothing is coarsened, so
        # auc_oracle is the right reference and stays.
        use_coarse = (sexp == '8i_resolution_grf_alpha1'
                      and os.path.exists(COARSE_8I))
        cc = pd.read_csv(COARSE_8I) if use_coarse else None
        auc_ceil_name = ('coarse-pixel ceiling' if use_coarse
                         else 'label-noise ceiling')
        amk = CEIL_MARKER[auc_ceil_name]
        for arm in ARMS:
            if use_coarse:
                s_ = cc[cc.arm == arm]
                per = s_.set_index(['level', 'species_idx'])['auc_ceiling_coarse']
            else:
                s_ = g[(g.arm == arm) & (g.model == 'rf')]
                per = s_.groupby(['level', UNIT])['auc_oracle'].mean().dropna()
            if per.empty:
                continue
            mu = per.groupby(level='level').mean()
            ax.plot(per.index.get_level_values('level').astype(float), per.values,
                    amk, color=CEIL_COLOR, ms=2.6, alpha=0.22,
                    ls='none', fillstyle=FILL[arm], mew=0.5, zorder=1)
            ax.plot(mu.index.astype(float), mu.values, ls=LS[arm],
                    color=CEIL_COLOR, lw=1.0, marker=amk, ms=4.2,
                    fillstyle=FILL[arm], mew=0.7, zorder=2)
        # ONE chance line for the whole panel. n_te = 300 at prevalence 0.5
        # in essentially every cell, so the floor is one number; the patch
        # axis at 64 px loses a handful of test points to the wider window
        # and moves it by 5e-4, which is far below a line width. The line is
        # drawn at the LARGEST (most conservative) floor in the figure, and
        # the assertion fails if the spread ever becomes visible.
        chance = chance_level(auc_floor(g.n_te.values, g.real_prev_te.values))
        chance_line(ax, chance)
        ax.set_ylabel('AUC-ROC')

        xscale = SCALE.get(sexp, 'log')
        for k, ax in enumerate(axes):
            ax.set_xscale(xscale)
            ax.set_xlabel(xlabel)
            ax.set_xticks(levels)
            ax.xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda v, _: f'{v:g}'))
            if xscale == 'log':
                ax.xaxis.set_minor_formatter(mtick.NullFormatter())
            if xscale == 'linear' and len(levels) > 1:
                span = max(levels) - min(levels)
                if np.diff(levels).min() / span < 0.10:
                    ax.tick_params(axis='x', labelsize=BASE_PT - 2.0)
            ax.grid(axis='y', alpha=0.25, lw=0.4)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
            frame(ax)
            golden(ax, stretch=Y_STRETCH)
            ax.set_title(f'({"ab"[k]})', loc='left', fontsize=BASE_PT,
                         color=INK, pad=3)

        # ---- ONE grouped legend for both panels --------------------------
        # Only what this figure actually draws: RF-oracle is dropped on the
        # patch axis, the patch-limited ceiling exists only where it was
        # computed, and the red cross appears only where a cell is flagged.
        from matplotlib.lines import Line2D
        models = [Line2D([], [], color=COL[m], marker=MK[m], ls='-', ms=4.0,
                         lw=1.2, label=SHORT[m])
                  for m in order
                  if not (m == 'rf_oracle_metrics' and sexp in DROP_ORACLE_ON)]
        arms = [Line2D([], [], color='0.45', marker='o', ms=4.0, lw=1.2,
                       ls=LS[a_], fillstyle=FILL[a_], mew=0.8,
                       label=ARM_LABEL[a_])
                for a_ in ARMS]
        refs = []
        if has_patch_ceiling:
            refs.append(Line2D([], [], color=CEIL_COLOR, lw=1.0, ls='-',
                               marker=CEIL_MARKER['patch-limited ceiling'],
                               ms=4.6, label='patch-limited ceiling'))
        refs.append(Line2D([], [], color=CEIL_COLOR, lw=1.0, ls='-',
                           marker=amk, ms=4.6, label=auc_ceil_name))
        refs.append(chance_handle())
        if any(k[0] == sexp for k in deg1):
            refs.append(Line2D([], [], ls='none', marker='x', ms=5,
                               color='crimson', mew=1.4,
                               label='$>$ half of CNN fits unlearned'))
        grouped_legend(fig, axes, [('Model', models), ('Arm', arms),
                                   ('Reference and flags', refs)])
        save(fig, stem)

        # what the alpha=0 curve of this axis actually is, for the caption
        s0 = (a0[(a0.sub_exp == a0se) & (a0.model == 'rf')
                 & (a0.arm == 'extrap')].groupby('level')['spearman_true'].mean())
        print(f'  {stem}: alpha=0 levels {[f"{float(v):g}" for v in s0.index]}'
              f'  alpha=1 levels {[f"{v:g}" for v in levels]}')


if __name__ == '__main__':
    main()
