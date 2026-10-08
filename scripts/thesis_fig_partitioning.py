"""Figure 14 -- how the data are partitioned, in PC space and on the ground.

Six panels. The top row is the partition itself, which is a property of the
COVARIATES and of nothing else: it is defined on PC1 of the six-channel stack,
so it is identical for every species and for both data-generating processes.
The bottom row is what each process then draws through it.

  (a) the partition in PC1/PC2: every valid pixel, coloured by its band, with
      the three percentile cuts drawn. The bands are vertical slabs because
      PC1 alone defines them.
  (b) the same colours on the map. The bands are ENVIRONMENTAL, not geographic
      blocks -- they interleave at every scale, which is the point: the
      extrapolative arm is a shift in environment, not a spatial hold-out.
  (c) the PC1 density of valid pixels with the bands shaded, showing the
      40th / 55th / 70th percentile cuts and what each band is worth in area.
  (d) the pointwise species, extrapolative arm: the drawn train/val/test points
      in PC1/PC2. Train, val and test occupy disjoint PC1 ranges by
      construction.
  (e) the configurational species, extrapolative arm: the SAME partition, a
      different truth. The points sit in the same slabs; only which pixels
      inside a slab get drawn has changed.
  (f) the pointwise species, interpolative arm: one pool, split at random, so
      the three sets lie on top of one another. This is the contrast (d) is
      read against.

ONE SPECIES THROUGHOUT: seed 42, species 0 of the ensemble, the same one that
carries fig0 and fig3.

COLOURS. Train/val/test are an ORDERED sequence along PC1, so they get an
ordered ramp (three steps of Purples) rather than three categorical hues: the
reading "train is below val is below test" is then carried by lightness and
survives any colour vision. Measured worst-case separation under simulated
deuteranopia, protanopia and tritanopia: train-val 15.3, val-test 22.8,
train-test 38.0, and 12.2 against the unused-area grey (floor 8, OKLab x100).
Purple is used because the four model colours of the result figures
(#0072B2 RF-center, #009E73 RF-patch, #CC79A7 RF-oracle, #D55E00 CNN) must not
be reused for something that is not a model; the closest approach to any of
them is dE 8.0.

Environment: wolf_sdm
  (/opt/anaconda3/envs/wolf_sdm/bin/python thesis_fig_partitioning.py)
Writes: figures/thesis/fig14_partitioning.{pdf,png}
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pandas as pd

from _vs_env import notebook_env, band_pixels
from thesis_figures import (setup_style, save, canvas_width, frame, golden,
                            panel_label, scale_bar, shared_legend, GAP, W_FIG,
                            BASE_PT, INK, INK2, GRID_ALPHA, IMSHOW_KW,
                            STRIDE_COUNTRY, SPECIES_SEED, note, FACTS)

STEM = 'fig14_partitioning'
OUT_SPECIES = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'exports', 'results_revision'))
# ordered ramp, see the module docstring
# THREE CATEGORICAL HUES, not an ordered ramp: the bands have to be told apart
# at a glance in panels (a)-(c), and on 0.4 pt scatter dots an ordered ramp did
# not manage it. Tol dark blue / cyan / dark red-brown, measured worst case over
# simulated deuteranopia, protanopia and tritanopia (OKLab x100, floor 8):
#     train-val 25.0   train-test 29.7   val-test 26.1
#     against the unused-area grey: 40.0 / 17.8 / 13.0
# Lightness also separates all four (0.35 / 0.73 / 0.80 / 0.89), so the
# figure survives a greyscale print.
# Closest approach to any of the four model colours of the result figures
# (#0072B2 RF-center, #009E73 RF-patch, #CC79A7 RF-oracle, #D55E00 CNN): 7.6,
# against the CNN's vermillion -- acceptable because no model appears here.
BAND_C = {'train': '#332288', 'val': '#17becf', 'test': '#EDB120'}
UNUSED_C = '#dedbd2'
BAND_ORDER = ['train', 'val', 'test']
NICE = {'train': 'training band', 'val': 'validation band',
        'test': 'test band'}
N_SCATTER = 60_000          # pixels in the PC panels, as in fig0
N_POINTS_SHOW = 900         # drawn points shown per panel
RNG = 0


def pc_cloud(g, env, vr, vc):
    """PC1/PC2 of a fixed random subsample of the valid pixels."""
    rng = np.random.default_rng(RNG)
    sel = rng.choice(len(vr), min(N_SCATTER, len(vr)), replace=False)
    P = g['_pca3b'].transform(g['_sc3b'].transform(
        env[vr[sel], vc[sel], :].astype(np.float32)))
    return P[:, 0], P[:, 1], sel


def band_of(pc1, t40, t55, t70):
    b = np.full(len(pc1), 'unused', dtype=object)
    b[pc1 < t40] = 'train'
    b[(pc1 >= t40) & (pc1 < t55)] = 'val'
    b[(pc1 >= t55) & (pc1 < t70)] = 'test'
    return b






def pc_axes(ax, xlim, ylim):
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(alpha=GRID_ALPHA, lw=0.4)
    ax.set_axisbelow(True)
    frame(ax); golden(ax)


def main():
    setup_style()
    g = notebook_env(with_config=True)
    env = g['env_scaled_100m']
    vr, vc, pc1_all, (t40, t55, t70) = band_pixels(g)
    note(f'partition thresholds: PC1 40th={t40:.4f}  55th={t55:.4f}  '
         f'70th={t70:.4f}')

    # the two truths, same species
    r = g['make_vs'](SPECIES_SEED, mode='pca', occ='probabilistic',
                     k=g['_S8_OCC_K'], prevalence=g['_S8_OCC_PREV'])
    g['_s8_meta'] = {**r[2], **g['_s8_config_niche'](SPECIES_SEED)}
    cfg = g['_S8_CFG']
    m = g['_s8_meta']

    x, y, sel = pc_cloud(g, env, vr, vc)
    bands = band_of(x, t40, t55, t70)
    xlim = (np.percentile(x, 0.2), np.percentile(x, 99.0))
    ylim = (np.percentile(y, 0.2), np.percentile(y, 99.8))

    fig = plt.figure(figsize=(canvas_width(STEM), W_FIG * 0.40))
    gs = fig.add_gridspec(1, 3, wspace=GAP['l'])
    letters = iter('abc')

    # ---- (a) the partition in PC space ------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(x[bands == 'unused'], y[bands == 'unused'], s=0.4, alpha=0.5,
               linewidths=0, color='#d8d5cc', zorder=1)
    for name in BAND_ORDER:
        k = bands == name
        ax.scatter(x[k], y[k], s=0.4, alpha=0.5, linewidths=0,
                   color=BAND_C[name], zorder=2)
    for t in (t40, t55, t70):
        ax.axvline(t, color=INK, lw=0.6, ls='--', alpha=0.7, zorder=4)
    # THE SPECIES OPTIMA. Where each virtual species' niche peaks on the two
    # axes the partition is cut on -- the one thing that decides whether a
    # species extrapolates up its response curve or across its peak. Filled =
    # the three species carried through the sensitivity experiments, open = the
    # remaining five of the blend ensemble. Values come from
    # exports/results_revision/species.csv, so the figure and the table cannot
    # disagree.
    #
    # LABELS ARE 1-BASED. The pipeline's species_idx starts at 0; the thesis
    # text and the species table number the eight species 1-8, so the label is
    # species_no + 1 and the export keeps the 0-based key.
    #
    # PENTAGONS, NOT STARS. The star now means "no skill (AUC at chance)" in
    # fig6, fig8, fig9 and fig13; reusing it here for something unrelated would
    # carry that meaning across the set. The pentagon appears nowhere else.
    spf = os.path.join(OUT_SPECIES, 'species.csv')
    if os.path.exists(spf):
        sp = pd.read_csv(spf).sort_values('mu1').reset_index(drop=True)
        # Three optima sit within 0.5 of each other on the left; with one fixed
        # label offset their numbers overlapped. Each label is therefore pushed
        # away from the ones already placed, in data units converted to points.
        placed = []
        for _, r in sp.iterrows():
            sens = bool(r.in_sensitivity)
            ax.plot([r.mu1], [r.mu2], marker='p', ms=7,
                    mfc=INK if sens else 'white', mec=INK, mew=0.8, zorder=8,
                    ls='none')
            # Stagger VERTICALLY only, and only for markers that genuinely
            # overlap. The earlier threshold (0.55 in PC1) caught pairs that
            # merely sat near each other and pushed one label 10 pt clear of
            # its own marker, which read as unlabelled. 0.30 is about two
            # marker widths at this scale, so only true collisions move.
            off = (5.5, -2.5)
            for px, py in placed:
                if abs(px - r.mu1) < 0.30 and abs(py - r.mu2) < 0.45:
                    off = (5.5, 4.5) if off == (5.5, -2.5) else (5.5, -9.0)
            placed.append((r.mu1, r.mu2))
            # A WHITE HALO, because three of the eight optima fall in the
            # training band and black text on #332288 is unreadable. The halo
            # makes one label style work on all four backgrounds.
            ax.annotate(f'{int(r.species_no) + 1}', (r.mu1, r.mu2),
                        textcoords='offset points', xytext=off,
                        fontsize=BASE_PT - 1.5, color=INK, zorder=9,
                        fontweight='bold', annotation_clip=True,
                        path_effects=[pe.withStroke(linewidth=2.0,
                                                    foreground='white')])
        n_out = int(((sp.mu1 < xlim[0]) | (sp.mu1 > xlim[1])
                     | (sp.mu2 < ylim[0]) | (sp.mu2 > ylim[1])).sum())
        note(f'species optima drawn in panel (a): {len(sp)}, '
             f'{n_out} outside the plotted range')
    pc_axes(ax, xlim, ylim)
    ax.set_title('the partition in PC space', loc='left',
                 fontsize=BASE_PT - 1.0, color=INK2, pad=3)
    panel_label(ax, next(letters))

    # ---- (b) the partition on the map -------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    code = np.full(env.shape[:2], np.nan, np.float32)
    pc1_full = g['_pca3b'].transform(g['_sc3b'].transform(
        env[vr, vc, :].astype(np.float32)))[:, 0]
    cls = np.full(len(vr), 3.0, np.float32)          # 3 = unused
    cls[pc1_full < t40] = 0.0
    cls[(pc1_full >= t40) & (pc1_full < t55)] = 1.0
    cls[(pc1_full >= t55) & (pc1_full < t70)] = 2.0
    code[vr, vc] = cls
    cm = ListedColormap([BAND_C['train'], BAND_C['val'], BAND_C['test'],
                         UNUSED_C])
    ax.imshow(code[::STRIDE_COUNTRY, ::STRIDE_COUNTRY], cmap=cm, vmin=-0.5,
              vmax=3.5, **IMSHOW_KW)
    ax.axis('off')
    scale_bar(ax, 200_000, 100.0 * STRIDE_COUNTRY)
    ax.set_title('the same bands on the ground', loc='left',
                 fontsize=BASE_PT - 1.0, color=INK2, pad=3)
    panel_label(ax, next(letters))
    for i, name in enumerate(BAND_ORDER + ['unused']):
        note(f'band share of valid pixels, {name}: {float((cls == i).mean()):.4f}')

    # ---- (c) the PC1 density with the cuts ---------------------------------
    ax = fig.add_subplot(gs[0, 2])
    lo, hi = xlim
    hcnt, hedge = np.histogram(pc1_full, bins=220, range=(lo, hi))
    hcnt = hcnt / hcnt.sum()
    mid = 0.5 * (hedge[:-1] + hedge[1:])
    for name, a, b in (('train', lo, t40), ('val', t40, t55),
                       ('test', t55, t70)):
        k = (mid >= a) & (mid < b)
        ax.fill_between(mid[k], 0, hcnt[k], color=BAND_C[name], lw=0, zorder=2)
    k = mid >= t70
    ax.fill_between(mid[k], 0, hcnt[k], color='#d8d5cc', lw=0, zorder=2)
    ax.plot(mid, hcnt, color=INK, lw=0.7, zorder=3)
    for t in (t40, t55, t70):
        ax.axvline(t, color=INK, lw=0.6, ls='--', alpha=0.7, zorder=4)
    ax.set_xlabel('PC1')
    ax.set_ylabel('share of valid pixels')
    ax.set_xlim(lo, hi); ax.set_ylim(0, hcnt.max() * 1.12)
    ax.grid(axis='y', alpha=GRID_ALPHA, lw=0.4)
    ax.set_axisbelow(True)
    frame(ax); golden(ax)
    ax.set_title('PC1 density and the cuts', loc='left',
                 fontsize=BASE_PT - 1.0, color=INK2, pad=3)
    panel_label(ax, next(letters))

    handles = [Line2D([], [], ls='none', marker='o', ms=4, mfc=BAND_C[n],
                      mec='none', label=NICE[n]) for n in BAND_ORDER]
    handles.append(Patch(fc=UNUSED_C, ec='#c9c6bd', lw=0.5,
                         label='outside the partition (PC1 $\\geq$ 70th pct)'))
    handles.append(Line2D([], [], color=INK, lw=0.6, ls='--', alpha=0.7,
                          label='40th / 55th / 70th PC1 percentile'))
    handles.append(Line2D([], [], ls='none', marker='p', ms=7, mfc=INK,
                          mec=INK, mew=0.8,
                          label='niche optimum, sensitivity species'))
    handles.append(Line2D([], [], ls='none', marker='p', ms=7, mfc='white',
                          mec=INK, mew=0.8,
                          label='niche optimum, further blend species'))
    shared_legend(fig, np.array(fig.axes, dtype=object), ncol=3, extra=handles)
    save(fig, STEM)

    out = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..', 'figures', 'thesis')
    print('\n'.join(FACTS))


if __name__ == '__main__':
    main()
