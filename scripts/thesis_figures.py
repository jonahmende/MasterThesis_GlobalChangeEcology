"""Thesis figures 0, 2 and 3 (figure 1 is the TikZ file next to this one).

Every panel is drawn from the notebook's OWN objects, loaded through _vs_env:
_s8_make_W, _s8_W_fields, _s8_orthogonalize_E, _s8_suit_config_from_fields,
_s8_suit_fn, the PCA (_pca3b/_sc3b) and _s8_sample_points. Nothing is
reimplemented and no pipeline code or result file is touched.

  fig0  the existing PC1/PC2 + map figure (notebook section 3.1, cell 12),
        re-rendered in the thesis style so the set matches. The notebook
        version stays where it is.
  fig2  construction of W and the configurational fields, one map excerpt,
        GRF row and forest-map row, shared colour scales down the columns.
  fig3  a configurational species: the (P_w, E_perp) space with its
        suitability contours and the drawn training points, and the S_config
        map with the same points.

STYLE INHERITED FROM THE EXISTING FIGURE (cell 12): YlOrRd at vmin=0/vmax=1
for suitability, imshow(origin='upper', interpolation='none'), axis off on
maps, colorbar(fraction=0.03, pad=0.02), the 60k-pixel scatter at s=0.4/
alpha=0.5 with default_rng(0), grid(alpha=0.3), and axis labels that carry the
explained-variance share. NOT inherited: its figsize (15 in), its font sizes,
its titles -- those are set by the thesis layout below.

Environment: wolf_sdm  (/opt/anaconda3/envs/wolf_sdm/bin/python thesis_figures.py)
Writes: figures/thesis/fig0_pointwise_species.{pdf,png}
        figures/thesis/fig2_w_construction.{pdf,png}
        figures/thesis/fig3_configurational_species.{pdf,png}
        figures/thesis/figure_facts.txt   (the numbers the captions need)
"""
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

from _vs_env import notebook_env, FIGURES_DIR

# -- THESIS LAYOUT: the three numbers to change if the document changes -------
TEXTWIDTH_CM = 15.9          # full text width of the thesis page
BASE_PT      = 8.5           # base font size in the figures
# Latin Modern SANS for text and, via sansmath, for maths too, so $\rho$ and
# $\Delta$ match the labels around them. Change this one line (and font.family
# in setup_style) to go back to the serif face.
LATEX_FONT   = (r'\usepackage{lmodern}\usepackage[T1]{fontenc}'
                r'\renewcommand*\familydefault{\sfdefault}'
                r'\usepackage{sansmath}\sansmath')
USETEX       = True

OUT_DIR = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', 'thesis'))

PHI = (1 + 5 ** 0.5) / 2        # 1.618...
# One spacing scale for every thesis figure, so the gaps are related instead of
# guessed: each step is the previous one divided by phi.
GAP = {'xl': 0.618, 'l': 0.382, 'm': 0.236, 's': 0.146, 'xs': 0.090}

CM = 1 / 2.54
W_FIG = TEXTWIDTH_CM * CM                      # target width of the SAVED file

# savefig(bbox_inches='tight') crops the unused margin away, so the saved file
# comes out narrower than the canvas and LaTeX would scale it back up at
# \textwidth -- which would change the font size on the page. The canvas is
# therefore made wider by the margin each figure actually loses, measured from
# the rendered output; save() re-measures and warns if a figure drifts.
_MARGIN_IN = {'fig0_pointwise_species': 1.77,
              'fig4_paired_contrast': 2.15,
              'fig5_panels_8a': 1.32, 'fig5_panels_8b': 1.32,
              'fig5_panels_8c_fixed': 1.32, 'fig5_panels_8e': 1.32,
              'fig5b_panels_8c_fixed_coarse': 1.32,
              'fig7_8i_paired_contrast': 1.10,
              'fig8_regime_comparison': 1.22,
              'fig9_8j_blend': 1.60, 'fig13_8j_blend_auc': 1.60,
              'fig10_8j_corrlen': 1.14, 'fig12_8j_wsource': 1.60,
              'fig11_8j_oracle_val': 1.40,
              'fig6_8i_patch': 1.22, 'fig6_8i_sample_size': 1.22,
              'fig6_8i_prevalence': 1.22, 'fig6_8i_resolution': 1.22,
              'fig2_w_construction': 1.69,
              'fig3_configurational_species': 0.85,
              'fig14_partitioning': 1.32}


def canvas_width(stem):
    return W_FIG + _MARGIN_IN.get(stem, 0.0)

# -- style constants lifted from notebook cell 12 -----------------------------
# YlOrRd, but CLIPPED AT THE PALE END. The full ramp starts at #ffffcc, which
# is dE 4.5 from white (OKLab x100) -- a low-suitability dot on a white page is
# then invisible. Starting at 0.25 of the ramp puts the lightest colour at
# #fed976, dE 12.2 from white, and the ramp keeps strictly monotone lightness
# (0.90 -> 0.38), which is what makes a sequential map readable under any
# colour vision. The DATA range is untouched: vmin/vmax stay 0 and 1, only the
# colours assigned to them change, and the colourbar shows the clipped ramp.
CMAP_SUIT = LinearSegmentedColormap.from_list(
    'YlOrRd_lo', plt.get_cmap('YlOrRd')(np.linspace(0.25, 1.0, 256)))
SUIT_LIM = dict(vmin=0.0, vmax=1.0)
CBAR_KW = dict(fraction=0.03, pad=0.02)
SCATTER_KW = dict(s=0.4, alpha=0.5)
N_PC_SCATTER = 60_000
RNG_PC = 0
GRID_ALPHA = 0.3
IMSHOW_KW = dict(origin='upper', interpolation='none')

# -- colour maps for the configurational fields -------------------------------
CMAP_PW = 'YlGn'                 # local woody proportion, sequential
CMAP_E = 'magma'                 # local edge density, sequential
CMAP_EP = 'PuOr_r'               # orthogonalised driver, diverging (PuOr is
                                 # already the notebook's diverging map)
CMAP_W = ListedColormap(['#f2f0eb', '#1b7837'])          # open / woody
# Habitat score 0-5: an ORDINAL scale, so its lightness must be monotone --
# otherwise the order is unreadable for anyone whose hue discrimination is
# reduced, and the map carries no class labels to fall back on. The notebook's
# own palette was not: measured OKLab lightness 0.976 / 0.824 / 0.896 / 0.793 /
# 0.717 / 0.505, i.e. class 2 is LIGHTER than class 1, and under simulated
# protanopia that pair separates by only dE 6.1 (0-100 scale, 8 is the floor).
# Six steps of cividis are monotone (0.30 -> 0.89), worst adjacent pair dE 7.5
# under the three dichromacies, and the hue is distinct from YlGn (P_w) and
# YlOrRd (suitability), which share the figure.
# Reversed, so that a HIGHER score is DARKER. Panel (g) of fig2 draws woody as
# dark green and panel (c) draws high P_w as dark green; running the score the
# other way would make the one panel that feeds them read upside down.
CMAP_LU = ListedColormap([plt.get_cmap('cividis_r')(x)
                          for x in np.linspace(0.05, 0.95, 6)])
CMAP_DENS = LinearSegmentedColormap.from_list(
    'greys_lo', plt.get_cmap('Greys')(np.linspace(0.05, 0.62, 256)))
INK = '#0b0b0b'
INK2 = '#52514e'
PRES_KW = dict(marker='o', s=3.2, linewidths=0.30, facecolors=INK,
               edgecolors='white', zorder=5)
ABS_KW = dict(marker='o', s=3.2, linewidths=0.45, facecolors='none',
              edgecolors=INK, zorder=4)

# -- what the figures show -----------------------------------------------------
SPECIES_SEED = 42                # notebook SEED; species 0 of the ensemble
SRC_LABEL = {'grf': 'random field', 'landuse': 'forest map'}
EXCERPT_KM = 20.0                # side length of the map excerpt
SEED_8J_ALPHA1_GRF = 40140       # 8j: _S8J_SEED_BASE + 0*2000 + 100 + 4*10
SEED_8J_ALPHA1_LU = 40340        # 8j: ... + 300 + 4*10
N_MAP_POINTS = 300               # points drawn on the country-scale map
RNG_MAP_POINTS = 7
STRIDE_COUNTRY = 8               # display stride for the country-scale maps

FACTS = []


def note(line):
    print(line, flush=True)
    FACTS.append(line)


def setup_style():
    plt.rcParams.update({
        'text.usetex': USETEX,
        'text.latex.preamble': LATEX_FONT,
        'font.family': 'sans-serif',
        'font.size': BASE_PT,
        'axes.titlesize': BASE_PT,
        'axes.labelsize': BASE_PT,
        'xtick.labelsize': BASE_PT - 0.5,
        'ytick.labelsize': BASE_PT - 0.5,
        'legend.fontsize': BASE_PT - 0.5,
        'axes.labelcolor': INK2,
        'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK,
        'axes.edgecolor': '#b8b7b0', 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': False,
        'lines.linewidth': 0.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def shared_legend(fig, axes, ncol=4, gap_in=0.30, extra=None):
    """One boxed legend for the whole figure, below the panels.

    Entries are collected from every axis and de-duplicated by label, so a
    panel that carries its own ceiling (8c-fixed) contributes it once and the
    two panels still share a single legend.

    ANCHORED TO THE AXES, NOT TO THE FIGURE. A fixed offset in figure
    coordinates gives a different visual gap on every figure, because the
    panels do not all occupy the same fraction of their canvas -- with
    set_box_aspect they rarely do. The bottom edge of the lowest panel (tick
    labels and axis label included) is measured after a first draw, and the
    legend is placed gap_in inches below that.
    """
    seen, h, lb = set(), [], []
    axs = list(np.atleast_1d(axes).ravel())
    for ax in axs:
        for hi, li in zip(*ax.get_legend_handles_labels()):
            if li not in seen:
                seen.add(li); h.append(hi); lb.append(li)
    for e in (extra or []):
        h.append(e); lb.append(e.get_label())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y0 = min(inv.transform(ax.get_tightbbox(r).p0)[1] for ax in axs)
    return fig.legend(h, lb, loc='upper center',
                      bbox_to_anchor=(0.5, y0 - gap_in / fig.get_figheight()),
                      ncol=ncol, frameon=True, fancybox=False,
                      edgecolor='#9a9992', framealpha=1.0,
                      fontsize=BASE_PT - 1.5, handlelength=2.4,
                      columnspacing=1.6, handletextpad=0.5, borderpad=0.5)


# -- one vocabulary for the two arms, used by every figure in the set ---------
ARM_LABEL = {'random': 'interpolative', 'extrap': 'extrapolative'}
ARM_LS = {'random': '-', 'extrap': '--'}
ARM_FILL = {'random': 'full', 'extrap': 'none'}

# Every ceiling is grey and carries the same arm convention as the models, but
# each TYPE keeps its OWN marker across every experiment. One shared marker put
# the rank ceiling of panel (a) and the AUC ceiling of panel (b) in the same
# shape, so two different quantities looked like one series seen twice. The
# four model markers are o / s / D / ^, so the ceilings take shapes none of
# them can be confused with -- and none is a star: at the 2.6 pt of the
# per-species scatter a filled star and an open star are the same blob, which
# destroyed the arm distinction exactly where it was needed.
CEIL_COLOR = '#8a8a84'
CEIL_MARKER = {
    'patch-limited ceiling': 'v',      # down triangle
    'label-noise ceiling': 'h',        # hexagon
    'resolution-loss ceiling': 'P',    # thick plus
    'coarse-pixel ceiling': '<',       # left triangle
}


def chance_line(ax, level):
    """The no-skill reference, as a LINE across the AUC panel.

    Replaces the ring that used to be drawn around each no-skill point. One
    horizontal line states the chance level once for every model at the same
    time and leaves the markers alone; whether a curve has skill is then read
    off directly instead of hunting for rings. It also frees the open circle,
    which fig4 and fig7 use for 'inside the init-SD band'."""
    ax.axhline(level, color='0.45', lw=0.8, ls=(0, (4, 2)), zorder=1.5)


def stagger_xticks(ax, step=2, extra_pt=6.0):
    """Drop every `step`-th x tick label by `extra_pt` points.

    The prevalence ladder (0.05, 0.1, 0.2, 0.3, 0.5, 0.7) puts its two lowest
    levels 7.7 % of the range apart. In a two-column panel their labels touch
    even at the reduced tick size, and '0.05' '0.1' reads as '0.050.1'.
    Shrinking further would make them unreadable, and dropping one would hide
    a level that was actually run, so alternate labels are moved down instead.
    """
    for i, t in enumerate(ax.xaxis.get_major_ticks()):
        if i % step:
            t.set_pad(t.get_pad() + extra_pt)


def chance_level(floors, tol=0.01):
    """The single number chance_line() should be drawn at.

    `floors` is the prevalence-aware Mann-Whitney floor of every fit in the
    panel. It is one number by design -- n_te = 300 at prevalence 0.5 -- but
    a few cells lose test points to a wider patch window (8e and the 8i patch
    axis at 64 px, where n_te falls to 256 in the worst fit). That moves the
    floor by at most 0.0055, which is well under a line width, so the line is
    drawn at the LARGEST floor in the panel, the conservative one. The
    assertion fires if the spread ever becomes something a reader could see.
    """
    f = np.asarray(floors, float)
    f = f[np.isfinite(f)]
    assert f.size and np.ptp(f) < tol, 'chance level varies visibly in this panel'
    return float(f.max())


def chance_handle(label='chance level (no skill)'):
    """The legend entry that matches chance_line(). Same dash pattern, same
    grey, no marker, so the line in the panel is identifiable."""
    return Line2D([], [], color='0.45', lw=0.8, ls=(0, (4, 2)), label=label)


def no_skill_ring(ax, xs, ys, ms=13):
    """Deprecated, superseded by chance_line(). Draws nothing."""
    return None


class _Header(Line2D):
    """A legend entry that is a heading: it draws no handle at all.

    Its own class, not a bare Line2D -- the handler map keys on the type, and
    mapping Line2D would blank every line handle in the legend."""

    def __init__(self, text):
        super().__init__([], [], ls='none', marker='none', label=text)


class _NoArtist(HandlerBase):
    """Draws an EMPTY artist, not an empty list: HandlerBase.legend_artist
    indexes artists[0], so returning [] raises."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        a = Line2D([], [], ls='none', marker='none')
        a.set_transform(trans)
        return [a]


def _header(text):
    return _Header(text)


def grouped_legend(fig, axes, blocks, ncol=None, gap_in=0.20, fontsize=None,
                   pad_in=0.22):
    """One legend ROW for the whole figure, in blocks with bold headings.

    blocks is [(title, [handles]), ...]. Each block is its own legend box with
    a matplotlib `title`, because a title is the only heading matplotlib will
    left-align flush with the handles: an entry carrying a blank handle keeps
    the handle column, and the heading ends up indented by its width. The boxes
    are measured after a first draw and then laid out as one centred row, so
    they read as a single legend.

    Listing every model x arm combination separately made these legends longer
    than the panels they explained. The arm is a style convention, not a
    series, so it is stated once in its own block instead of doubling the model
    list.
    """
    blocks = [(t, [h for h in hs if h is not None]) for t, hs in blocks if hs]
    if not blocks:
        return []
    axs = list(np.atleast_1d(axes).ravel())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    # MEASURE THE X AXIS, NOT THE AXES BOX. golden() fixes the data area with
    # set_box_aspect, so the axes rectangle keeps its original height and
    # reaches far below the plot -- anchoring to it left a band of white
    # between the panels and the legend. The x axis (ticks plus label) sits at
    # the bottom of what is actually drawn.
    def _bottom(ax):
        bb = ax.xaxis.get_tightbbox(r) or ax.get_tightbbox(r)
        return inv.transform(bb.p0)[1]
    y0 = min(_bottom(ax) for ax in axs)
    y = y0 - gap_in / fig.get_figheight()
    fs = fontsize or (BASE_PT - 1.5)

    legs = []
    for title, hs in blocks:
        # usetex ignores fontweight, so the heading is bolded in TeX itself.
        ttl = (r'\textbf{%s}' % title) if plt.rcParams['text.usetex'] else title
        leg = fig.legend(hs, [h.get_label() for h in hs], title=ttl,
                         loc='upper left', bbox_to_anchor=(0.0, y),
                         frameon=True, fancybox=False, edgecolor='#9a9992',
                         framealpha=1.0, fontsize=fs, handlelength=2.4,
                         handletextpad=0.5, borderpad=0.5, labelspacing=0.4)
        leg._legend_box.align = 'left'
        leg.get_title().set_fontsize(fs)
        leg.get_title().set_fontweight('bold')   # no-op under usetex
        legs.append(leg)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x0f = inv.transform((0, 0))[0]
    widths = [inv.transform(l.get_window_extent(r).size)[0] - x0f for l in legs]
    pad = pad_in / fig.get_figwidth()
    x = 0.5 - (sum(widths) + pad * (len(legs) - 1)) / 2.0
    for l, w in zip(legs, widths):
        l.set_bbox_to_anchor((x, y), transform=fig.transFigure)
        x += w + pad
    return legs


def panel_legend(fig, ax, ncol=1, gap_in=0.30, extra=None, **kw):
    """A boxed legend under ONE panel, centred on it.

    Same anchoring rule as shared_legend: the panel's own bottom edge (tick
    labels and axis label included) is measured after a draw, so the gap is the
    same everywhere regardless of panel shape."""
    h, lb = ax.get_legend_handles_labels()
    for e in (extra or []):            # proxy handles for the flag symbols
        h.append(e); lb.append(e.get_label())
    if not h:
        return None
    fig.canvas.draw()
    bb = ax.get_tightbbox(fig.canvas.get_renderer())
    inv = fig.transFigure.inverted()
    x0, y0 = inv.transform(bb.p0)
    x1, _ = inv.transform(bb.p1)
    opts = dict(frameon=True, fancybox=False, edgecolor='#9a9992',
                framealpha=1.0, fontsize=BASE_PT - 1.5, handlelength=2.4,
                columnspacing=1.6, handletextpad=0.5, borderpad=0.5)
    opts.update(kw)
    return fig.legend(h, lb, loc='upper center',
                      bbox_to_anchor=(0.5 * (x0 + x1),
                                      y0 - gap_in / fig.get_figheight()),
                      ncol=ncol, **opts)


def golden(ax, stretch=1.0):
    """Give the plotting area itself the golden ratio (w:h = phi:1).

    set_box_aspect fixes the DATA area, so panels keep the same shape whatever
    the tick labels around them do -- which is what makes a multi-panel figure
    look even.

    stretch > 1 makes the panel taller than golden while keeping its width. The
    8i panel figures need it: eight curves (four models x two arms) plus the
    flag rings in one frame collide at the golden height, and the x axis is a
    fixed ladder, so the only free direction is y."""
    ax.set_box_aspect(stretch / PHI)


def frame(ax, color='#9a9992', lw=0.6):
    """A closed box around the plot area, thin and recessive."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)
    ax.tick_params(top=False, right=False)


def despine(ax, keep=('left', 'bottom')):
    """Drop the frame down to the two axes that carry data."""
    for name, sp in ax.spines.items():
        sp.set_visible(name in keep)
    ax.tick_params(top=False, right=False)


def panel_label(ax, letter, extra=None, inside=True):
    """(a), (b), ... in the corner of the panel, optionally with a caption."""
    ax.text(0.015 if inside else -0.02, 0.985,
            f'({letter}) {extra}' if extra else f'({letter})',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=BASE_PT - (2.0 if extra else 0.0), color=INK,
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none',
                      alpha=0.75) if inside else None, zorder=10)


def scale_bar(ax, length_m, res_m, label=None, frac_y=0.055, frac_x=0.06,
              color=INK, below=True, pad_ax=0.045):
    """Horizontal scale bar for a map panel. Bar length is in DATA units (px).

    BELOW THE PANEL BY DEFAULT. Placed inside the axes it landed on the map --
    across southern Germany in the country panels, and on open data everywhere
    else -- because an imshow panel is filled edge to edge by definition, so
    there is no empty corner to retreat to. The bar now sits in the margin under
    the axes: x still in data coordinates, so the length stays a true scale,
    y in axes coordinates, so the bar cannot collide with anything drawn. The
    label goes beside the bar rather than under it, to keep the margin shallow.

    below=False restores the old in-panel placement for a panel that has room.
    """
    x0, x1 = ax.get_xlim()
    y1, y0 = ax.get_ylim()[1], ax.get_ylim()[0]     # origin='upper'
    w = abs(x1 - x0)
    h = abs(y0 - y1)
    n_px = length_m / res_m
    xs = min(x0, x1) + (0.0 if below else frac_x * w)
    if label is None:
        label = (f'{length_m/1000:.0f}\\,km' if length_m >= 1000
                 else f'{length_m:.0f}\\,m')
    if below:
        from matplotlib.transforms import blended_transform_factory
        tf = blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot([xs, xs + n_px], [-pad_ax, -pad_ax], transform=tf, color=color,
                lw=1.4, solid_capstyle='butt', zorder=9, clip_on=False)
        ax.text(xs + n_px * 1.08, -pad_ax, label, transform=tf, ha='left',
                va='center', fontsize=BASE_PT - 1.5, color=color, zorder=9,
                clip_on=False)
        return
    ys = max(y0, y1) - frac_y * h
    ax.plot([xs, xs + n_px], [ys, ys], color=color, lw=1.4,
            solid_capstyle='butt', zorder=9, clip_on=False)
    ax.text(xs + n_px / 2, ys - 0.012 * h, label, ha='center', va='bottom',
            fontsize=BASE_PT - 1.5, color=color, zorder=9,
            bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none',
                      alpha=0.7))


def save(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, kw in (('pdf', {}), ('png', dict(dpi=600))):
        p = os.path.join(OUT_DIR, f'{stem}.{ext}')
        fig.savefig(p, bbox_inches='tight', pad_inches=0.01, **kw)
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w_cm = bb.width * 2.54
    plt.close(fig)
    flag = '' if abs(w_cm - TEXTWIDTH_CM) < 0.25 else '   <-- OFF TARGET'
    note(f'saved: {stem}.pdf / {stem}.png  (width {w_cm:.2f} cm, '
         f'target {TEXTWIDTH_CM:.2f}){flag}')


# =============================================================================
# shared inputs
# =============================================================================
def build_inputs(g):
    """Everything the figures share: the fields of both W sources, the
    species, and the map excerpt."""
    env = g['env_scaled_100m']
    cfg = g['_S8_CFG']
    valid = np.all(np.isfinite(env), axis=2)
    note(f'stack: {env.shape[0]}x{env.shape[1]} px at 100 m, '
         f'{env.shape[2]} covariates; valid = {int(valid.sum()):,} px')
    note(f'_S8_CFG: r_m={cfg["r_m"]:.0f} m, corr_len={cfg["corr_len_m"]:.0f} m, '
         f'frac_woody={cfg["frac_woody"]}, grf_seed={cfg["grf_seed"]}, '
         f'mu_P(requested)={cfg["mu_P"]}, sig_P={cfg["sig_P"]}, w_P={cfg["w_P"]}')

    # the species (pointwise niche + its configurational niche)
    r = g['make_vs'](SPECIES_SEED, mode='pca', occ='probabilistic',
                     k=g['_S8_OCC_K'], prevalence=g['_S8_OCC_PREV'])
    g['_s8_meta'] = {**r[2], **g['_s8_config_niche'](SPECIES_SEED)}
    meta = g['_s8_meta']
    suit_point = g['_s8_suit_fn'](env)
    g['_s8_suit_100m'] = suit_point
    note(f'species seed {SPECIES_SEED}: mu_E={meta["mu_E"]:.4f} '
         f'sig_E={meta["sig_E"]:.4f}; pointwise niche mu_PC=({meta["mus_pc"][0]:.3f}, '
         f'{meta["mus_pc"][1]:.3f}) power={meta["power"]:.3f}')

    # the two W sources and their fields
    smooth = None
    fields = {}
    for src in ('grf', 'landuse'):
        if src == 'grf':
            rng = np.random.default_rng(cfg['grf_seed'])
            noise = rng.standard_normal(env.shape[:2]).astype(np.float32)
            smooth = g['_s8_gf'](noise, sigma=cfg['corr_len_m'] / 100.0)
            del noise
            W = g['_s8_make_W'](env.shape[:2], cfg['corr_len_m'] / 100.0,
                                cfg['frac_woody'], seed=cfg['grf_seed'])
        else:
            W = np.isin(g['landuse'], list(g['WOODY_CLASSES'])).astype(np.float32)
        Pw, E, win = g['_s8_W_fields'](W, valid, 100.0, cfg['r_m'])
        inner = g['minimum_filter'](valid.view(np.uint8), size=win,
                                    mode='constant', cval=0).astype(bool)
        mu_P = float(np.nanmean(Pw[inner]))
        Ep = g['_s8_orthogonalize_E'](E, Pw, inner)
        ok = inner & np.isfinite(Ep)
        Sc = g['_s8_suit_config_from_fields'](Pw, Ep, ok, mu_P, cfg['sig_P'],
                                              meta['mu_E'], meta['sig_E'],
                                              cfg['w_P'])
        fields[src] = dict(W=W, Pw=Pw, E=E, Ep=Ep, ok=ok, inner=inner,
                           mu_P=mu_P, S=Sc, win=int(win))
        note(f'[{src}] window={win} px, realised woody fraction='
             f'{float(W[valid].mean()):.3f}, derived mu_P={mu_P:.4f}, '
             f'S_config defined on {int(ok.sum()):,} px')
    return env, valid, suit_point, fields, smooth, meta, cfg


def pick_excerpt(g, fields, valid, side_km=EXCERPT_KM):
    """One window, the same for both rows, fully inside BOTH interior buffers,
    with a genuine woody/open mix on the forest map. Deterministic scan."""
    side = int(round(side_km * 1000 / 100))
    ok_both = fields['grf']['ok'] & fields['landuse']['ok']
    Wlu = fields['landuse']['W']
    H, W = ok_both.shape
    best, best_score = None, -np.inf
    step = 100
    for r0 in range(0, H - side, step):
        for c0 in range(0, W - side, step):
            sub_ok = ok_both[r0:r0 + side:10, c0:c0 + side:10]
            if not sub_ok.all():
                continue
            f = float(Wlu[r0:r0 + side:5, c0:c0 + side:5].mean())
            if not (0.30 <= f <= 0.60):
                continue
            # prefer a genuine mix AND fine-grained structure (many boundaries)
            sub = Wlu[r0:r0 + side:2, c0:c0 + side:2]
            edges = float((sub[:, 1:] != sub[:, :-1]).mean()
                          + (sub[1:, :] != sub[:-1, :]).mean())
            score = edges - 2.0 * abs(f - 0.45)
            if score > best_score:
                best_score, best = score, (r0, c0)
    if best is None:
        raise RuntimeError('no excerpt satisfied the criteria')
    r0, c0 = best
    note(f'excerpt: rows {r0}-{r0+side}, cols {c0}-{c0+side} '
         f'({side} px = {side_km:.0f} km per side); forest-map woody fraction '
         f'in it = {float(Wlu[r0:r0+side, c0:c0+side].mean()):.3f}')
    return r0, c0, side


# =============================================================================
# figure 0 -- the existing PC1/PC2 figure, thesis style
# =============================================================================
def fig0(g, env, suit_point):
    w = canvas_width('fig0_pointwise_species')
    fig, axes = plt.subplots(1, 2, figsize=(w, W_FIG * 0.44),
                             gridspec_kw=dict(width_ratios=[1.0, 1.35],
                                              wspace=0.22))
    ax = axes[0]
    im = ax.imshow(suit_point[::STRIDE_COUNTRY, ::STRIDE_COUNTRY],
                   cmap=CMAP_SUIT, **IMSHOW_KW, **SUIT_LIM)
    ax.axis('off')
    scale_bar(ax, 200_000, 100.0 * STRIDE_COUNTRY)
    panel_label(ax, 'a')

    ax = axes[1]
    vr, vc = g['_vr3b'], g['_vc3b']
    n = min(N_PC_SCATTER, len(vr))
    sel = np.random.default_rng(RNG_PC).choice(len(vr), size=n, replace=False)
    E = env[vr[sel], vc[sel], :]
    P = g['_pca3b'].transform(g['_sc3b'].transform(E))
    S = suit_point[vr[sel], vc[sel]]
    m = np.isfinite(S)
    sc = ax.scatter(P[m, 0], P[m, 1], c=S[m], cmap=CMAP_SUIT, **SCATTER_KW,
                    **SUIT_LIM, rasterized=True, linewidths=0)
    # THE COLOURBAR GETS ITS OWN MAPPABLE. Built from the scatter it inherited
    # the scatter's alpha = 0.5 and rendered half-transparent on white, so its
    # top end read as mauve while the map and the points at the same value are
    # deep red -- the bar disagreed with the data it was labelling.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(SUIT_LIM['vmin'], SUIT_LIM['vmax']),
                        cmap=CMAP_SUIT)
    cb = fig.colorbar(sm, ax=ax, **CBAR_KW)
    cb.set_label('suitability', fontsize=BASE_PT)
    cb.ax.tick_params(labelsize=BASE_PT - 0.5)
    evr = g['_pca3b'].explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}\\,\\% var)')
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}\\,\\% var)')
    ax.grid(alpha=GRID_ALPHA, lw=0.4)
    ax.set_axisbelow(True)
    panel_label(ax, 'b')
    note(f'fig0: PC scatter on {int(m.sum()):,} of {len(vr):,} valid pixels '
         f'(rng {RNG_PC}); map displayed at stride {STRIDE_COUNTRY}')
    save(fig, 'fig0_pointwise_species')


# =============================================================================
# figure 2 -- construction of W and the configurational fields
# =============================================================================
def fig2(g, fields, smooth, exc, cfg):
    r0, c0, side = exc
    sl = (slice(r0, r0 + side), slice(c0, c0 + side))
    win = fields['grf']['win']

    # shared limits down the columns, computed on the excerpt of BOTH rows
    def lim(key, lo=1.0, hi=99.0):
        v = np.concatenate([fields[s][key][sl][np.isfinite(fields[s][key][sl])]
                            for s in ('grf', 'landuse')])
        return float(np.percentile(v, lo)), float(np.percentile(v, hi))

    pw_lo, pw_hi = lim('Pw')
    e_lo, e_hi = lim('E')
    ep_a, ep_b = lim('Ep', 2.0, 98.0)
    ep_m = max(abs(ep_a), abs(ep_b))
    note(f'fig2 shared colour limits: P_w [{pw_lo:.3f}, {pw_hi:.3f}], '
         f'E [{e_lo:.4f}, {e_hi:.4f}], E_perp +/-{ep_m:.2f} '
         f'(1st/99th and 2nd/98th percentile over both rows of the excerpt)')

    w = canvas_width('fig2_w_construction')
    fig, axes = plt.subplots(2, 5, figsize=(w, W_FIG * 0.455),
                             gridspec_kw=dict(wspace=0.05, hspace=0.06))
    letters = iter('abcdefghij')
    col_titles = [None, r'binary $W$', r'$P_w$', r'$E$', r'$E_\perp$']

    for row, src in enumerate(('grf', 'landuse')):
        f = fields[src]
        # column 0: the source of W
        ax = axes[row, 0]
        if src == 'grf':
            sm = smooth[sl]
            ax.imshow(sm, cmap='Greys_r', **IMSHOW_KW,
                      vmin=np.percentile(sm, 1), vmax=np.percentile(sm, 99))
            lab = 'smoothed noise'
        else:
            imlu = ax.imshow(g['landuse'][sl], cmap=CMAP_LU, **IMSHOW_KW,
                             norm=BoundaryNorm(np.arange(-0.5, 6.5, 1.0), 6))
            lab = 'habitat score'
        ax.set_ylabel(('random field' if src == 'grf' else 'forest map'),
                      fontsize=BASE_PT, color=INK, labelpad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        panel_label(ax, next(letters), extra=lab)
        scale_bar(ax, 5000, 100.0)
        if src == 'landuse':
            cb = fig.colorbar(imlu, ax=ax, location='bottom', fraction=0.055,
                              pad=0.08, aspect=14, ticks=range(6))
            cb.ax.tick_params(labelsize=BASE_PT - 2.0, length=1.5, pad=1.5)
            cb.outline.set_linewidth(0.4)

        # column 1: binary W, with the focal window drawn at its true size
        ax = axes[row, 1]
        ax.imshow(f['W'][sl], cmap=CMAP_W, **IMSHOW_KW, vmin=0, vmax=1)
        cw = side * 0.70
        ax.add_patch(plt.Rectangle((cw - win / 2, cw - win / 2), win, win,
                                   fill=False, ec='#D55E00', lw=1.0, zorder=6))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        panel_label(ax, next(letters))

        # columns 2-4: the three fields, shared scales
        for j, (key, cmap, kw) in enumerate((
                ('Pw', CMAP_PW, dict(vmin=pw_lo, vmax=pw_hi)),
                ('E', CMAP_E, dict(vmin=e_lo, vmax=e_hi)),
                ('Ep', CMAP_EP, dict(vmin=-ep_m, vmax=ep_m))), start=2):
            ax = axes[row, j]
            im = ax.imshow(f[key][sl], cmap=cmap, **IMSHOW_KW, **kw)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            panel_label(ax, next(letters))
            if row == 1:
                cb = fig.colorbar(im, ax=axes[:, j], location='bottom',
                                  fraction=0.055, pad=0.06, aspect=28)
                cb.ax.tick_params(labelsize=BASE_PT - 2.0, length=1.5, pad=1.5)
                cb.outline.set_linewidth(0.4)
                cb.set_label(col_titles[j], fontsize=BASE_PT - 0.5, labelpad=1)

    for j, t in enumerate(col_titles):
        if t is not None and j == 1:
            axes[0, j].set_title(t, fontsize=BASE_PT - 0.5, color=INK2, pad=3)
    save(fig, 'fig2_w_construction')


# =============================================================================
# figure 3 -- a configurational species
# =============================================================================
def draw_points(g, env, fields, src, seed, cfg):
    """The training fold of the interpolative arm at alpha = 1, drawn by the
    pipeline's own sampler against S_config."""
    S = fields[src]['S']
    patch_ok = g['_s8_patch_ok'](env, g['_s8_patch_buffer_px'](100, g['PATCH'],
                                                               None))
    vm = g['make_valid_mask'](env, S, g['PATCH']) & patch_ok
    valid_S = np.all(np.isfinite(env), axis=2) & np.isfinite(S)
    occ_c = g['_s8_solve_c'](S[valid_S], g['_S8_OCC_K'], g['_S8_OCC_PREV'])
    p_occ = g['_s8_logistic'](S[valid_S], occ_c, g['_S8_OCC_K'])
    occupancy = float(p_occ.mean())
    n_draw = g['_S8_N_POINTS'] + 2 * g['_S8_N_WIN']
    r, c, lbl = g['_s8_sample_points'](
        S, vm, n_draw, g['PRES_RATIO'], seed=seed,
        min_sep_px=g['PATCH'] * g['_S8_SEP_PATCHES'], occ_prev=None,
        occ_c=occ_c)
    lbl = lbl.astype(np.int8)
    tr_i, _ = g['_tts'](np.arange(len(lbl)), test_size=2 * g['_S8_N_WIN'],
                        stratify=lbl, random_state=seed)
    r, c, lbl = r[tr_i], c[tr_i], lbl[tr_i]
    note(f'[{src}] alpha=1 training fold: {len(lbl)} points '
         f'({int(lbl.sum())} presences / {int((1-lbl).sum())} absences), '
         f'sampler seed {seed}, intercept c={occ_c:.4f}, '
         f'landscape occupancy={occupancy:.4f}')
    return r, c, lbl.astype(bool), occupancy


def fig3(g, env, fields, exc, meta, cfg):
    r0, c0, side = exc
    sl = (slice(r0, r0 + side), slice(c0, c0 + side))
    rows = ('grf', 'landuse')
    # FOUR ROWS, TWO COLUMNS. The column is the W source, the row is the view:
    # driver space, PC space, the country map, the excerpt. Reading down a
    # column now follows one W source through all four; reading across a row
    # compares the two sources on the same view.
    fig = plt.figure(figsize=(canvas_width('fig3_configurational_species'),
                              W_FIG * 1.52))
    # the country map is tall and narrow, so at equal row heights it came
    # out much smaller than the square panels above and below it
    gs = fig.add_gridspec(4, 2, wspace=GAP['l'], hspace=GAP['m'],
                          height_ratios=[1.0, 1.0, 1.45, 1.0])
    letters = iter('abcdefgh')
    rng_pts = np.random.default_rng(RNG_MAP_POINTS)

    for col, src in enumerate(rows):
        f = fields[src]
        # NO DRAWN LABELS ANYWHERE IN THIS FIGURE. It belongs to the part of
        # the thesis that introduces the suitability surface, before occurrence
        # is drawn at all, so every panel carries S and nothing else. The
        # presence/absence overlay, its two hard-to-separate markers and the
        # pixel-count colourbar are gone; one suitability scale now runs through
        # all eight panels.
        # ---- row 0: the (P_w, E_perp) space, coloured by mean S ------------
        ax = fig.add_subplot(gs[0, col])
        ok = f['ok']
        idx = np.flatnonzero(ok.ravel())
        sub = np.random.default_rng(1).choice(idx, min(400_000, idx.size),
                                              replace=False)
        x = f['Pw'].ravel()[sub]
        y = f['Ep'].ravel()[sub]
        sv = f['S'].ravel()[sub]
        hb = ax.hexbin(x, y, C=sv, reduce_C_function=np.mean, gridsize=55,
                       cmap=CMAP_SUIT, **SUIT_LIM, mincnt=1, linewidths=0.0,
                       rasterized=True)
        # the analytic niche over the same space, as three level lines. Dark
        # grey, not the vermillion of the earlier version: that colour is the
        # CNN's everywhere else in the set.
        gx = np.linspace(*ax.get_xlim(), 220)
        gy = np.linspace(*ax.get_ylim(), 220)
        GX, GY = np.meshgrid(gx, gy)
        Z = g['_s8_suit_config_from_fields'](
            GX.astype(np.float32), GY.astype(np.float32),
            np.ones_like(GX, bool), f['mu_P'], cfg['sig_P'],
            meta['mu_E'], meta['sig_E'], cfg['w_P'])
        cs = ax.contour(GX, GY, Z, levels=[0.25, 0.5, 0.75], colors=INK2,
                        linewidths=0.6, zorder=6)
        cs.set(label=None)
        ax.clabel(cs, fmt='%.2f', fontsize=BASE_PT - 2.0, inline=True)
        ax.set_xlabel(r'$P_w$ (local woody proportion)')
        ax.set_ylabel(r'$E_\perp$ (orthogonalised edge density)')
        ax.grid(alpha=GRID_ALPHA, lw=0.4)
        ax.set_axisbelow(True)
        frame(ax); golden(ax)
        ax.set_title(SRC_LABEL[src], loc='left', fontsize=BASE_PT, color=INK2,
                     pad=3)
        panel_label(ax, next(letters))

        # ---- row 1: the SAME points in PC1/PC2 -----------------------------
        # The configurational niche lives on (P_w, E_perp), which is built from
        # W alone and is close to orthogonal to the environmental axes. Plotted
        # on PC1/PC2 the presences therefore do NOT separate from the absences:
        # that non-separation is the point of the alpha = 1 design, and it is
        # what a pointwise model is handed. Compare fig0(b), where the same two
        # axes carry the pointwise species' whole niche.
        ax = fig.add_subplot(gs[1, col])
        sub2 = np.random.default_rng(2).choice(
            len(g['_vr3b']), min(N_PC_SCATTER, len(g['_vr3b'])), replace=False)
        br, bc = g['_vr3b'][sub2], g['_vc3b'][sub2]
        Pb = g['_pca3b'].transform(g['_sc3b'].transform(
            env[br, bc, :].astype(np.float32)))
        Sb = f['S'][br, bc]
        m2 = np.isfinite(Sb)
        ax.scatter(Pb[m2, 0], Pb[m2, 1], c=Sb[m2], cmap=CMAP_SUIT, **SUIT_LIM,
                   **SCATTER_KW, rasterized=True, linewidths=0)
        evr = g['_pca3b'].explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}\\,\\% var)')
        ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}\\,\\% var)')
        ax.set_xlim(np.percentile(Pb[:, 0], [0.2, 99.0]))
        ax.set_ylim(np.percentile(Pb[:, 1], [0.2, 99.8]))
        ax.grid(alpha=GRID_ALPHA, lw=0.4)
        ax.set_axisbelow(True)
        frame(ax); golden(ax)
        panel_label(ax, next(letters))
        # how much of S these two axes carry at all
        for j, nm in ((0, 'PC1'), (1, 'PC2')):
            r_ = np.corrcoef(Pb[m2, j], Sb[m2])[0, 1]
            note(f'[{src}] Pearson r(S, {nm}) over the plotted pixels = '
                 f'{r_:+.4f}')

        # ---- row 2: country-scale S_config map -----------------------------
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(f['S'][::STRIDE_COUNTRY, ::STRIDE_COUNTRY], cmap=CMAP_SUIT,
                  **IMSHOW_KW, **SUIT_LIM)
        ax.add_patch(plt.Rectangle((c0 / STRIDE_COUNTRY, r0 / STRIDE_COUNTRY),
                                   side / STRIDE_COUNTRY, side / STRIDE_COUNTRY,
                                   fill=False, ec='#0072B2', lw=0.8, zorder=7))
        ax.axis('off')
        scale_bar(ax, 200_000, 100.0 * STRIDE_COUNTRY)
        panel_label(ax, next(letters))

        # ---- row 3: the excerpt at full resolution -------------------------
        ax = fig.add_subplot(gs[3, col])
        im = ax.imshow(f['S'][sl], cmap=CMAP_SUIT, **IMSHOW_KW, **SUIT_LIM)
        for s in ax.spines.values():
            s.set_edgecolor('#0072B2'); s.set_linewidth(0.8)
        ax.set_xticks([]); ax.set_yticks([])
        scale_bar(ax, 5000, 100.0)
        panel_label(ax, next(letters))

    # ONE colourbar for the whole figure: every panel now shows the same
    # quantity on the same scale. Labelled in words -- a subscripted
    # $S_{\mathrm{config}}$ at 7.5 pt was not readable on the printed page.
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.016, pad=0.015,
                      location='right')
    cb.set_label('configurational suitability $S$', fontsize=BASE_PT)
    cb.ax.tick_params(labelsize=BASE_PT - 0.5)
    save(fig, 'fig3_configurational_species')


def main():
    setup_style()
    g = notebook_env(with_config=True)
    env, valid, suit_point, fields, smooth, meta, cfg = build_inputs(g)
    exc = pick_excerpt(g, fields, valid)
    fig0(g, env, suit_point)
    fig2(g, fields, smooth, exc, cfg)
    fig3(g, env, fields, exc, meta, cfg)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'figure_facts.txt'), 'w') as fh:
        fh.write('\n'.join(FACTS) + '\n')
    print('\nfacts written to figure_facts.txt')


if __name__ == '__main__':
    main()
