"""Shared loader for the v6 diagnostic scripts in this folder.

Builds the same namespace the thesis notebook has after its setup cells, by
executing those cells straight out of the notebook file. Nothing is re-
implemented here, so a diagnostic can never drift from the pipeline it
describes: _s8_eds_decorr, _s8_W_fields, _s8_orthogonalize_E, the PCA, the PC1
band thresholds and the sampler are the notebook's own objects.

Environment: wolf_sdm  (conda activate wolf_sdm)

usage:
    from _vs_env import notebook_env, FIGURES_DIR
    g = notebook_env()             # dict of the notebook's globals
    env = g['env_scaled_100m']
"""
import json
import os

# PROJECT ROOT. Set PROJECT_ROOT to the folder that holds code/ and data/;
# it defaults to the parent of this repository checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(_REPO))

# The experiment notebook of this repository: the helper functions below are
# taken from it by marker string, so the diagnostics use the same definitions
# as the experiment rather than copies of them.
NOTEBOOK = os.environ.get(
    "EXPERIMENT_NOTEBOOK",
    os.path.join(_REPO, "notebooks", "synthetic_experiment.ipynb"))
_CODE_OUT = os.environ.get("CODE_OUT", os.path.join(PROJECT_ROOT, "code.nosync"))
FIGURES_DIR = os.path.join(_CODE_OUT, "figures", "sus_scrofa", "synthetic")

# Cells are found by a marker string, never by index: inserting a cell in the
# notebook must not silently change what a diagnostic executes.
_MARKERS = [
    "os.environ.setdefault('OMP_NUM_THREADS'",   # 1   config, paths, hyperparameters
    "assert meta.get('scaled') is False",        # 2   loader: stack + per-layer min-max
    "def resample_env(",                         # 2.2 raster helpers
    "_valid_px3b = np.all(",                     # 3   PCA + virtual-species generator
    "def build_cnn(",                            # 5   CNN architecture
    "def _s8_run_level(env_base",                # 6   framework: sampler, bands, eds_decorr
]
_DEFS_ONLY = "def _s8_run_level_fixedtruth("     # 8.3 definitions (no sweep)
_CFG_CELL = "def _s8_make_W("                    # 8.6 configurational machinery


def _cell(nb, marker):
    hits = [i for i, c in enumerate(nb['cells']) if marker in ''.join(c['source'])]
    if len(hits) != 1:
        raise RuntimeError(f'marker {marker!r} matched {len(hits)} cells')
    return hits[0]


def notebook_env(with_config=True, verbose=True):
    """Execute the notebook's setup cells and return their globals.

    with_config=True also loads the configurational machinery (8f) and the
    fixed-truth definitions it depends on. Takes a few minutes: the framework
    cell fits the PCA on 200k pixels and runs its own eds_decorr diagnostic,
    and 8f runs its self-tests.
    """
    import matplotlib
    matplotlib.use('Agg')                       # scripts never open a window
    nb = json.load(open(NOTEBOOK))
    src = lambda i: ''.join(nb['cells'][i]['source'])
    g = {'__name__': '__main__'}
    for mk in _MARKERS:
        i = _cell(nb, mk)
        if verbose:
            print(f'  [_vs_env] exec cell {i}  ({mk[:36]}…)', flush=True)
        exec(compile(src(i), f'nb_cell{i}', 'exec'), g)
    if with_config:
        # definitions only -- the rest of that cell is the 8c-fixed sweep
        i = _cell(nb, _DEFS_ONLY)
        s = src(i)
        exec(compile(s[:s.index('# -- Run across both resolution conventions')],
                     f'nb_cell{i}_defs', 'exec'), g)
        i = _cell(nb, _CFG_CELL)
        if verbose:
            print(f'  [_vs_env] exec cell {i}  (8f configurational machinery)', flush=True)
        exec(compile(src(i), f'nb_cell{i}', 'exec'), g)
    return g


def exec_def(g, name, marker=None):
    """Execute ONE function definition out of the notebook into g.

    For helpers that live in a cell whose remainder is a sweep -- running the
    whole cell would launch the experiment. The source still comes from the
    notebook, so the diagnostic cannot drift from the pipeline.
    """
    import re
    nb = json.load(open(NOTEBOOK))
    pat = re.compile(r'^def ' + re.escape(name) + r'\(.*?(?=\n(?:def |[^\s#]))',
                     re.S | re.M)
    for c in nb['cells']:
        src = ''.join(c['source'])
        if marker and marker not in src:
            continue
        m = pat.search(src)
        if m:
            exec(compile(m.group(0), f'nb_def_{name}', 'exec'), g)
            return g[name]
    raise RuntimeError(f'definition {name!r} not found in the notebook')


def band_pixels(g):
    """(rows, cols, pc1) of every pixel valid in all covariates, plus the three
    PC1 band thresholds -- the same classification the extrapolative arm uses."""
    import numpy as np
    env = g['env_scaled_100m']
    ok2d = np.all(np.isfinite(env), axis=2)
    vr, vc = np.where(ok2d)
    pc1 = g['_pca3b'].transform(
        g['_sc3b'].transform(env[vr, vc, :].astype(np.float32)))[:, 0]
    return vr, vc, pc1, (g['_S8_PC1_TR'], g['_S8_PC1_VAL'], g['_S8_PC1_TE'])
