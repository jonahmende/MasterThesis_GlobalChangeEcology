"""Appendix exports: EDS robustness, RF seed refits, blend fit health.

Assembles existing diagnostic outputs into one place, unrounded. Nothing is
refitted and no analysis changes.

Environment: wolf_sdm
"""
import os

import numpy as np
import pandas as pd

from _vs_env import FIGURES_DIR
from export_results_revision import OUT, read, unlearned, pooled_sd, UNIT, CN_RAW
from thesis_fig_panels_8i import auc_floor


def eds():
    """Both EDS sweeps in one long table: drawn sample and band pixels."""
    out = []
    p = os.path.join(FIGURES_DIR, 'eds_metric_properties.csv')
    d = pd.read_csv(p, float_precision='round_trip')
    for _, r in d.iterrows():
        out.append(dict(
            source='drawn sample', space='covariates',
            pixel_pool='the cell\'s own train/test points', cell=int(r.cell),
            arm=r.arm, n_ref=int(r.n_ref), n_query=int(r.n_query),
            pair='te_tr', check=r.check, variant=r.variant,
            K=int(r.K), value=float(r.value)))
    p = os.path.join(FIGURES_DIR, 'eds_band_robustness.csv')
    b = pd.read_csv(p, float_precision='round_trip')
    for _, r in b.iterrows():
        out.append(dict(
            source='band pixels', space=r.space, pixel_pool=r.pixel_pool,
            cell=np.nan, arm='extrapolative bands', n_ref=int(r.n_per_band),
            n_query=int(r.n_per_band), pair=r.pair, check='K',
            variant=f'K={int(r.K)}', K=int(r.K), value=float(r.value),
            seed=int(r.seed), n_pool_pixels=int(r.n_pool_pixels)))
    return pd.DataFrame(out)


def rf_seeds():
    p = os.path.join(FIGURES_DIR, 'rf_seed_variance.csv')
    d = pd.read_csv(p, float_precision='round_trip')
    g = d.groupby(['species_idx', 'species_seed', 'arm'])
    out = g.agg(n_seeds=('rf_random_state', 'size'),
                rho_mean=('spearman_true', 'mean'),
                rho_sd=('spearman_true', lambda x: x.std(ddof=1)),
                rho_min=('spearman_true', 'min'),
                rho_max=('spearman_true', 'max'),
                auc_mean=('roc_auc', 'mean'),
                auc_sd=('roc_auc', lambda x: x.std(ddof=1)),
                auc_min=('roc_auc', 'min'),
                auc_max=('roc_auc', 'max')).reset_index()
    out = out.rename(columns={'species_idx': 'species_no'})
    out['thesis_species'] = out.species_no + 1
    out['rf_random_states'] = ', '.join(str(s) for s in
                                        sorted(d.rf_random_state.unique()))
    return out


def blend_fit_health():
    d = read('8j_species_ensemble')
    rows = []
    for sexp in ('8j_alpha_grf', '8j_alpha_landuse'):
        g = d[d.sub_exp == sexp]
        src = 'random field' if sexp.endswith('grf') else 'forest map'
        for arm in sorted(g.arm.unique()):
            for a in sorted(g.alpha.dropna().unique()):
                s = g[(g.arm == arm) & np.isclose(g.alpha, a)]
                u = unlearned(s)
                per = (s[s.model == CN_RAW].groupby(UNIT)['spearman_true']
                       .std(ddof=1))
                rows.append(dict(
                    alpha=float(a), w_source=src, arm=arm,
                    pooled_init_sd_rho=pooled_sd(s),
                    init_sd_rho_min=float(per.min()),
                    init_sd_rho_max=float(per.max()),
                    n_species=int(per.notna().sum()),
                    n_inits_per_species=int(len(u) / max(per.notna().sum(), 1)),
                    cnn_fits_unlearned=int((~u.learned).sum()),
                    cnn_fits_total=int(len(u)),
                    unlearned_condition=bool(len(u) and
                                             (~u.learned).sum() / len(u) > 0.5),
                    n_val=int(s.n_vl.iloc[0]),
                    val_prevalence=float(s.real_prev_vl.iloc[0]),
                    val_auc_floor=float(auc_floor(s.n_vl.iloc[0],
                                                  s.real_prev_vl.iloc[0]))))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn, t in (('eds_robustness.csv', eds()),
                  ('rf_seed_variance_sd.csv', rf_seeds()),
                  ('blend_fit_health.csv', blend_fit_health())):
        t.to_csv(os.path.join(OUT, fn), index=False)
        print(f'{fn:<26} {len(t):>4} rows x {t.shape[1]:>2} cols')
    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    main()
