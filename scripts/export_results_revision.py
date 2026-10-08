"""Raw, unrounded exports for the Results revision.

Every definition here is taken from the figure modules, not restated, so the
tables and the figures cannot drift apart:

  auc_floor, degenerate   thesis_fig_panels_8i  (no-skill and fit health)
  pooled init SD          as in fig4 / fig7: sqrt(mean over species of the
                          variance across that species' three CNN inits)
  contrast                formed PER SPECIES, then averaged -- the data draw
                          cancels, which it does not if means are differenced
  ceilings                fig5 / fig6: rank ceiling = oracle_spearman_infoloss
                          (pointwise, resolution axis only) or
                          ceiling_config_patch (configurational); AUC ceiling =
                          auc_oracle, replaced by the coarse-pixel ceiling on
                          both resolution axes

NOTHING IS ROUNDED and no analysis is changed. Writes CSVs plus README.md.

Environment: wolf_sdm
  (/opt/anaconda3/envs/wolf_sdm/bin/python export_results_revision.py)
"""
import os

import numpy as np
import pandas as pd

from _vs_env import FIGURES_DIR
from thesis_fig_panels_8i import auc_floor, N_SE_ABOVE_CHANCE

V = 'v6_bio11'
HEADLINE = ['constant_footprint', 'baseline_100m']
OUT = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', '..', '..',
                                    'exports', 'results_revision'))
THESIS = os.path.normpath(os.path.join(FIGURES_DIR, '..', '..', 'thesis'))
UNIT = 'species_idx'
CN, CN_RAW = 'cnn_mean', 'cnn'

# how the stored model keys are named in the exports
NAME_POINT = {'rf': 'RF', CN: 'CNN-Ensemble'}
NAME_CFG = {'rf': 'RF-centre', 'rf_patch_summary': 'RF-patch',
            'rf_oracle_metrics': 'RF-oracle', CN: 'CNN-Ensemble'}

# (process, experiment, source file, sub_exp, alpha-0 twin for the 8i axes)
SENS = [
    ('pointwise', 'sample size', 'two_regime', '8a', None),
    ('pointwise', 'prevalence', 'two_regime', '8b', None),
    ('pointwise', 'resolution', 'c_fixed', '8c-fixed', None),
    ('pointwise', 'patch size', 'two_regime', '8e', None),
    ('configurational', 'sample size', 'alpha_axes', '8i_npoints_grf_alpha1', '8a'),
    ('configurational', 'prevalence', 'alpha_axes', '8i_prevalence_grf_alpha1', '8b'),
    ('configurational', 'resolution', 'alpha_axes', '8i_resolution_grf_alpha1', '8c-fixed'),
    ('configurational', 'patch size', 'alpha_axes', '8i_patch_grf_alpha1', '8e'),
]


def read(stem):
    return pd.read_csv(os.path.join(FIGURES_DIR, f'section{stem}_{V}.csv'),
                       float_precision='round_trip')


def load_all():
    tr = read('8_two_regime')
    cf = read('8c_fixed_truth')
    cf = cf[cf.res_mode.isin(HEADLINE)]
    ia = read('8i_alpha_axes')
    ia = ia[ia.res_mode.isna() | ia.res_mode.isin(HEADLINE)]
    bl = read('8j_species_ensemble')
    coarse = {}
    for tag, fn in (('8c-fixed', 'coarse_auc_ceiling_8c.csv'),
                    ('8i_resolution_grf_alpha1', 'coarse_auc_ceiling_8i.csv')):
        p = os.path.join(THESIS, fn)
        if os.path.exists(p):
            coarse[tag] = pd.read_csv(p, float_precision='round_trip')
    return dict(two_regime=tr, c_fixed=cf, alpha_axes=ia, blend=bl,
                coarse=coarse)


def pooled_sd(d):
    """fig4/fig7 noise band for ONE condition: pool the within-species init
    variance, i.e. sqrt(mean over species of Var across that species' inits)."""
    sd = d[d.model == CN_RAW].groupby(UNIT)['spearman_true'].std()
    return float(np.sqrt((sd ** 2).mean())) if len(sd) else np.nan


def unlearned(d):
    """per-init flag: best validation AUC below the prevalence-aware floor."""
    c = d[(d.model == CN_RAW) & d.val_auc_best.notna()].copy()
    c['learned'] = c.val_auc_best >= auc_floor(c.n_vl, c.real_prev_vl)
    return c


# =============================================================================
def sensitivity(D):
    rows_sp, rows_in, rows_cond = [], [], []
    for proc, exp, src, sexp, _a0 in SENS:
        d = D[src]
        g = d[d.sub_exp == sexp]
        if g.empty:
            continue
        names = NAME_POINT if proc == 'pointwise' else NAME_CFG
        cz = D['coarse'].get(sexp)
        for arm in sorted(g.arm.unique()):
            for lv in sorted(g.level.dropna().unique().astype(float)):
                s = g[(g.arm == arm) & (g.level == lv)]
                if s.empty:
                    continue
                # ---- per species x model ---------------------------------
                for key, nm in names.items():
                    for sp in sorted(s[UNIT].dropna().unique()):
                        r = s[(s.model == key) & (s[UNIT] == sp)]
                        if r.empty:
                            continue
                        rho = float(r.spearman_true.mean())
                        auc = float(r.roc_auc.mean())
                        rc = (float(r.oracle_spearman_infoloss.mean())
                              if proc == 'pointwise'
                              and 'oracle_spearman_infoloss' in r
                              and r.oracle_spearman_infoloss.notna().any()
                              else (float(r.ceiling_config_patch.mean())
                                    if proc == 'configurational'
                                    and r.ceiling_config_patch.notna().any()
                                    else np.nan))
                        if cz is not None:
                            m = cz[(cz.arm == arm) & (cz.level == lv)
                                   & (cz.species_idx == sp)]
                            ac = float(m.auc_ceiling_coarse.mean()) if len(m) \
                                else np.nan
                        else:
                            ac = float(r.auc_oracle.mean())
                        fl = float(auc_floor(r.n_te.mean(),
                                             r.real_prev_te.mean()))
                        rows_sp.append(dict(
                            process=proc, experiment=exp, level=lv, arm=arm,
                            species_no=int(sp),
                            seed=int(r.species_seed.iloc[0]), model=nm,
                            rho=rho, auc=auc, rho_ceiling=rc, auc_ceiling=ac,
                            no_skill=bool(np.isfinite(auc) and auc < fl),
                            auc_chance_floor=fl,
                            n_train=int(r.n_tr.iloc[0]),
                            n_val=int(r.n_vl.iloc[0]),
                            n_test=int(r.n_te.iloc[0])))
                # ---- per CNN initialisation -------------------------------
                u = unlearned(s)
                for _, r in u.iterrows():
                    rows_in.append(dict(
                        process=proc, experiment=exp, level=lv, arm=arm,
                        species_no=int(r[UNIT]), seed=int(r.species_seed),
                        init=int(r.cnn_init), rho=float(r.spearman_true),
                        auc=float(r.roc_auc),
                        best_val_auc=float(r.val_auc_best),
                        learned=bool(r.learned)))
                # ---- the condition ----------------------------------------
                base = 'rf' if proc == 'pointwise' else 'rf_patch_summary'
                hi = s[s.model == CN].groupby(UNIT)['spearman_true'].mean()
                lo = s[s.model == base].groupby(UNIT)['spearman_true'].mean()
                con = (hi - lo).dropna()
                sd = pooled_sd(s)
                cmean = float(con.mean()) if len(con) else np.nan
                nbad = int((~u.learned).sum()) if len(u) else 0
                row = dict(process=proc, experiment=exp, level=lv, arm=arm,
                           n_species=int(s[s.model == base][UNIT].nunique()))
                for key, nm in names.items():
                    t = s[s.model == key].groupby(UNIT)
                    row[f'rho_{nm}'] = float(t['spearman_true'].mean().mean())
                    row[f'auc_{nm}'] = float(t['roc_auc'].mean().mean())
                    au = float(t['roc_auc'].mean().mean())
                    fl = float(auc_floor(s.n_te.mean(), s.real_prev_te.mean()))
                    row[f'no_skill_{nm}'] = bool(np.isfinite(au) and au < fl)
                row['rho_ceiling'] = (
                    float(s.oracle_spearman_infoloss.mean())
                    if proc == 'pointwise'
                    and 'oracle_spearman_infoloss' in s
                    and s.oracle_spearman_infoloss.notna().any()
                    else (float(s.ceiling_config_patch.mean())
                          if proc == 'configurational'
                          and s.ceiling_config_patch.notna().any() else np.nan))
                if cz is not None:
                    m = cz[(cz.arm == arm) & (cz.level == lv)]
                    row['auc_ceiling'] = float(m.auc_ceiling_coarse.mean())
                else:
                    row['auc_ceiling'] = float(
                        s[s.model == 'rf'].groupby(UNIT)['auc_oracle']
                        .mean().mean())
                row.update(
                    contrast=cmean, contrast_baseline=NAME_POINT.get(base,
                                                                     names[base]),
                    pooled_init_sd=sd,
                    abs_contrast_over_sd=(abs(cmean) / sd
                                          if np.isfinite(sd) and sd > 0
                                          else np.nan),
                    within_sd=bool(np.isfinite(sd) and abs(cmean) <= sd),
                    cnn_fits_unlearned=nbad, cnn_fits_total=int(len(u)),
                    unlearned=bool(len(u) and nbad / len(u) > 0.5),
                    auc_chance_floor=float(auc_floor(s.n_te.mean(),
                                                     s.real_prev_te.mean())))
                rows_cond.append(row)
    return (pd.DataFrame(rows_sp), pd.DataFrame(rows_in),
            pd.DataFrame(rows_cond))


# =============================================================================
SRC = {'8j_alpha_grf': 'random field', '8j_alpha_landuse': 'forest map'}


def blend(D):
    d = D['blend']
    rows_sp, rows_cond = [], []
    for sexp, srcname in SRC.items():
        g = d[d.sub_exp == sexp]
        for arm in sorted(g.arm.unique()):
            for a in sorted(g.alpha.dropna().unique()):
                s = g[(g.arm == arm) & np.isclose(g.alpha, a)]
                u = unlearned(s)
                for key, nm in NAME_CFG.items():
                    for sp in sorted(s[UNIT].unique()):
                        r = s[(s.model == key) & (s[UNIT] == sp)]
                        if r.empty:
                            continue
                        ui = u[u[UNIT] == sp]
                        rows_sp.append(dict(
                            alpha=float(a), w_source=srcname, arm=arm,
                            species_no=int(sp),
                            seed=int(r.species_seed.iloc[0]), model=nm,
                            rho=float(r.spearman_true.mean()),
                            auc=float(r.roc_auc.mean()),
                            rho_ceiling_patch=float(
                                r.ceiling_config_patch.mean()),
                            auc_ceiling_labelnoise=float(r.auc_oracle.mean()),
                            cnn_inits_unlearned=(int((~ui.learned).sum())
                                                 if key == CN and len(ui)
                                                 else np.nan),
                            cnn_inits_total=(int(len(ui)) if key == CN and
                                             len(ui) else np.nan)))
                # ---- condition-level ---------------------------------------
                hi = s[s.model == CN].groupby(UNIT)['spearman_true'].mean()
                lo = s[s.model == 'rf_patch_summary'].groupby(
                    UNIT)['spearman_true'].mean()
                con = (hi - lo).dropna()
                ceil = s.groupby(UNIT)['ceiling_config_patch'].mean()
                for key, nm in NAME_CFG.items():
                    t = s[s.model == key].groupby(UNIT)
                    rho = t['spearman_true'].mean()
                    share = (rho / ceil).dropna()
                    row = dict(
                        alpha=float(a), w_source=srcname, arm=arm, model=nm,
                        n_species=int(rho.notna().sum()),
                        rho_mean=float(rho.mean()), rho_median=float(rho.median()),
                        auc_mean=float(t['roc_auc'].mean().mean()),
                        auc_median=float(t['roc_auc'].mean().median()),
                        rho_ceiling_patch_mean=float(ceil.mean()),
                        rho_ceiling_patch_median=float(ceil.median()),
                        auc_ceiling_labelnoise_mean=float(
                            s[s.model == 'rf'].groupby(UNIT)['auc_oracle']
                            .mean().mean()),
                        auc_ceiling_labelnoise_median=float(
                            s[s.model == 'rf'].groupby(UNIT)['auc_oracle']
                            .mean().median()),
                        share_of_ceiling_mean=(float(share.mean())
                                               if len(share) else np.nan),
                        share_of_ceiling_median=(float(share.median())
                                                 if len(share) else np.nan),
                        cnn_fits_unlearned=int((~u.learned).sum()) if len(u) else 0,
                        cnn_fits_total=int(len(u)))
                    if key == CN:
                        row.update(
                            contrast_mean=float(con.mean()),
                            contrast_median=float(con.median()),
                            contrast_min=float(con.min()),
                            contrast_max=float(con.max()),
                            n_species_cnn_above_rfpatch=int((con > 0).sum()))
                    rows_cond.append(row)
    return pd.DataFrame(rows_sp), pd.DataFrame(rows_cond)


def blend_extra(D):
    d = D['blend']
    # ---- (c) alpha = 0 with the W channel vs the no-W pointwise baseline ----
    base = []
    for arm in sorted(d.arm.unique()):
        for key, nm in NAME_CFG.items():
            a = (d[(d.sub_exp == '8j_pointwise_baseline') & (d.arm == arm)
                   & (d.model == key)].groupby(UNIT)['spearman_true'].mean())
            b = (d[(d.sub_exp == '8j_alpha_grf') & np.isclose(d.alpha, 0.0)
                   & (d.arm == arm) & (d.model == key)]
                 .groupby(UNIT)['spearman_true'].mean())
            if a.empty or b.empty:
                continue
            sh = sorted(set(a.index) & set(b.index))
            diff = np.array([b[s] - a[s] for s in sh])
            base.append(dict(
                arm=arm, model=nm, n_species=len(sh),
                rho_no_W_mean=float(a.mean()), rho_no_W_median=float(a.median()),
                rho_with_W_mean=float(b.mean()),
                rho_with_W_median=float(b.median()),
                difference_mean=float(diff.mean()),
                difference_median=float(np.median(diff)),
                difference_max_abs=float(np.abs(diff).max())))
    # ---- (d) correlation length at alpha = 1 -------------------------------
    sw = d[d.sub_exp == '8j_corrlen_grf_alpha1']
    a1 = d[(d.sub_exp == '8j_alpha_grf') & np.isclose(d.alpha, 1.0)]
    cl = pd.concat([sw, a1], ignore_index=True)
    corr = []
    for arm in sorted(cl.arm.unique()):
        for L in sorted(cl.corr_len_m.unique()):
            s = cl[(cl.arm == arm) & np.isclose(cl.corr_len_m, L)]
            u = unlearned(s)
            hi = s[s.model == CN].groupby(UNIT)['spearman_true'].mean()
            lo = s[s.model == 'rf_patch_summary'].groupby(
                UNIT)['spearman_true'].mean()
            con = (hi - lo).dropna()
            corr.append(dict(
                corr_len_m=float(L), arm=arm, n_species=len(hi),
                from_alpha_sweep=bool(np.isclose(L, 300.0)),
                rho_cnn_mean=float(hi.mean()), rho_cnn_median=float(hi.median()),
                rho_rfpatch_mean=float(lo.mean()),
                rho_rfpatch_median=float(lo.median()),
                contrast_mean=float(con.mean()),
                contrast_median=float(con.median()),
                contrast_min=float(con.min()), contrast_max=float(con.max()),
                cnn_fits_unlearned=int((~u.learned).sum()) if len(u) else 0,
                cnn_fits_total=int(len(u))))
    # ---- (e) primary vs oracle validation ----------------------------------
    orc = []
    pairs = [(1.0, '8j_alpha_grf', '8j_oracleval_alpha1'),
             (0.0, '8j_pointwise_baseline', '8j_oracleval_pointwise')]
    for a, prim_exp, orc_exp in pairs:
        for arm in sorted(d.arm.unique()):
            p = d[(d.sub_exp == prim_exp) & (d.arm == arm) & (d.model == CN)]
            if prim_exp == '8j_alpha_grf':
                p = p[np.isclose(p.alpha, a)]
            o = d[(d.sub_exp == orc_exp) & (d.arm == arm) & (d.model == CN)]
            pm = p.groupby(UNIT)['spearman_true'].mean()
            om = o.groupby(UNIT)['spearman_true'].mean()
            sh = sorted(set(pm.index) & set(om.index))
            if not sh:
                continue
            ch = np.array([om[s] - pm[s] for s in sh])
            orc.append(dict(
                alpha=a, arm=arm, n_species=len(sh),
                rho_primary_mean=float(pm[sh].mean()),
                rho_primary_median=float(pm[sh].median()),
                rho_oracleval_mean=float(om[sh].mean()),
                rho_oracleval_median=float(om[sh].median()),
                change_mean=float(ch.mean()), change_median=float(np.median(ch)),
                change_min=float(ch.min()), change_max=float(ch.max()),
                rho_primary_min=float(pm[sh].min()),
                rho_primary_max=float(pm[sh].max()),
                rho_oracleval_min=float(om[sh].min()),
                rho_oracleval_max=float(om[sh].max())))
    return pd.DataFrame(base), pd.DataFrame(corr), pd.DataFrame(orc)


# =============================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    D = load_all()
    sp, ini, cond = sensitivity(D)
    bsp, bcond = blend(D)
    bbase, bcorr, borc = blend_extra(D)
    tables = {
        'sens_species.csv': sp, 'sens_cnn_inits.csv': ini,
        'sens_conditions.csv': cond, 'blend_species.csv': bsp,
        'blend_conditions.csv': bcond, 'blend_baseline.csv': bbase,
        'blend_corrlen.csv': bcorr, 'blend_oracleval.csv': borc,
    }
    for fn, t in tables.items():
        t.to_csv(os.path.join(OUT, fn), index=False)
        print(f'{fn:<24} {len(t):>5} rows x {t.shape[1]:>2} cols')
    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    main()
