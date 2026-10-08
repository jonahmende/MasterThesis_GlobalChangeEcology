"""species.csv -- every virtual species the thesis reports on.

Reproduces the pipeline's own niche parameters (make_vs + _s8_config_niche) and
its own suitability definition (_s8_suit_fn: a 2-D Gaussian on PC1/PC2 raised
to `power`, then rescaled so the maximum over valid pixels is 1), evaluated on
all 35.6 M valid pixels, so the per-band means are population means and not a
sample.

species_no is the pipeline's own species_idx: it is the POSITION IN THE
ADMITTED LIST that _s8_species_iter returns, and it STARTS AT 0. The
sensitivity experiments take the first three of that list, the blend experiment
the first eight, so sensitivity species 0/1/2 ARE blend species 0/1/2.

Environment: wolf_sdm
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np
import pandas as pd

from _vs_env import notebook_env, band_pixels, FIGURES_DIR
from export_results_revision import OUT, read, V

SENS_SEEDS = [42, 55838, 63757]


def main():
    os.makedirs(OUT, exist_ok=True)
    g = notebook_env(with_config=True)
    env = g['env_scaled_100m']
    vr, vc, pc1, (t40, t55, t70) = band_pixels(g)
    pc = g['_pca3b'].transform(g['_sc3b'].transform(
        env[vr, vc, :].astype(np.float32)))
    pc1, pc2 = pc[:, 0], pc[:, 1]
    band = np.where(pc1 < t40, 'train',
                    np.where(pc1 < t55, 'val',
                             np.where(pc1 < t70, 'test', 'beyond')))
    print(f'valid pixels: {len(pc1):,}   PC1 cuts {t40:.6f} / {t55:.6f} / {t70:.6f}')

    # which species_idx carries which seed, straight from the result files
    seen = {}
    for stem, exps in (('8_two_regime', 'sensitivity'),
                       ('8c_fixed_truth', 'sensitivity'),
                       ('8i_alpha_axes', 'sensitivity'),
                       ('8j_species_ensemble', 'blend')):
        d = read(stem)
        for idx, sd in d[['species_idx', 'species_seed']].dropna().drop_duplicates().values:
            seen.setdefault(int(sd), {'idx': int(idx), 'exp': set()})['exp'].add(exps)
    rows = []
    for sd, info in sorted(seen.items(), key=lambda kv: kv[1]['idx']):
        r = g['make_vs'](sd, mode='pca', occ='probabilistic',
                         k=g['_S8_OCC_K'], prevalence=g['_S8_OCC_PREV'])
        meta = {**r[2], **g['_s8_config_niche'](sd)}
        mu1, mu2 = float(meta['mus_pc'][0]), float(meta['mus_pc'][1])
        s1, s2 = float(meta['sigs_pc'][0]), float(meta['sigs_pc'][1])
        p = float(meta.get('power', 1.0))
        s = np.exp(-0.5 * (((pc1 - mu1) / s1) ** 2 + ((pc2 - mu2) / s2) ** 2)) ** p
        mx = float(np.nanmax(s))
        if mx > 0:
            s = s / mx
        pct = float((pc1 < mu1).mean() * 100.0)
        where = ('below the 40th' if mu1 < t40 else
                 '40th-55th' if mu1 < t55 else
                 '55th-70th' if mu1 < t70 else 'above the 70th')
        rows.append(dict(
            species_no=info['idx'], seed=sd,
            experiments='+'.join(sorted(info['exp'])),
            in_sensitivity=sd in SENS_SEEDS, in_blend=True,
            mu1=mu1, mu2=mu2, sigma1=s1, sigma2=s2, power=p,
            sigma1_effective=s1 / np.sqrt(max(p, 1e-9)),
            mu1_pc1_percentile=pct, mu1_band=where,
            mean_suitability_train=float(s[band == 'train'].mean()),
            mean_suitability_val=float(s[band == 'val'].mean()),
            mean_suitability_test=float(s[band == 'test'].mean()),
            mean_suitability_beyond_test=float(s[band == 'beyond'].mean()),
            mu_E=float(meta['mu_E']), sigma_E=float(meta['sig_E'])))
        print(f"  sp{info['idx']} seed {sd:<7} mu1={mu1:+.4f} "
              f"pct={pct:5.1f} {where:<15} "
              f"S tr/val/te = {rows[-1]['mean_suitability_train']:.4f} / "
              f"{rows[-1]['mean_suitability_val']:.4f} / "
              f"{rows[-1]['mean_suitability_test']:.4f}")
    df = pd.DataFrame(rows).sort_values('species_no')
    df.to_csv(os.path.join(OUT, 'species.csv'), index=False)
    print(f"\nspecies.csv  {len(df)} rows x {df.shape[1]} cols -> {OUT}")


if __name__ == '__main__':
    main()
