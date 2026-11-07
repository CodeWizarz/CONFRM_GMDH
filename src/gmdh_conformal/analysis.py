"""Detailed statistical analysis and figures, ported from the supplied script."""

import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from .config import DATASETS, N_SPLITS, THRESHOLDS

PUBLISHED_R0 = {
    ("Yacht", "GMDH-d2"): (0.919, 0.396),
    ("Energy", "GMDH-d2"): (0.891, 1.003),
    ("Concrete", "GMDH-d2"): (0.926, 1.843),
    ("Airfoil", "GMDH-d2"): (0.898, 2.029),
    ("California", "GMDH-d2"): (0.905, 2.115),
    ("Yacht", "EBM"): (0.828, 0.084),
    ("California", "EBM"): (0.903, 1.221),
    ("Concrete", "CQR-XGB"): (0.891, 1.654),
    ("Airfoil", "MLP"): (0.905, 0.943),
}


def exact_perm_spearman(x, y):
    obs = spearmanr(x, y).correlation
    perms = list(itertools.permutations(range(len(y))))
    null = np.array([spearmanr(x, np.asarray(y)[list(p)]).correlation for p in perms])
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), len(perms)


def analyse(args):
    D = pd.concat([pd.read_csv(f) for f in glob.glob(f"{args.results_dir}/folds_*.csv")])
    P = pd.concat([pd.read_csv(f) for f in glob.glob(f"{args.results_dir}/pnf_*_r*.csv")])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    order = ["GMDH-d2", "GMDH-d1", "Poly2-Ridge", "Poly2-RidgeCV", "RF", "XGB", "EBM", "CQR-XGB", "CQR-HGB", "MLP"]
    D = D[D.dataset.isin(DATASETS)]
    nf = D.groupby(["dataset", "rep"]).fold.nunique()
    np_ = P.groupby(["dataset", "rep"]).fold.nunique()
    datasets = tuple(name.strip() for name in args.datasets.split(",") if name.strip()) if args.datasets else DATASETS
    expected = [(dataset, repeat) for dataset in datasets for repeat in range(args.repeats)]
    bad = [key for key in expected if nf.get(key, 0) != N_SPLITS or np_.get(key, 0) != N_SPLITS]
    print("Completeness check:", "all repeats complete" if not bad else f"INCOMPLETE {bad} (rerun these)")
    import scipy
    import sklearn

    print(
        f"Environment: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, sklearn {sklearn.__version__}, scipy {scipy.__version__}"
    )
    split_counts = D[["dataset", "rep", "fold"]].drop_duplicates().groupby("dataset").size().to_dict()
    print(f"Splits per dataset: {split_counts}")

    print("\n0. REPRODUCTION CHECK (repeat 0 vs published Table 1)")
    for (ds, mth), (c, w) in PUBLISHED_R0.items():
        g = D[(D.dataset == ds) & (D.method == mth) & (D.rep == 0)]
        if len(g):
            print(
                f"   {ds:10s} {mth:8s} published cov {c:.3f} width {w:.3f} | now cov {g['cov'].mean():.3f} width {g.width.mean():.3f}"
            )

    for metric, lab in (
        ("cov", "COVERAGE"),
        ("width", "WIDTH"),
        ("winkler", "WINKLER SCORE (lower is better)"),
        ("mse", "POINT MSE"),
    ):
        t = D.groupby(["dataset", "method"])[metric].agg(["mean", "std"]).unstack("method")
        fmt = (
            t["mean"].reindex(columns=order).round(3).astype(str)
            + " ± "
            + t["std"].reindex(columns=order).round(3).astype(str)
        )
        print(f"\n1. {lab}: mean ± sd over all splits")
        print(fmt.reindex(DATASETS).to_string())
    print("\n2. WINKLER RANK of each method per dataset (1 = best)")
    print(
        D.groupby(["dataset", "method"])
        .winkler.mean()
        .unstack()
        .reindex(columns=order)
        .rank(axis=1)
        .reindex(DATASETS)
        .to_string()
    )

    print("\n3. GMDH-d2 versus baselines: Winkler score, paired over splits.")
    print("   Test: corrected resampled t-test for repeated k-fold CV (Nadeau and Bengio 2003), which accounts for the")
    print("   overlap between splits; the naive paired p-value is shown only for reference and is too optimistic.")
    from scipy.stats import t as tdist

    for ds in DATASETS:
        sub = D[D.dataset == ds]
        g = sub.pivot_table(index=["rep", "fold"], columns="method", values="winkler")
        if g.empty or "GMDH-d2" not in g:
            continue
        ratio = float((sub.n_test / (sub.n_train + sub.n_cal)).mean())
        for base in ("GMDH-d1", "Poly2-Ridge", "Poly2-RidgeCV", "EBM"):
            if base not in g:
                continue
            d = (g["GMDH-d2"] - g[base]).dropna().values
            J = len(d)
            if J < 3:
                continue
            se = np.sqrt((1 / J + ratio) * d.var(ddof=1))
            tstat = d.mean() / se if se > 0 else np.nan
            p_corr = 2 * tdist.sf(abs(tstat), J - 1)
            p_naive = wilcoxon(d).pvalue if J > 5 else float("nan")
            print(
                f"   {ds:10s} GMDH-d2 minus {base:14s}: mean diff {d.mean():+.3f} (GMDH better in {(d < 0).mean():.0%} of {J} splits); "
                f"corrected p={p_corr:.4f}  [naive Wilcoxon p={p_naive:.4f}]"
            )

    print(
        "\n4. CQR DIAGNOSTICS (mean over splits): raw quantile band before conformal correction, q_hat, crossing share"
    )
    print(
        D[D.method.str.startswith("CQR")]
        .groupby(["dataset", "method"])[["raw_band_width", "raw_band_cov_cal", "q", "crossing_share", "width", "mse"]]
        .mean()
        .round(3)
        .to_string()
    )
    print("\n5. MLP DIAGNOSTICS: iterations and width variability")
    print(
        D[D.method == "MLP"]
        .groupby("dataset")[["n_iter", "best_val", "width"]]
        .agg(["mean", "std", "min", "max"])
        .round(3)
        .to_string()
    )

    print("\n6. PNF: per layer and threshold (depth 2, mean over splits)")
    P2 = P[P.depth == 2]
    cols = [c for c in P2.columns if c.startswith("pnf_L")]
    print(P2.groupby("dataset")[cols + ["cond_median"]].mean().reindex(DATASETS).round(3).T.to_string())
    w = D[D.method == "GMDH-d2"].groupby("dataset").width.mean().reindex(DATASETS)
    print("\n7. PNF-WIDTH ASSOCIATION (dataset level, n = 5): Spearman rho with exact permutation p-value")
    for thr in THRESHOLDS:
        pnf = P2.groupby("dataset")[f"pnf_mean_{thr:.0e}"].mean().reindex(DATASETS)
        if pnf.isna().any() or len(pnf) < 3:
            continue
        rho, p, nperm = exact_perm_spearman(pnf.values, 1 / w.values)
        print(
            f"   threshold {thr:.0e}: PNF {np.round(pnf.values, 3).tolist()}  rho={rho:.3f}  exact p={p:.3f} ({nperm} permutations)"
        )

    print("\n8. DEPTH SENSITIVITY (all splits)")
    print(
        D[D.method.str.startswith("GMDH")]
        .groupby(["dataset", "method"])[["cov", "width", "winkler", "mse"]]
        .mean()
        .round(4)
        .reindex(DATASETS, level=0)
        .to_string()
    )
    print("   California depth-2 MSE per split (to show the extrapolation outliers):")
    print("  ", np.round(np.sort(D[(D.dataset == "California") & (D.method == "GMDH-d2")].mse.values), 3).tolist())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.fig_dir, exist_ok=True)
    for metric in ("winkler", "width", "cov"):
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 4))
        for ax, ds in zip(axes, DATASETS):
            g = D[D.dataset == ds]
            ax.boxplot([g[g.method == m][metric].values for m in order], showfliers=False)
            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels(order, rotation=90, fontsize=7)
            ax.set_title(ds)
            if metric == "cov":
                ax.axhline(0.9, ls="--", color="grey", lw=0.8)
        fig.tight_layout()
        fig.savefig(f"{args.fig_dir}/box_{metric}.png", dpi=200)
        plt.close(fig)
    print(f"\nFigures: {args.fig_dir}/box_winkler.png, {args.fig_dir}/box_width.png, {args.fig_dir}/box_cov.png")
