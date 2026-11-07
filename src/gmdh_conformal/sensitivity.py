"""PNF condition-number sensitivity analysis from the supplied revision script."""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .analysis import exact_perm_spearman
from .config import CAL_FRAC, N_SPLITS, RIDGE_ALPHA, SEED, THRESHOLDS
from .data import load_dataset
from .gmdh import gmdh_layer


def pnfsens(args):
    """PNF sensitivity: condition numbers on the fitting subset only, with and without the ridge term."""
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for name in args.datasets.split(","):
        X, y = load_dataset(name, args.synthetic)
        for rep in range(args.repeats):
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
            for fold_no, (tr, te) in enumerate(kf.split(X), start=1):
                seed_base = SEED + 1000 * rep
                rng = np.random.default_rng(seed_base + fold_no * 10000)
                perm = rng.permutation(len(tr))
                n_cal = int(len(tr) * CAL_FRAC)
                trn = tr[perm[n_cal:]]
                sx = StandardScaler().fit(X[trn])
                F = sx.transform(X[trn])
                yy = (y[trn] - y[trn].mean()) / y[trn].std()
                ms = seed_base + fold_no * 100 + 1
                for L in range(2):
                    r = np.random.default_rng(ms + 1000 * (L + 1))
                    n = len(F)
                    idx = r.permutation(n)
                    fit = idx[: int(n * 0.7)]
                    reg, unreg = [], []
                    for i in range(F.shape[1]):
                        for j in range(
                            i + 1, F.shape[1]
                        ):  # bivariate nodes only (self-pairs are rank-deficient by design)
                            B = np.column_stack(
                                [
                                    np.ones(len(fit)),
                                    F[fit, i],
                                    F[fit, j],
                                    F[fit, i] ** 2,
                                    F[fit, i] * F[fit, j],
                                    F[fit, j] ** 2,
                                ]
                            )
                            unreg.append(np.linalg.cond(B))
                            reg.append(np.linalg.cond(B.T @ B + RIDGE_ALPHA * np.eye(6)))
                    rows.append(
                        dict(
                            dataset=name,
                            rep=rep,
                            fold=fold_no,
                            layer=L + 1,
                            **{f"pnf_fitsub_reg_{t:.0e}": float(np.mean(np.array(reg) < t)) for t in THRESHOLDS},
                            **{
                                f"pnf_fitsub_unreg_{t:.0e}": float(np.mean(np.array(unreg) < np.sqrt(t)))
                                for t in THRESHOLDS
                            },
                        )
                    )
                    F, _, _ = gmdh_layer(F, yy, np.random.default_rng(ms + 1000 * (L + 1)))
    Q = pd.DataFrame(rows)
    Q.to_csv(f"{args.output_dir}/pnf_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print("PNF computed on the fitting subset only (bivariate nodes). 'unreg' uses cond(B) against sqrt(threshold),")
    print("which is equivalent to cond(B^T B) against the threshold without the ridge term.")
    print(Q.groupby(["dataset", "layer"]).mean(numeric_only=True).drop(columns=["rep", "fold"]).round(3).to_string())
    D = pd.concat([pd.read_csv(f) for f in glob.glob(f"{args.output_dir}/folds_*.csv")])
    w = D[D.method == "GMDH-d2"].groupby("dataset").width.mean()
    for col in [c for c in Q.columns if c.startswith("pnf_")]:
        p = Q.groupby("dataset")[col].mean().reindex(w.index)
        if p.nunique() > 1 and len(p) >= 3:
            rho, pv, nperm = exact_perm_spearman(p.values, 1 / w.values)
            print(f"   {col:28s} rho={rho:+.3f}  exact p={pv:.3f} ({nperm} permutations)")
