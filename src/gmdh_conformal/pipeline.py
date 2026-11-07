"""Repeat/fold orchestration and compact analysis of saved split-level output."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

from .config import DATASETS, N_SPLITS, SEED
from .data import load_dataset
from .experiment import run_fold


def run_experiments(datasets=DATASETS, repeats=5, output_dir="outputs", synthetic=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        X, y = load_dataset(name, synthetic=synthetic)
        for rep in range(repeats):
            fold_path = out / f"folds_{name}_r{rep}.csv"
            pnf_path = out / f"pnf_{name}_r{rep}.csv"
            if fold_path.exists() or pnf_path.exists():
                raise FileExistsError(f"Refusing to overwrite a partial or existing result: {fold_path} / {pnf_path}")
            results, pnf_results = [], []
            splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
            for fold_no, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
                r, p = run_fold(X, y, name, rep, fold_no, train_idx, test_idx)
                results.extend(r)
                pnf_results.extend(p)
            pd.DataFrame(results).to_csv(fold_path, index=False)
            pd.DataFrame(pnf_results).to_csv(pnf_path, index=False)
            print(f"Wrote {fold_path} and {pnf_path}")


def summarize_results(output_dir="outputs"):
    """Print per-dataset/method mean and standard deviation for stored split metrics."""
    files = sorted(Path(output_dir).glob("folds_*.csv"))
    if not files:
        raise FileNotFoundError(f"No folds_*.csv result files found in {output_dir}")
    results = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    metrics = [column for column in ("cov", "width", "winkler", "mse") if column in results]
    print(results.groupby(["dataset", "method"])[metrics].agg(["mean", "std"]).round(4).to_string())
