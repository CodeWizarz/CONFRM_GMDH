"""Validation of archived split outputs without changing the supplied results."""

from pathlib import Path

import pandas as pd

from .config import DATASETS, N_SPLITS

EXPECTED_METHODS = {
    "GMDH-d1",
    "GMDH-d2",
    "Poly2-Ridge",
    "Poly2-RidgeCV",
    "RF",
    "XGB",
    "EBM",
    "CQR-XGB",
    "CQR-HGB",
    "MLP",
}
EXPECTED_R0 = {
    ("Yacht", "GMDH-d2"): (0.919, 0.396),
    ("Energy", "GMDH-d2"): (0.891, 1.003),
    ("Concrete", "GMDH-d2"): (0.926, 1.843),
    ("Airfoil", "GMDH-d2"): (0.898, 2.029),
    ("California", "GMDH-d2"): (0.905, 2.115),
}


def validate_results(results_dir="provenance/results/supplement_20261107/gmdh_revision/rev_out"):
    root = Path(results_dir)
    errors = []
    files = sorted(root.glob("folds_*.csv"))
    if not files:
        raise FileNotFoundError(f"No fold results found in {root}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    required = {"dataset", "rep", "fold", "method", "cov", "width", "winkler", "mse"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        errors.append(f"missing fold columns: {sorted(missing_columns)}")
    for dataset in DATASETS:
        for rep in range(5):
            group = data[(data.dataset == dataset) & (data.rep == rep)]
            fold_count = group.fold.nunique()
            if fold_count != N_SPLITS:
                errors.append(f"{dataset} repeat {rep}: expected {N_SPLITS} folds, found {fold_count}")
            for fold in range(1, N_SPLITS + 1):
                observed = set(group.loc[group.fold == fold, "method"])
                if observed != EXPECTED_METHODS:
                    errors.append(f"{dataset} repeat {rep} fold {fold}: unexpected method set {sorted(observed)}")
    pnf_files = sorted(root.glob("pnf_*_r*.csv"))
    pnf = pd.concat([pd.read_csv(path) for path in pnf_files], ignore_index=True) if pnf_files else pd.DataFrame()
    for dataset in DATASETS:
        for rep in range(5):
            group = pnf[(pnf.dataset == dataset) & (pnf.rep == rep)] if not pnf.empty else pnf
            if group.fold.nunique() != N_SPLITS or set(group.depth) != {1, 2}:
                errors.append(f"{dataset} repeat {rep}: PNF output missing folds or depths")
    checks = []
    for key, (expected_cov, expected_width) in EXPECTED_R0.items():
        ds, method = key
        group = data[(data.dataset == ds) & (data.method == method) & (data.rep == 0)]
        if len(group) != N_SPLITS:
            errors.append(f"reproduction check {key}: expected {N_SPLITS} rows, found {len(group)}")
            continue
        observed = (float(group["cov"].mean()), float(group["width"].mean()))
        ok = abs(observed[0] - expected_cov) < 0.00051 and abs(observed[1] - expected_width) < 0.00051
        checks.append((key, observed, ok))
        if not ok:
            errors.append(f"reproduction check {key}: observed {observed}, expected {(expected_cov, expected_width)}")
    print(f"Fold result files: {len(files)}; PNF files: {len(pnf_files)}; rows: {len(data)}")
    print(f"Repeat/fold/method structure: {'PASS' if not errors else 'CHECK FAILED'}")
    print(f"Published GMDH repeat-0 coverage/width checks: {sum(ok for _, _, ok in checks)}/{len(checks)} passed")
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "fold_files": len(files),
        "pnf_files": len(pnf_files),
        "rows": len(data),
        "reproduction_checks": len(checks),
    }
