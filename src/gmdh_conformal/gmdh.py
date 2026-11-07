"""GMDH layers and transformation, matching the supplied revision script."""

import numpy as np
from sklearn.linear_model import Ridge

from .config import GMDH_K, RIDGE_ALPHA


def gmdh_layer(X, y, rng):
    n, d = X.shape
    idx = rng.permutation(n)
    n_tr = min(max(int(n * 0.7), 1), n - 1)
    tr, va = idx[:n_tr], idx[n_tr:]
    cands, conds = [], []
    for i in range(d):
        for j in range(i, d):
            u, v = X[:, i], X[:, j]
            F = np.column_stack([np.ones(n), u, v, u**2, u * v, v**2])
            conds.append(float(np.linalg.cond(F.T @ F + RIDGE_ALPHA * np.eye(6))))
            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(F[tr], y[tr])
            cands.append((float(np.mean((y[va] - model.predict(F[va])) ** 2)), i, j, model.coef_.copy()))
    cands.sort(key=lambda candidate: candidate[0])
    selected = cands[: min(GMDH_K, len(cands))]
    out = np.column_stack(
        [
            np.column_stack([np.ones(n), X[:, i], X[:, j], X[:, i] ** 2, X[:, i] * X[:, j], X[:, j] ** 2]) @ coef
            for _, i, j, coef in selected
        ]
    )
    return out, selected, np.array(conds)


def gmdh_transform(X, layers):
    F = X.copy()
    for nodes in layers:
        F = np.column_stack(
            [
                np.column_stack([np.ones(len(F)), F[:, i], F[:, j], F[:, i] ** 2, F[:, i] * F[:, j], F[:, j] ** 2])
                @ coef
                for _, i, j, coef in nodes
            ]
        )
    return F


def fit_gmdh(X, y, depth, seed):
    F, layers, conditions = X.copy(), [], []
    for layer_index in range(depth):
        F, selected, cond = gmdh_layer(F, y, np.random.default_rng(seed + 1000 * (layer_index + 1)))
        layers.append(selected)
        conditions.append(cond)
    return layers, Ridge(alpha=RIDGE_ALPHA).fit(F, y), conditions
