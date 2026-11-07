"""Yacht GMDH equations and EBM term importance export."""

import numpy as np
import sympy as sp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .config import CAL_FRAC, FEATURES, N_SPLITS, SEED
from .data import load_dataset
from .gmdh import fit_gmdh


def interpret():
    name = "Yacht"
    X, y = load_dataset(name)
    tr, _ = next(iter(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X)))
    rng = np.random.default_rng(SEED + 10000)
    perm = rng.permutation(len(tr))
    n_cal = int(len(tr) * CAL_FRAC)
    trn = tr[perm[n_cal:]]
    sx = StandardScaler().fit(X[trn])
    Xtr = sx.transform(X[trn])
    ym, ys = y[trn].mean(), y[trn].std()
    ytr = (y[trn] - ym) / ys
    names = FEATURES.get(name, [f"x{i}" for i in range(X.shape[1])])
    syms = sp.symbols([f"z{i + 1}" for i in range(X.shape[1])])
    print(
        f"Interpretability export: {name}, repeat 0, fold 1 training split (n={len(trn)}). z_i = standardised predictor i:"
    )
    for k, (s, n) in enumerate(zip(syms, names)):
        print(f"   {s} = ({n} - {sx.mean_[k]:.4f}) / {sx.scale_[k]:.4f}")
    print(f"   target: resistance = {ym:.4f} + {ys:.4f} * (model output)")
    print("   Self-pair nodes (same input twice) are shown in their exact 3-term form a0 + a1*u + a2*u^2,")
    print("   obtained by summing the duplicated basis coefficients; predictions are unchanged.")
    for depth in (1, 2):
        layers, final, _ = fit_gmdh(Xtr, ytr, depth, SEED + 101)
        print(f"\n=== GMDH depth {depth} ===")
        inputs = list(syms)
        for L, nodes in enumerate(layers, start=1):
            outs = []
            for k, (mse, i, j, c) in enumerate(nodes, start=1):
                u, v = inputs[i], inputs[j]
                expr = c[0] + c[1] * u + c[2] * v + c[3] * u**2 + c[4] * u * v + c[5] * v**2
                lab = f"L{L}N{k}"
                inp = f"{names[i]}, {names[j]}" if L == 1 else f"L{L - 1}N{i + 1}, L{L - 1}N{j + 1}"
                if i == j:
                    print(
                        f"   {lab} (input: {inp.split(', ')[0]} only; internal val. MSE {mse:.4f}): "
                        f"{c[0]:+.3f} {c[1] + c[2]:+.3f}*a {c[3] + c[4] + c[5]:+.3f}*a^2"
                    )
                else:
                    print(
                        f"   {lab} (inputs: {inp}; internal val. MSE {mse:.4f}): "
                        f"{c[0]:+.3f} {c[1]:+.3f}*a {c[2]:+.3f}*b {c[3]:+.3f}*a^2 {c[4]:+.3f}*a*b {c[5]:+.3f}*b^2"
                    )
                outs.append(sp.expand(expr))
            inputs = outs
        full = sp.expand(final.intercept_ + sum(float(w) * o for w, o in zip(final.coef_, inputs)))
        poly = sp.Poly(full, *syms)
        terms = sorted(poly.terms(), key=lambda t: -abs(float(t[1])))
        print(f"   final ridge weights: {np.round(final.coef_, 3).tolist()}, intercept {final.intercept_:+.3f}")
        print(f"   expanded polynomial in standardised predictors: degree {poly.total_degree()}, {len(terms)} terms")
        print("   10 largest terms:")
        for mon, coef in terms[:10]:
            desc = " * ".join(f"{names[i]}^{e}" if e > 1 else names[i] for i, e in enumerate(mon) if e) or "intercept"
            print(f"      {float(coef):+9.4f}  {desc}")
        used = sorted({names[i] for nodes in layers[:1] for _, i, j, _ in nodes for i in (i, j)})
        print(f"   predictors entering layer 1: {used}")
    from interpret.glassbox import ExplainableBoostingRegressor

    ebm = ExplainableBoostingRegressor(random_state=SEED + 104, feature_names=names).fit(Xtr, ytr)
    imp = sorted(zip(ebm.term_names_, ebm.term_importances()), key=lambda t: -t[1])
    print("\n=== EBM term importances (same split) ===")
    for n, v in imp[:10]:
        print(f"   {v:.4f}  {n}")
