"""One outer split's model training and evaluation."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import ALPHA, CAL_FRAC, PNF_THR, RIDGE_ALPHA, SEED, THRESHOLDS
from .gmdh import fit_gmdh, gmdh_transform
from .metrics import absres, conformal_q, interval_metrics


def run_fold(X, y, name, rep, fold_no, tr_idx, te_idx):
    seed_base = SEED + 1000 * rep
    rng = np.random.default_rng(seed_base + fold_no * 10000)
    perm = rng.permutation(len(tr_idx))
    n_cal = min(max(int(len(tr_idx) * CAL_FRAC), 1), len(tr_idx) - 1)
    cal_i, trn_i = tr_idx[perm[:n_cal]], tr_idx[perm[n_cal:]]
    sx = StandardScaler().fit(X[trn_i])
    ym, ys = y[trn_i].mean(), y[trn_i].std()
    Xtr, Xca, Xte = sx.transform(X[trn_i]), sx.transform(X[cal_i]), sx.transform(X[te_idx])
    ytr, yca, yte = (y[trn_i] - ym) / ys, (y[cal_i] - ym) / ys, (y[te_idx] - ym) / ys
    ms = seed_base + fold_no * 100 + 1
    rows, pnf_rows = [], []
    base = {"dataset": name, "rep": rep, "fold": fold_no, "n_train": len(trn_i), "n_cal": n_cal, "n_test": len(te_idx)}

    def add(method, result, **extra):
        rows.append({**base, "method": method, **result, **extra})

    for depth in (1, 2):
        layers, final, conds = fit_gmdh(Xtr, ytr, depth, ms)
        result = absres(
            final.predict(gmdh_transform(Xca, layers)), yca, final.predict(gmdh_transform(Xte, layers)), yte
        )
        pnf = {
            f"pnf_L{layer + 1}_{threshold:.0e}": float(np.mean(cond < threshold))
            for layer, cond in enumerate(conds)
            for threshold in THRESHOLDS
        }
        pnf_row = {
            **base,
            "depth": depth,
            **pnf,
            **{
                f"pnf_mean_{threshold:.0e}": float(np.mean([np.mean(cond < threshold) for cond in conds]))
                for threshold in THRESHOLDS
            },
            "cond_median": float(np.median(np.concatenate(conds))),
        }
        pnf_rows.append(pnf_row)
        add(f"GMDH-d{depth}", result, pnf=pnf_row[f"pnf_mean_{PNF_THR:.0e}"])

    poly = PolynomialFeatures(2, include_bias=False).fit(Xtr)
    Ptr, Pca, Pte = poly.transform(Xtr), poly.transform(Xca), poly.transform(Xte)
    model = Ridge(alpha=RIDGE_ALPHA).fit(Ptr, ytr)
    add("Poly2-Ridge", absres(model.predict(Pca), yca, model.predict(Pte), yte), n_coef=Ptr.shape[1] + 1)
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Ptr, ytr)
    add(
        "Poly2-RidgeCV",
        absres(model.predict(Pca), yca, model.predict(Pte), yte),
        ridge_alpha=float(model.alpha_),
        n_coef=Ptr.shape[1] + 1,
    )

    model = RandomForestRegressor(n_estimators=100, random_state=ms + 1).fit(Xtr, ytr)
    add("RF", absres(model.predict(Xca), yca, model.predict(Xte), yte))
    from xgboost import XGBRegressor

    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=ms + 2, verbosity=0).fit(
        Xtr, ytr
    )
    add("XGB", absres(model.predict(Xca), yca, model.predict(Xte), yte))
    from interpret.glassbox import ExplainableBoostingRegressor

    model = ExplainableBoostingRegressor(random_state=ms + 3).fit(Xtr, ytr)
    add("EBM", absres(model.predict(Xca), yca, model.predict(Xte), yte))

    lo = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=ALPHA / 2,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=ms + 4,
        verbosity=0,
    ).fit(Xtr, ytr)
    hi = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=1 - ALPHA / 2,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=ms + 4,
        verbosity=0,
    ).fit(Xtr, ytr)
    hgb_lo = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA / 2, random_state=ms + 6).fit(Xtr, ytr)
    hgb_hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - ALPHA / 2, random_state=ms + 6).fit(Xtr, ytr)
    for tag, (lo_model, hi_model) in (("CQR-XGB", (lo, hi)), ("CQR-HGB", (hgb_lo, hgb_hi))):
        lc, hc, lt, ht = lo_model.predict(Xca), hi_model.predict(Xca), lo_model.predict(Xte), hi_model.predict(Xte)
        qh = conformal_q(np.maximum(lc - yca, yca - hc))
        result = interval_metrics(lt - qh, ht + qh, yte)
        result.update(mse=float(np.mean((yte - (lt + ht) / 2) ** 2)), q=qh)
        add(
            tag,
            result,
            raw_band_width=float(np.mean(ht - lt)),
            raw_band_cov_cal=float(np.mean((lc <= yca) & (yca <= hc))),
            crossing_share=float(np.mean(ht < lt)),
        )

    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=ms + 5,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    ).fit(Xtr, ytr)
    add(
        "MLP",
        absres(model.predict(Xca), yca, model.predict(Xte), yte),
        n_iter=int(model.n_iter_),
        best_val=float(model.best_validation_score_),
    )
    return rows, pnf_rows
