"""
gmdh_revision.py - experiments for the Next Research major revision (NEXRES-D-26-03152).

Protocol identical to the Phase-1 corrected benchmark (leakage-free scaling, deterministic train/cal split,
exact conformal order statistic, alpha = 0.10), extended with:
  R2.1  degree-2 polynomial ridge + split conformal (lambda = 1 and lambda tuned by internal CV)
  R2.2  Winkler (interval) score for every method
  R2.4  repeated splits: R repeats x 5 folds (repeat 0 reproduces the published Phase-1 folds exactly);
        CQR diagnostics (raw quantile band width, q_hat) and a second CQR built on sklearn's
        HistGradientBoosting quantile loss; MLP diagnostics (iterations, early-stopping score)
  R2.5/R4.1/R4.2  PNF per layer and at condition thresholds 1e3, 1e4, 1e5, 1e6; exact permutation test
  Depth   GMDH depth 1 and depth 2 for all five datasets (the published Table 7 was only partly re-runnable)
  R2.3  interpretability export: node equations, expanded polynomial, EBM term importances (Yacht)

Colab:
  !pip -q install interpret xgboost openpyxl scipy sympy
  !python gmdh_revision.py run --repeats 5          # resumable; about 6-8 h for all datasets (EBM dominates)
  !python gmdh_revision.py run --repeats 5 --datasets Yacht,Energy   # run in pieces if you prefer
  !python gmdh_revision.py analyse > revision_report.txt; cat revision_report.txt
  !python gmdh_revision.py interpret > interpretability.txt; cat interpretability.txt
  !python gmdh_revision.py pnfsens > pnf_sensitivity.txt; cat pnf_sensitivity.txt   # GMDH only, a few minutes
Outputs: rev_out/folds_<dataset>_r<k>.csv, rev_out/pnf_<dataset>_r<k>.csv, figs/*.png, *.txt reports
"""
import argparse, glob, itertools, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
warnings.filterwarnings("ignore")

SEED, ALPHA, N_SPLITS, CAL_FRAC = 42, 0.10, 5, 0.18
GMDH_K, RIDGE_ALPHA, PNF_THR = 6, 1.0, 1e4
THRESHOLDS = (1e3, 1e4, 1e5, 1e6)
OUT = "rev_out"
DATASETS = ["Yacht", "Energy", "Concrete", "Airfoil", "California"]
FEATURES = {
    "Yacht": ["LCB position", "Prismatic coef.", "Length-displacement", "Beam-draught", "Length-beam", "Froude number"],
}


# ------------------------------------------------------------------ data (Phase-1 loaders)
def load_dataset(name, synthetic=False):
    if synthetic:                                                     # offline smoke test only
        rng = np.random.default_rng(len(name)); n = {"California": 3000}.get(name, 400); d = 6
        X = rng.normal(size=(n, d)); y = X[:, 0] ** 2 + X[:, 1] * X[:, 2] + 0.3 * rng.normal(size=n)
        return X, y
    if name == "Yacht":
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data", sep=r"\s+", header=None)
        return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(float)
    if name == "Energy":
        df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx")
        return df.iloc[:, :-2].values.astype(float), df.iloc[:, -2].values.astype(float)
    if name == "Concrete":
        df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls")
        return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(float)
    if name == "Airfoil":
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat", sep=r"\t", header=None)
        return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(float)
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing(); return d.data.astype(float), d.target.astype(float)


# ------------------------------------------------------------------ conformal and scores
def conformal_q(scores, alpha=ALPHA):
    s = np.sort(np.asarray(scores, float)); m = len(s); k = int(np.ceil((m + 1) * (1 - alpha)))
    return np.inf if k > m else float(s[k - 1])


def interval_metrics(lower, upper, y, alpha=ALPHA):
    width = upper - lower
    wink = width + (2 / alpha) * np.maximum(lower - y, 0) + (2 / alpha) * np.maximum(y - upper, 0)
    return dict(cov=float(np.mean((lower <= y) & (y <= upper))), width=float(np.mean(width)), winkler=float(np.mean(wink)))


def absres(pred_cal, y_cal, pred_te, y_te):
    q = conformal_q(np.abs(y_cal - pred_cal)); r = interval_metrics(pred_te - q, pred_te + q, y_te)
    r.update(mse=float(np.mean((y_te - pred_te) ** 2)), q=q); return r


# ------------------------------------------------------------------ GMDH (Phase-1 implementation, plus condition numbers)
def gmdh_layer(X, y, rng):
    n, d = X.shape; idx = rng.permutation(n); n_tr = min(max(int(n * 0.7), 1), n - 1)
    tr, va = idx[:n_tr], idx[n_tr:]; cands, conds = [], []
    for i in range(d):
        for j in range(i, d):
            u, v = X[:, i], X[:, j]; F = np.column_stack([np.ones(n), u, v, u ** 2, u * v, v ** 2])
            conds.append(float(np.linalg.cond(F.T @ F + RIDGE_ALPHA * np.eye(6))))
            m = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(F[tr], y[tr])
            cands.append((float(np.mean((y[va] - m.predict(F[va])) ** 2)), i, j, m.coef_.copy()))
    cands.sort(key=lambda c: c[0]); sel = cands[:min(GMDH_K, len(cands))]
    out = np.column_stack([np.column_stack([np.ones(n), X[:, i], X[:, j], X[:, i] ** 2, X[:, i] * X[:, j], X[:, j] ** 2]) @ c
                           for _, i, j, c in sel])
    return out, sel, np.array(conds)


def gmdh_transform(X, layers):
    F = X.copy()
    for nodes in layers:
        F = np.column_stack([np.column_stack([np.ones(len(F)), F[:, i], F[:, j], F[:, i] ** 2, F[:, i] * F[:, j], F[:, j] ** 2]) @ c
                             for _, i, j, c in nodes])
    return F


def fit_gmdh(X, y, depth, seed):
    F, layers, conds = X.copy(), [], []
    for L in range(depth):
        F, sel, cd = gmdh_layer(F, y, np.random.default_rng(seed + 1000 * (L + 1))); layers.append(sel); conds.append(cd)
    return layers, Ridge(alpha=RIDGE_ALPHA).fit(F, y), conds


# ------------------------------------------------------------------ one fold, all methods
def run_fold(X, y, name, rep, fold_no, tr_idx, te_idx):
    seed_base = SEED + 1000 * rep                                       # rep 0 == published Phase-1 seeds
    rng = np.random.default_rng(seed_base + fold_no * 10000); perm = rng.permutation(len(tr_idx))
    n_cal = min(max(int(len(tr_idx) * CAL_FRAC), 1), len(tr_idx) - 1)
    cal_i, trn_i = tr_idx[perm[:n_cal]], tr_idx[perm[n_cal:]]
    sx = StandardScaler().fit(X[trn_i]); ym, ys = y[trn_i].mean(), y[trn_i].std()
    Xtr, Xca, Xte = sx.transform(X[trn_i]), sx.transform(X[cal_i]), sx.transform(X[te_idx])
    ytr, yca, yte = (y[trn_i] - ym) / ys, (y[cal_i] - ym) / ys, (y[te_idx] - ym) / ys
    ms = seed_base + fold_no * 100 + 1
    rows, pnf_rows = [], []
    base = dict(dataset=name, rep=rep, fold=fold_no, n_train=len(trn_i), n_cal=n_cal, n_test=len(te_idx))

    def add(method, r, **extra):
        rows.append({**base, "method": method, **r, **extra})

    for depth in (1, 2):
        layers, final, conds = fit_gmdh(Xtr, ytr, depth, ms)
        r = absres(final.predict(gmdh_transform(Xca, layers)), yca, final.predict(gmdh_transform(Xte, layers)), yte)
        pnf = {f"pnf_L{L + 1}_{thr:.0e}": float(np.mean(c < thr)) for L, c in enumerate(conds) for thr in THRESHOLDS}
        pnf_rows.append({**base, "depth": depth, **pnf,
                         **{f"pnf_mean_{thr:.0e}": float(np.mean([np.mean(c < thr) for c in conds])) for thr in THRESHOLDS},
                         "cond_median": float(np.median(np.concatenate(conds)))})
        add(f"GMDH-d{depth}", r, pnf=pnf_rows[-1][f"pnf_mean_{PNF_THR:.0e}"])

    poly = PolynomialFeatures(2, include_bias=False).fit(Xtr)
    Ptr, Pca, Pte = poly.transform(Xtr), poly.transform(Xca), poly.transform(Xte)
    m = Ridge(alpha=RIDGE_ALPHA).fit(Ptr, ytr); add("Poly2-Ridge", absres(m.predict(Pca), yca, m.predict(Pte), yte), n_coef=Ptr.shape[1] + 1)
    m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Ptr, ytr)
    add("Poly2-RidgeCV", absres(m.predict(Pca), yca, m.predict(Pte), yte), ridge_alpha=float(m.alpha_), n_coef=Ptr.shape[1] + 1)

    m = RandomForestRegressor(n_estimators=100, random_state=ms + 1).fit(Xtr, ytr)
    add("RF", absres(m.predict(Xca), yca, m.predict(Xte), yte))
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=ms + 2, verbosity=0).fit(Xtr, ytr)
    add("XGB", absres(m.predict(Xca), yca, m.predict(Xte), yte))
    from interpret.glassbox import ExplainableBoostingRegressor
    m = ExplainableBoostingRegressor(random_state=ms + 3).fit(Xtr, ytr)
    add("EBM", absres(m.predict(Xca), yca, m.predict(Xte), yte))

    # CQR as published (XGBoost quantile objective) + diagnostics
    lo = XGBRegressor(objective="reg:quantileerror", quantile_alpha=ALPHA / 2, n_estimators=100, max_depth=6, learning_rate=0.1,
                      random_state=ms + 4, verbosity=0).fit(Xtr, ytr)
    hi = XGBRegressor(objective="reg:quantileerror", quantile_alpha=1 - ALPHA / 2, n_estimators=100, max_depth=6, learning_rate=0.1,
                      random_state=ms + 4, verbosity=0).fit(Xtr, ytr)
    for tag, (lo_m, hi_m) in (("CQR-XGB", (lo, hi)),
                              ("CQR-HGB", (HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA / 2, random_state=ms + 6).fit(Xtr, ytr),
                                           HistGradientBoostingRegressor(loss="quantile", quantile=1 - ALPHA / 2, random_state=ms + 6).fit(Xtr, ytr)))):
        lc, hc, lt, ht = lo_m.predict(Xca), hi_m.predict(Xca), lo_m.predict(Xte), hi_m.predict(Xte)
        qh = conformal_q(np.maximum(lc - yca, yca - hc))
        r = interval_metrics(lt - qh, ht + qh, yte); r.update(mse=float(np.mean((yte - (lt + ht) / 2) ** 2)), q=qh)
        add(tag, r, raw_band_width=float(np.mean(ht - lt)), raw_band_cov_cal=float(np.mean((lc <= yca) & (yca <= hc))),
            crossing_share=float(np.mean(ht < lt)))

    m = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=ms + 5, early_stopping=True,
                     validation_fraction=0.1, n_iter_no_change=10).fit(Xtr, ytr)
    add("MLP", absres(m.predict(Xca), yca, m.predict(Xte), yte), n_iter=int(m.n_iter_), best_val=float(m.best_validation_score_))
    return rows, pnf_rows


def run(args):
    os.makedirs(OUT, exist_ok=True)
    for name in args.datasets.split(","):
        X, y = load_dataset(name, args.synthetic)
        for rep in range(args.repeats):
            f = f"{OUT}/folds_{name}_r{rep}.csv"
            if os.path.exists(f):
                print(f"{name} repeat {rep}: done, skipping"); continue
            t0 = time.time(); R, P = [], []
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
            for fold_no, (tr, te) in enumerate(kf.split(X), start=1):
                r, p = run_fold(X, y, name, rep, fold_no, tr, te); R += r; P += p
            pd.DataFrame(R).to_csv(f, index=False); pd.DataFrame(P).to_csv(f"{OUT}/pnf_{name}_r{rep}.csv", index=False)
            print(f"{name} repeat {rep}: {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ analysis
PUBLISHED_R0 = {  # Table 1 of the submitted manuscript (mean coverage, mean width), for the reproduction check
    ("Yacht", "GMDH-d2"): (0.919, 0.396), ("Energy", "GMDH-d2"): (0.891, 1.003), ("Concrete", "GMDH-d2"): (0.926, 1.843),
    ("Airfoil", "GMDH-d2"): (0.898, 2.029), ("California", "GMDH-d2"): (0.905, 2.115), ("Yacht", "EBM"): (0.828, 0.084),
    ("California", "EBM"): (0.903, 1.221), ("Concrete", "CQR-XGB"): (0.891, 1.654), ("Airfoil", "MLP"): (0.905, 0.943)}


def exact_perm_spearman(x, y):
    obs = spearmanr(x, y).correlation; perms = list(itertools.permutations(range(len(y))))
    null = np.array([spearmanr(x, np.asarray(y)[list(p)]).correlation for p in perms])
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), len(perms)


def analyse(args):
    D = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/folds_*.csv")])
    P = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/pnf_*.csv")])
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    order = ["GMDH-d2", "GMDH-d1", "Poly2-Ridge", "Poly2-RidgeCV", "RF", "XGB", "EBM", "CQR-XGB", "CQR-HGB", "MLP"]
    D = D[D.dataset.isin(DATASETS)] if not args.synthetic else D
    nf = D.groupby(["dataset", "rep"]).fold.nunique(); np_ = P.groupby(["dataset", "rep"]).fold.nunique()
    bad = [k for k in nf.index if nf[k] != N_SPLITS or np_.get(k, 0) != N_SPLITS]
    print("Completeness check:", "all repeats complete" if not bad else f"INCOMPLETE {bad} (rerun these)")
    import sklearn, scipy
    print(f"Environment: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, sklearn {sklearn.__version__}, scipy {scipy.__version__}")
    print(f"Splits per dataset: {D.groupby('dataset').apply(lambda g: g[['rep', 'fold']].drop_duplicates().shape[0]).to_dict()}")

    print("\n0. REPRODUCTION CHECK (repeat 0 vs published Table 1)")
    for (ds, mth), (c, w) in PUBLISHED_R0.items():
        g = D[(D.dataset == ds) & (D.method == mth) & (D.rep == 0)]
        if len(g):
            print(f"   {ds:10s} {mth:8s} published cov {c:.3f} width {w:.3f} | now cov {g['cov'].mean():.3f} width {g.width.mean():.3f}")

    for metric, lab in (("cov", "COVERAGE"), ("width", "WIDTH"), ("winkler", "WINKLER SCORE (lower is better)"), ("mse", "POINT MSE")):
        t = D.groupby(["dataset", "method"])[metric].agg(["mean", "std"]).unstack("method")
        fmt = t["mean"].reindex(columns=order).round(3).astype(str) + " ± " + t["std"].reindex(columns=order).round(3).astype(str)
        print(f"\n1. {lab}: mean ± sd over all splits"); print(fmt.reindex(DATASETS).to_string())
    print("\n2. WINKLER RANK of each method per dataset (1 = best)")
    print(D.groupby(["dataset", "method"]).winkler.mean().unstack().reindex(columns=order).rank(axis=1).reindex(DATASETS).to_string())

    print("\n3. GMDH-d2 versus baselines: Winkler score, paired over splits.")
    print("   Test: corrected resampled t-test for repeated k-fold CV (Nadeau and Bengio 2003), which accounts for the")
    print("   overlap between splits; the naive paired p-value is shown only for reference and is too optimistic.")
    from scipy.stats import t as tdist
    for ds in DATASETS:
        sub = D[D.dataset == ds]
        g = sub.pivot_table(index=["rep", "fold"], columns="method", values="winkler")
        if g.empty or "GMDH-d2" not in g: continue
        ratio = float((sub.n_test / (sub.n_train + sub.n_cal)).mean())
        for base in ("GMDH-d1", "Poly2-Ridge", "Poly2-RidgeCV", "EBM"):
            if base not in g: continue
            d = (g["GMDH-d2"] - g[base]).dropna().values; J = len(d)
            if J < 3: continue
            se = np.sqrt((1 / J + ratio) * d.var(ddof=1)); tstat = d.mean() / se if se > 0 else np.nan
            p_corr = 2 * tdist.sf(abs(tstat), J - 1)
            p_naive = wilcoxon(d).pvalue if J > 5 else float("nan")
            print(f"   {ds:10s} GMDH-d2 minus {base:14s}: mean diff {d.mean():+.3f} (GMDH better in {(d < 0).mean():.0%} of {J} splits); "
                  f"corrected p={p_corr:.4f}  [naive Wilcoxon p={p_naive:.4f}]")

    print("\n4. CQR DIAGNOSTICS (mean over splits): raw quantile band before conformal correction, q_hat, crossing share")
    print(D[D.method.str.startswith("CQR")].groupby(["dataset", "method"])[["raw_band_width", "raw_band_cov_cal", "q", "crossing_share", "width", "mse"]].mean().round(3).to_string())
    print("\n5. MLP DIAGNOSTICS: iterations and width variability")
    print(D[D.method == "MLP"].groupby("dataset")[["n_iter", "best_val", "width"]].agg(["mean", "std", "min", "max"]).round(3).to_string())

    print("\n6. PNF: per layer and threshold (depth 2, mean over splits)")
    P2 = P[P.depth == 2]
    cols = [c for c in P2.columns if c.startswith("pnf_L")]
    print(P2.groupby("dataset")[cols + ["cond_median"]].mean().reindex(DATASETS).round(3).T.to_string())
    w = D[D.method == "GMDH-d2"].groupby("dataset").width.mean().reindex(DATASETS)
    print("\n7. PNF-WIDTH ASSOCIATION (dataset level, n = 5): Spearman rho with exact permutation p-value")
    for thr in THRESHOLDS:
        pnf = P2.groupby("dataset")[f"pnf_mean_{thr:.0e}"].mean().reindex(DATASETS)
        if pnf.isna().any() or len(pnf) < 3: continue
        rho, p, nperm = exact_perm_spearman(pnf.values, 1 / w.values)
        print(f"   threshold {thr:.0e}: PNF {np.round(pnf.values, 3).tolist()}  rho={rho:.3f}  exact p={p:.3f} ({nperm} permutations)")

    print("\n8. DEPTH SENSITIVITY (all splits)")
    print(D[D.method.str.startswith("GMDH")].groupby(["dataset", "method"])[["cov", "width", "winkler", "mse"]].mean().round(4).reindex(DATASETS, level=0).to_string())
    print("   California depth-2 MSE per split (to show the extrapolation outliers):")
    print("  ", np.round(np.sort(D[(D.dataset == "California") & (D.method == "GMDH-d2")].mse.values), 3).tolist())

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs("figs", exist_ok=True)
    for metric in ("winkler", "width", "cov"):
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 4))
        for ax, ds in zip(axes, DATASETS):
            g = D[D.dataset == ds]
            ax.boxplot([g[g.method == m][metric].values for m in order], showfliers=False)
            ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=90, fontsize=7); ax.set_title(ds)
            if metric == "cov": ax.axhline(0.9, ls="--", color="grey", lw=0.8)
        fig.tight_layout(); fig.savefig(f"figs/box_{metric}.png", dpi=200); plt.close(fig)
    print("\nFigures: figs/box_winkler.png, figs/box_width.png, figs/box_cov.png")


# ------------------------------------------------------------------ interpretability export (Yacht, repeat 0, fold 1)
def interpret(args):
    import sympy as sp
    name = "Yacht"; X, y = load_dataset(name, args.synthetic)
    tr, te = next(iter(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X)))
    rng = np.random.default_rng(SEED + 10000); perm = rng.permutation(len(tr)); n_cal = int(len(tr) * CAL_FRAC)
    trn = tr[perm[n_cal:]]; sx = StandardScaler().fit(X[trn]); Xtr = sx.transform(X[trn]); ym, ys = y[trn].mean(), y[trn].std()
    ytr = (y[trn] - ym) / ys
    names = FEATURES.get(name, [f"x{i}" for i in range(X.shape[1])])
    syms = sp.symbols([f"z{i + 1}" for i in range(X.shape[1])])
    print(f"Interpretability export: {name}, repeat 0, fold 1 training split (n={len(trn)}). z_i = standardised predictor i:")
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
                expr = c[0] + c[1] * u + c[2] * v + c[3] * u ** 2 + c[4] * u * v + c[5] * v ** 2
                lab = f"L{L}N{k}"
                inp = (f"{names[i]}, {names[j]}" if L == 1 else f"L{L - 1}N{i + 1}, L{L - 1}N{j + 1}")
                if i == j:
                    print(f"   {lab} (input: {inp.split(', ')[0]} only; internal val. MSE {mse:.4f}): "
                          f"{c[0]:+.3f} {c[1] + c[2]:+.3f}*a {c[3] + c[4] + c[5]:+.3f}*a^2")
                else:
                    print(f"   {lab} (inputs: {inp}; internal val. MSE {mse:.4f}): "
                          f"{c[0]:+.3f} {c[1]:+.3f}*a {c[2]:+.3f}*b {c[3]:+.3f}*a^2 {c[4]:+.3f}*a*b {c[5]:+.3f}*b^2")
                outs.append(sp.expand(expr))
            inputs = outs
        full = sp.expand(final.intercept_ + sum(float(w) * o for w, o in zip(final.coef_, inputs)))
        poly = sp.Poly(full, *syms); terms = sorted(poly.terms(), key=lambda t: -abs(float(t[1])))
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
    for n, v in imp[:10]: print(f"   {v:.4f}  {n}")


def pnfsens(args):
    """PNF sensitivity: condition numbers on the fitting subset only, with and without the ridge term."""
    rows = []
    for name in args.datasets.split(","):
        X, y = load_dataset(name, args.synthetic)
        for rep in range(args.repeats):
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
            for fold_no, (tr, te) in enumerate(kf.split(X), start=1):
                seed_base = SEED + 1000 * rep; rng = np.random.default_rng(seed_base + fold_no * 10000)
                perm = rng.permutation(len(tr)); n_cal = int(len(tr) * CAL_FRAC); trn = tr[perm[n_cal:]]
                sx = StandardScaler().fit(X[trn]); F = sx.transform(X[trn]); yy = (y[trn] - y[trn].mean()) / y[trn].std()
                ms = seed_base + fold_no * 100 + 1
                for L in range(2):
                    r = np.random.default_rng(ms + 1000 * (L + 1)); n = len(F); idx = r.permutation(n); fit = idx[:int(n * 0.7)]
                    reg, unreg = [], []
                    for i in range(F.shape[1]):
                        for j in range(i + 1, F.shape[1]):          # bivariate nodes only (self-pairs are rank-deficient by design)
                            B = np.column_stack([np.ones(len(fit)), F[fit, i], F[fit, j], F[fit, i] ** 2, F[fit, i] * F[fit, j], F[fit, j] ** 2])
                            unreg.append(np.linalg.cond(B)); reg.append(np.linalg.cond(B.T @ B + RIDGE_ALPHA * np.eye(6)))
                    rows.append(dict(dataset=name, rep=rep, fold=fold_no, layer=L + 1,
                                     **{f"pnf_fitsub_reg_{t:.0e}": float(np.mean(np.array(reg) < t)) for t in THRESHOLDS},
                                     **{f"pnf_fitsub_unreg_{t:.0e}": float(np.mean(np.array(unreg) < np.sqrt(t))) for t in THRESHOLDS}))
                    F, _, _ = gmdh_layer(F, yy, np.random.default_rng(ms + 1000 * (L + 1)))
    Q = pd.DataFrame(rows); Q.to_csv(f"{OUT}/pnf_sensitivity.csv", index=False); pd.set_option("display.width", 250)
    print("PNF computed on the fitting subset only (bivariate nodes). 'unreg' uses cond(B) against sqrt(threshold),")
    print("which is equivalent to cond(B^T B) against the threshold without the ridge term.")
    print(Q.groupby(["dataset", "layer"]).mean(numeric_only=True).drop(columns=["rep", "fold"]).round(3).to_string())
    D = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/folds_*.csv")])
    w = D[D.method == "GMDH-d2"].groupby("dataset").width.mean()
    for col in [c for c in Q.columns if c.startswith("pnf_")]:
        p = Q.groupby("dataset")[col].mean().reindex(w.index)
        if p.nunique() > 1 and len(p) >= 3:
            rho, pv, nperm = exact_perm_spearman(p.values, 1 / w.values)
            print(f"   {col:28s} rho={rho:+.3f}  exact p={pv:.3f} ({nperm} permutations)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "analyse", "interpret", "pnfsens"])
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--synthetic", action="store_true", help="offline smoke test with synthetic data")
    a = ap.parse_args(); {"run": run, "analyse": analyse, "interpret": interpret, "pnfsens": pnfsens}[a.cmd](a)
