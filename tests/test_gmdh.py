import numpy as np

from gmdh_conformal.gmdh import fit_gmdh, gmdh_transform


def test_gmdh_transform_and_fit_are_deterministic_for_fixed_seed():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] ** 2 + 0.3 * X[:, 1] - X[:, 2]
    layers_a, model_a, _ = fit_gmdh(X, y, depth=2, seed=23)
    layers_b, model_b, _ = fit_gmdh(X, y, depth=2, seed=23)
    pred_a = model_a.predict(gmdh_transform(X, layers_a))
    pred_b = model_b.predict(gmdh_transform(X, layers_b))
    np.testing.assert_allclose(pred_a, pred_b)
    assert gmdh_transform(X, layers_a).shape == (80, 6)


def test_gmdh_returns_expected_number_of_selected_nodes():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + X[:, 1] ** 2
    layers, model, _ = fit_gmdh(X, y, depth=1, seed=4)
    assert len(layers[0]) == 6
    assert model.coef_.shape == (6,)
