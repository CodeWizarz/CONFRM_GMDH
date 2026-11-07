import math

import numpy as np

from gmdh_conformal.metrics import conformal_q, interval_metrics


def test_conformal_quantile_uses_finite_sample_order_statistic():
    assert conformal_q([1, 2, 3, 4, 5], alpha=0.2) == 5


def test_conformal_quantile_is_infinite_when_sample_is_too_small():
    assert math.isinf(conformal_q([1, 2], alpha=0.1))


def test_interval_metrics_report_coverage_width_and_winkler_score():
    got = interval_metrics(np.array([0.0, 1.0]), np.array([2.0, 3.0]), np.array([1.0, 4.0]), alpha=0.5)
    assert got["cov"] == 0.5
    assert got["width"] == 2.0
    assert got["winkler"] == 4.0
