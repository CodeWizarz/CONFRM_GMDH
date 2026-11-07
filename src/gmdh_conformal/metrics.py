"""Split-conformal calibration and interval metrics."""

import numpy as np

from .config import ALPHA


def conformal_q(scores, alpha=ALPHA):
    s = np.sort(np.asarray(scores, float))
    m = len(s)
    k = int(np.ceil((m + 1) * (1 - alpha)))
    return np.inf if k > m else float(s[k - 1])


def interval_metrics(lower, upper, y, alpha=ALPHA):
    width = upper - lower
    wink = width + (2 / alpha) * np.maximum(lower - y, 0) + (2 / alpha) * np.maximum(y - upper, 0)
    return {
        "cov": float(np.mean((lower <= y) & (y <= upper))),
        "width": float(np.mean(width)),
        "winkler": float(np.mean(wink)),
    }


def absres(pred_cal, y_cal, pred_te, y_te):
    q = conformal_q(np.abs(y_cal - pred_cal))
    result = interval_metrics(pred_te - q, pred_te + q, y_te)
    result.update(mse=float(np.mean((y_te - pred_te) ** 2)), q=q)
    return result
