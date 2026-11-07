"""Study protocol constants carried over from the supplied revision script."""

SEED = 42
ALPHA = 0.10
N_SPLITS = 5
CAL_FRAC = 0.18
GMDH_K = 6
RIDGE_ALPHA = 1.0
PNF_THR = 1e4
THRESHOLDS = (1e3, 1e4, 1e5, 1e6)
DATASETS = ("Yacht", "Energy", "Concrete", "Airfoil", "California")
FEATURES = {
    "Yacht": ["LCB position", "Prismatic coef.", "Length-displacement", "Beam-draught", "Length-beam", "Froude number"],
}
