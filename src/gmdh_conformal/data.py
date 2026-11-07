"""Dataset loading. Public datasets are fetched from their original sources."""

import numpy as np
import pandas as pd


def load_dataset(name: str, synthetic: bool = False):
    """Load one study dataset; synthetic mode is for offline smoke checks only."""
    if synthetic:
        rng = np.random.default_rng(len(name))
        n = {"California": 3000}.get(name, 400)
        X = rng.normal(size=(n, 6))
        y = X[:, 0] ** 2 + X[:, 1] * X[:, 2] + 0.3 * rng.normal(size=n)
        return X, y
    if name == "Yacht":
        df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
            sep=r"\s+",
            header=None,
        )
        return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(float)
    if name == "Energy":
        df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx")
        return df.iloc[:, :-2].values.astype(float), df.iloc[:, -2].values.astype(float)
    if name == "Concrete":
        df = pd.read_excel(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
        )
        return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(float)
    if name == "Airfoil":
        df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
            sep=r"\t",
            header=None,
        )
        return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(float)
    if name == "California":
        from sklearn.datasets import fetch_california_housing

        dataset = fetch_california_housing()
        return dataset.data.astype(float), dataset.target.astype(float)
    raise ValueError(f"Unknown dataset {name!r}; expected one of Yacht, Energy, Concrete, Airfoil, California")
