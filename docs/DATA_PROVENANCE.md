# Data provenance

The supplied revision script loads these benchmark datasets at run time:

| Dataset | Source in supplied script | Handling |
|---|---|---|
| Yacht Hydrodynamics | UCI archive, `00243/yacht_hydrodynamics.data` | Downloaded directly; six predictors and final response column |
| Energy Efficiency | UCI archive, `00242/ENB2012_data.xlsx` | Downloaded directly; final two columns are treated as responses and the first response is modeled |
| Concrete Compressive Strength | UCI archive, `concrete/compressive/Concrete_Data.xls` | Downloaded directly; final column is response |
| Airfoil Self-Noise | UCI archive, `00291/airfoil_self_noise.dat` | Downloaded directly; final column is response |
| California Housing | `sklearn.datasets.fetch_california_housing` | Obtained through scikit-learn's dataset cache/download mechanism |

The archived CSV result files came from the user-supplied ZIP created during the
revision work. The source data snapshots and their retrieval dates/checksums
were not included. Therefore, a new run may change if a remote dataset changes;
the archived result bundle remains the provenance record for the supplied
numbers. Before formal release, record actual retrieval dates and file checksums
if the input data are archived or pinned.

The archived original scripts contain an offline synthetic-data smoke-test
branch. It is preserved as part of the original source snapshot, is not one of
the study datasets, and is not exposed by the packaged CLI.
