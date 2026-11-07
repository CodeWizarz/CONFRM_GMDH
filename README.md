# GMDH conformal prediction revision

This repository packages the supplied revision experiment as an installable
Python project. It keeps the original scripts and result bundle under
`provenance/` unchanged and adds a modular implementation, protocol config,
tests, and result checks. It does not claim new experiment results.

## Contents

- `src/gmdh_conformal/`: dataset loaders, metrics, GMDH implementation,
  experiment runner, CLI, and archived-result validator.
- `configs/revision.yaml`: protocol values transcribed from the supplied script.
- `.github/workflows/ci.yml`: locked dependency, lint, and test checks on pushes and pull requests.
- `provenance/originals/`: original script versions copied byte-for-byte.
- `provenance/results/`: original and follow-up ZIPs, reports, diagnostics,
  extracted split outputs, and figures. These are immutable source results.
- `provenance/additional_attachments/`: separate California PNF CSV attachments
  supplied alongside the ZIP, preserved independently.
- `docs/`: dataset provenance and code/scientific audit notes.

## Local setup on macOS

Install `uv` using its official macOS installer, then install Python 3.12 with
`uv` (the project supports Python 3.10–3.13):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
cd /path/to/gmdh-conformal-revision
uv venv --python 3.12
uv sync --locked --extra dev
source .venv/bin/activate
```

The original Colab report identifies Python 3.13.18, NumPy 2.4.6, pandas 2.2.3,
scikit-learn 1.6.1, and SciPy 1.16.3. The committed `uv.lock` resolves a current
Python 3.12 environment for this package; it does not recreate the original
Colab versions. Regenerate it only when intentionally updating dependencies.
On macOS, XGBoost also needs the OpenMP runtime. If it is not already installed,
install it with `brew install libomp` before running experiments.

## Validate supplied results

From the repository root:

```sh
gmdh-conformal validate
```

This checks the archived fold and PNF file structure and compares the recorded
repeat-0 GMDH coverage and width summaries against the constants transcribed from
the supplied original script. It reads the follow-up bundle only. The first ZIP
is retained as originally supplied and is incomplete; the follow-up ZIP includes
all California repeat-4 fold and PNF files. It does not regenerate or rewrite
results.

## Run experiments

The full protocol can take several hours; the source script notes roughly 6–8
hours for five repeats across all datasets, chiefly due to EBM. Run a smaller
subset first if needed:

```sh
gmdh-conformal run --datasets Yacht,Energy --repeats 1 --output-dir outputs
gmdh-conformal analyse --datasets Yacht,Energy --repeats 1 --output-dir outputs
gmdh-conformal interpret
gmdh-conformal pnfsens --datasets Yacht,Energy --repeats 1 --output-dir outputs
```

The detailed analysis writes the corrected resampled comparisons and figures;
`interpret` exports the Yacht equations and EBM importance values; `pnfsens`
computes the condition-number sensitivity diagnostics. The package runner loads
only the five named study datasets. It refuses to overwrite existing result
files; choose a new output directory for each run. Public benchmark data are fetched from the sources documented in
`docs/DATA_PROVENANCE.md`. California Housing may need a first-run download.

## Tests

```sh
pytest
```

Tests cover conformal order-statistic behavior, interval scores, and deterministic
GMDH fitting, and the completeness and reference checks on the archived result
bundle. Passing software tests does not validate the statistical claims or prove
that a new run matches the manuscript.

## Journal review and repository identity

The manuscript code identifies *Next Research*. Elsevier's [Next journals Guide
for Authors](https://www-prod.elsevier.com/subject/next/guide-for-authors)
specifies a single-anonymized review process. This public repository identifies
the GitHub account in its history and citation. The MIT license and software
citation are provided in the repository root. Review the scientific and
reproducibility notes in `docs/CODE_AND_SCIENCE_AUDIT.md` before submission.
