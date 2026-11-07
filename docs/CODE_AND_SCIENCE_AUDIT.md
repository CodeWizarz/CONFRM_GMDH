# Code and scientific audit notes

This package preserves the supplied methods and outputs. The items below need
author review before treating the repository or manuscript as publication-ready.

1. **No immutable data snapshots were supplied.** The scripts retrieve four UCI
   datasets over the network and California Housing through scikit-learn. Record
   source versions, retrieval dates, and checksums for long-term reproducibility.
   The first revision ZIP contains 48 CSVs; California repeat 4 fold and PNF
   CSVs were absent. A follow-up ZIP supplied on 2026-11-07 includes all 50 fold
   and repeat PNF files. Its California repeat-3 fold and PNF files differ from
   the first ZIP, so both bundles are preserved and the follow-up bundle is the
   default validation target. The separate PNF sensitivity CSV is not a per-run
   PNF file and is intentionally excluded from fold-level validation.
2. **The original code reports metrics on standardized targets.** `mse`, interval
   width, and Winkler scores use targets standardized using the fitting subset.
   State the units clearly in the manuscript and do not compare these values to
   original-unit metrics without conversion.
3. **Depth-2 California GMDH is unstable.** The supplied report gives split MSE
   values up to 908.08 (standardized target units) and mean MSE 52.554, with
   conspicuous extrapolation outliers. Those outputs are preserved; they are not
   silently clipped or removed.
4. **The PNF–width association is exploratory.** It correlates five dataset-level
   summaries. The supplied report's threshold 1e4 result is rho=0.975, exact
   permutation p=0.033, n=5; this is sensitive to the threshold and has limited
   generalizability. Avoid a broad causal or population-level claim.
5. **GMDH internal selection and PNF need method clarification.** Candidate
   condition numbers in `gmdh_layer` are computed on the complete layer input,
   while candidate coefficients are fit on an internal 70% subset. In later
   layers, those inputs are outputs of prior-layer nodes selected/fitted before
   the later layer's internal validation split. Confirm that this matches the
   intended validation protocol and explain which rows enter each PNF estimate.
6. **The California loader assumes internet/cache availability.** The other
   benchmark loaders also have source-host availability dependencies.
7. **The lock does not recreate the original environment.** The report only provides exact
   versions for Python, NumPy, pandas, scikit-learn, and SciPy. Exact versions
   for XGBoost, InterpretML, Matplotlib, SymPy, and Excel readers are unknown.
   The committed lock resolves a current Python 3.12 environment, not the exact
   Colab environment from the report. XGBoost also requires the system OpenMP
   runtime on macOS.
8. **License and software citation choices.** The repository uses the MIT
   license and cites the software under the supplied GitHub handle
   `CodeWizarz`; update the copyright holder or citation if the rights holder or
   preferred author name differs.

The supplied analysis report claims corrected resampled t-tests for the split
comparisons. Review the exact test formulation, degrees of freedom, and
pre-specified comparison family against the methods section before submission.
This package does not alter those inferential results.

The source script's repeat-0 reproduction table also includes four non-GMDH
comparators. At least the Yacht EBM coverage differs from the earlier manuscript
value in the supplied report (0.835 versus 0.828). The result validator checks
the five GMDH-d2 reference values only and does not incorrectly require those
known comparator differences to reproduce the earlier table.
