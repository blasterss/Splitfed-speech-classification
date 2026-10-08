# E0: cross-corpus domain shift

This is a descriptive analysis of corpus distributions, without model training.

Run E0 domain shift without training or a train/test split:

```bash
uv run python -c "from src.experiments.domain_shift.analysis import main; main()" \
  --config-file configs/experiments/config.e0.yaml \
  --artifact-root artifacts/e0_domain_shift/final_seed_42
```

The E0 configuration contains only dataset roots and feature settings; model,
optimizer, noise, channels and training/split options are not used.
The runner uses all successfully extracted records without a train/test split.
Each record is represented by valid-frame temporal means (13 MFCC, RMS, ZCR).
Client-local normalization uses population statistics of all record vectors
in each corpus independently, separate from train-only training normalization.

MMD squared uses 480 records per corpus, 50 repetitions, raw and client-local
normalized spaces, and 500 permutations. Both spaces use the same indices.
RBF bandwidth is the square root of the pooled median positive squared distance
per pair/repeat/space. Do not subtract raw and normalized values or report a
percentage reduction. With 500 permutations the p-value floor is 1/501.

Exact multivariate Euclidean W1 uses only client-local normalized features,
240 records per empirical distribution and 50 repetitions per pair. Each
corpus supplies two disjoint random halves A and B without replacement.
Between compares the first halves of the two corpora; within compares A and B
inside each corpus. R = between / ((within_left + within_right) / 2), computed
per repetition. A zero denominator leaves R undefined (NaN). R is a relative
finite-sample estimate, not a population effect size. At least
max(MMD sample size, twice W1 sample size) valid records are required.

Outputs are MMD `pairwise_repeats.csv` and `pairwise_summary.csv`,
`wasserstein_repeats.csv` and `wasserstein_summary.csv`, descriptive Table 1
`corpus_summary.csv` and Table 2 `feature_summary.csv`, and
`resolved_analysis.yaml` with extraction loss, actor/class counts, normalization
statistics and replayable sample indices. Sinkhorn is excluded.
Repeat quantiles describe subsampling variation, not confidence intervals.
Record sampling and permutations ignore actor dependence; within halves are
record-disjoint, not actor-disjoint. P-values are descriptive under this
limitation. SAVEE at MMD n=480 uses all records every repetition; W1 randomly
partitions its records into two halves. The notebook delegates computations to
the runner and creates no model or checkpoint.

## Implementation and interpretation

- Runner: [analysis.py](../../src/experiments/domain_shift/analysis.py).
- MMD: [mmd.py](../../src/experiments/domain_shift/mmd.py).
- Exact Euclidean W1: [wasserstein.py](../../src/experiments/domain_shift/wasserstein.py).
- Notebook: [E0](../../notebooks/E0_cross_corpus_heterogeneity.ipynb).
- Focused checks: [test_domain_shift.py](../../tests/experiments/test_domain_shift.py).

MMD indicates discrepancy in its selected kernel space. Bandwidth changes across
pairs and repetitions, limiting comparisons of its magnitude between pairs.
W1 uses the same sample size for between and within estimates. R > 1 means
between exceeds the average within baseline in that repetition; it is not a
hypothesis test. The mean R is the mean of per-repeat ratios, not a ratio of
aggregated distances. Within halves are record-disjoint but may share actors.

Table 1 describes corpus/class/actor counts. Table 2 describes each raw feature
across all available record vectors. Neither requires a training split.
Generated artifacts remain under ignored `artifacts/`. Use a distinct output
directory for each protocol/seed; the runner does not prevent overwriting files.
Historical runs with Sinkhorn and W1 at n=480 use a different protocol and must
not be mixed with the current W1 results. A saved run is not a CI acceptance gate.
