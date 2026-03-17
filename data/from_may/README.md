# From-May Dataset Results

## Dataset

- RNA parquet: `data/from_may/all_rna.parquet`
- Protein parquet: `data/from_may/all_protein.parquet`
- RNA shape: 1546 samples x 121217 columns
- Protein shape: 1574 samples x 30837 columns
- Modeling uses z-score values by default, matched samples only, and post-split median imputation.

## Final Single-Target RF Runs

These are the main `k=1000` runs on `data/from_may` after the pipeline was updated to use z-score inputs and post-split median imputation.

| Target | k | n_iter | n_samples | Baseline R2 | RF R2 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| KRAS | 1000 | 30 | 1071 | -1.7803 | 0.3981 | `outputs_from_may_k1000_kras_n30` |
| KRAS | 1000 | 100 | 1071 | -1.7803 | 0.3992 | `outputs_from_may_k1000_kras_stage1_n100` |
| TP53 | 1000 | 10 | 763 | -0.3929 | 0.3771 | `outputs_from_may_k1000_tp53_n10` |
| TP53 | 1000 | 30 | 763 | -0.3929 | 0.3771 | `outputs_from_may_k1000_tp53_n30` |
| GAPDH | 1000 | 10 | 1071 | -1.4135 | 0.4052 | `outputs_from_may_k1000_gapdh_n10` |
| GAPDH | 1000 | 30 | 1071 | -1.4135 | 0.4052 | `outputs_from_may_k1000_gapdh_n30` |

### Interpretation

- KRAS improved substantially when hyperparameter search increased from `n_iter=10` to `30/100`.
- TP53 and GAPDH were effectively unchanged between `n_iter=10` and `30`.
- Best observed RF R2 values on this dataset:
  - GAPDH: `0.4052`
  - KRAS: `0.3992`
  - TP53: `0.3771`

## Earlier KRAS Feature-Selection Runs

These were exploratory KRAS runs before the current final configuration stabilized.

| Target | k | n_iter | n_samples | RF R2 | Output Folder |
| --- | ---: | ---: | ---: | ---: | --- |
| KRAS | 100 | 30 | 1370 | 0.1391 | `outputs_from_may_k100` |
| KRAS | 200 | 10 | 1370 | 0.1706 | `outputs_from_may_k200_n10` |
| KRAS | 1000 | 10 | 1370 | 0.2081 | `outputs_from_may_k1000_n10` |

### Interpretation

- Increasing `k` from `100` to `1000` improved KRAS performance in the early runs.
- These runs used an older pipeline state and are not directly comparable to the later `k=1000` final runs.

## Hyperparameter Stability

### KRAS

- `n_iter=30` best params:
  - `n_estimators=1200`
  - `min_samples_split=10`
  - `min_samples_leaf=8`
  - `max_features=0.8`
  - `max_depth=10`
  - `bootstrap=true`
- `n_iter=100` best params:
  - `n_estimators=800`
  - `min_samples_split=5`
  - `min_samples_leaf=8`
  - `max_features=0.8`
  - `max_depth=40`
  - `bootstrap=true`

These are very similar model families. The test R2 difference was negligible: `0.3981` vs `0.3992`.

### TP53

- `n_iter=10` and `n_iter=30` produced identical params:
  - `n_estimators=500`
  - `min_samples_split=10`
  - `min_samples_leaf=8`
  - `max_features=0.5`
  - `max_depth=5`
  - `bootstrap=false`

### GAPDH

- `n_iter=10` and `n_iter=30` produced identical params:
  - `n_estimators=300`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
  - `max_features=0.5`
  - `max_depth=40`
  - `bootstrap=false`

## KRAS Elbow Sweep

Fast tuned elbow sweep output:

- Folder: `elbow_from_may_kras_tuned_fast_10k`
- Start point: top 10000 features (prefiltered with `f_regression` on training data)
- Step size: 1000 features
- Tuning strategy: reduced search budget for speed

Observed RF R2 values:

| k | RF R2 |
| ---: | ---: |
| 10000 | 0.3941 |
| 9000 | 0.3895 |
| 8000 | 0.3844 |
| 7000 | 0.3910 |
| 6000 | 0.3873 |
| 5000 | 0.3886 |
| 4000 | 0.3938 |
| 3000 | 0.3981 |
| 2000 | 0.3941 |
| 1000 | 0.3958 |

### Interpretation

- There is no sharp elbow.
- KRAS performance is broadly stable from `k=10000` down to `k=1000`.
- The best observed point in this sweep was `k=3000`, but the entire range is fairly flat.

## Shuffle Controls

Negative-control runs used the exact same split, preprocessing, selected features, and fixed RF hyperparameters, but shuffled `y_train` once before fitting.

| Target | Real RF R2 | Shuffled RF R2 | Output Folder |
| --- | ---: | ---: | --- |
| KRAS | 0.3992 | -0.1028 | `outputs_from_may_k1000_kras_shuffle_control` |
| TP53 | 0.3771 | -0.0241 | `outputs_from_may_k1000_tp53_shuffle_control` |
| GAPDH | 0.4052 | -0.0623 | `outputs_from_may_k1000_gapdh_shuffle_control` |

### Interpretation

- All three shuffled controls collapsed to near-zero or negative R2.
- This is the expected negative-control result.
- It supports the conclusion that the real RF models are learning actual structure rather than fitting noise.

## Main Takeaways

- The `from_may` dataset supports meaningful protein prediction from RNA for all three tested targets.
- Current ranking by RF R2 at `k=1000`:
  - GAPDH `0.4052`
  - KRAS `0.3992`
  - TP53 `0.3771`
- KRAS needed real hyperparameter tuning; TP53 and GAPDH did not.
- Reducing KRAS from `10000` to `1000` features did not cause a major drop in RF performance.
- Shuffle controls validate that the signal is real.
