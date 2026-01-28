from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils_io import load_matrix_parquet, select_value_type, standardize_gene_ids


RNA_PARQUET_PATH = Path("data/processed/all_rna.parquet")
PROT_PARQUET_PATH = Path("data/processed/all_protein.parquet")
OUTPUT_DIR = Path("outputs")
TARGETS = ["KRAS"]
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_JOBS = -1


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RF models to predict protein abundance from RNA-seq.",
    )
    parser.add_argument("--rna", type=str, default=str(RNA_PARQUET_PATH))
    parser.add_argument("--prot", type=str, default=str(PROT_PARQUET_PATH))
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--targets", type=str, default=",".join(TARGETS))
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument(
        "--feature-selection",
        choices=["none", "kbest"],
        default="kbest",
        help="Optional feature selection using SelectKBest(f_regression).",
    )
    parser.add_argument("--k-best", type=int, default=2000)
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def _clean_sample_ids(ids: Iterable) -> pd.Index:
    return pd.Index([str(value).strip() for value in ids])


def _flatten_feature_index(index: pd.Index) -> pd.Index:
    if not isinstance(index, pd.MultiIndex):
        return pd.Index(index.astype(str))
    if "Name" in index.names:
        return pd.Index(index.get_level_values("Name").astype(str))
    return pd.Index(index.get_level_values(0).astype(str))


def _prepare_feature_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix = select_value_type(matrix, prefer="raw")
    names = _flatten_feature_index(matrix.index)
    names = standardize_gene_ids(names)
    matrix = matrix.copy()
    matrix.index = names
    matrix = matrix[matrix.index != ""]
    matrix = matrix.groupby(matrix.index).mean()
    return matrix


def _resolve_target(index: pd.Index, target: str) -> Optional[str]:
    if target in index:
        return target
    lower_map = {str(name).lower(): name for name in index}
    return lower_map.get(target.lower())


def _pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _build_baseline_pipeline(k_best: Optional[int]) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if k_best is not None:
        steps.append(("select", SelectKBest(f_regression, k=k_best)))
    steps.extend(
        [
            ("scale", StandardScaler()),
            ("model", Ridge()),
        ]
    )
    return Pipeline(steps)


def _build_rf_pipeline(seed: int, n_jobs: int, k_best: Optional[int]) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if k_best is not None:
        steps.append(("select", SelectKBest(f_regression, k=k_best)))
    steps.append(
        (
            "model",
            RandomForestRegressor(
                n_estimators=500,
                random_state=seed,
                n_jobs=n_jobs,
                oob_score=False,
            ),
        )
    )
    return Pipeline(steps)


def _feature_names_after_selection(pipeline: Pipeline, feature_names: Iterable[str]) -> pd.Index:
    if "select" not in pipeline.named_steps:
        return pd.Index(feature_names)
    selector = pipeline.named_steps["select"]
    if hasattr(selector, "get_support"):
        mask = selector.get_support()
        return pd.Index(feature_names)[mask]
    return pd.Index(feature_names)


def _save_pred_plot(out_dir: Path, target: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.4)
    min_val = np.nanmin([y_true.min(), y_pred.min()])
    max_val = np.nanmax([y_true.max(), y_pred.max()])
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
    plt.xlabel("True protein abundance")
    plt.ylabel("Predicted protein abundance")
    plt.title(f"{target} predicted vs true")
    plt.tight_layout()
    plot_path = out_dir / f"pred_vs_true_{target}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()


def _write_progress(out_dir: Path, payload: dict) -> None:
    progress_path = out_dir / "progress.json"
    progress_path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    _setup_logging()
    args = _parse_args()
    np.random.seed(args.seed)
    start_time = time.time()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        logging.error("No targets provided.")
        return 1
    _write_progress(
        out_dir,
        {
            "stage": "init",
            "targets": targets,
            "start_time": start_time,
        },
    )

    logging.info("Loading RNA from %s", args.rna)
    rna = load_matrix_parquet(args.rna)
    logging.info("Loading proteomics from %s", args.prot)
    prot = load_matrix_parquet(args.prot)
    _write_progress(
        out_dir,
        {
            "stage": "loaded_inputs",
            "targets": targets,
            "elapsed_sec": round(time.time() - start_time, 1),
        },
    )

    rna.columns = _clean_sample_ids(rna.columns)
    prot.columns = _clean_sample_ids(prot.columns)

    rna = _prepare_feature_matrix(rna)
    prot = _prepare_feature_matrix(prot)

    common_samples = sorted(set(rna.columns) & set(prot.columns))
    if not common_samples:
        logging.error("No overlapping sample IDs between RNA and proteomics.")
        return 1

    rna = rna.loc[:, common_samples]
    prot = prot.loc[:, common_samples]

    missing_frac = rna.isna().mean(axis=0)
    keep_samples = missing_frac[missing_frac <= 0.2].index
    rna = rna.loc[:, keep_samples]
    prot = prot.loc[:, keep_samples]

    metrics = []
    missing_targets = []
    feature_selection = args.feature_selection

    for target in targets:
        target_start = time.time()
        _write_progress(
            out_dir,
            {
                "stage": "target_start",
                "target": target,
                "elapsed_sec": round(time.time() - start_time, 1),
            },
        )
        resolved = _resolve_target(prot.index, target)
        if resolved is None:
            logging.warning("Target %s not found in proteomics. Skipping.", target)
            metrics.append(
                {
                    "target": target,
                    "n_samples": 0,
                    "baseline_r2": np.nan,
                    "baseline_mae": np.nan,
                    "baseline_pearson": np.nan,
                    "rf_r2": np.nan,
                    "rf_mae": np.nan,
                    "rf_pearson": np.nan,
                    "rf_best_cv_r2": np.nan,
                    "rf_best_params": "",
                    "target_rna_in_top50": False,
                }
            )
            missing_targets.append(target)
            continue

        y = prot.loc[resolved].dropna()
        if y.empty:
            logging.warning("Target %s has no non-missing proteomics values.", target)
            continue

        X = rna.loc[:, y.index].T
        y = y.loc[X.index]
        X = X.loc[~X.index.duplicated(keep="first")]
        y = y.loc[~y.index.duplicated(keep="first")]
        X, y = X.align(y, join="inner", axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )

        k_best = None
        if feature_selection == "kbest":
            k_best = min(args.k_best, X_train.shape[1])
            if k_best < 2:
                k_best = None

        baseline = _build_baseline_pipeline(k_best)
        baseline.fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)
        baseline_r2 = r2_score(y_test, baseline_pred)
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_pearson = _pearson_corr(y_test.values, baseline_pred)

        rf_pipeline = _build_rf_pipeline(args.seed, args.n_jobs, k_best)
        param_distributions = {
            "model__n_estimators": [300, 500, 800, 1200],
            "model__max_depth": [None, 5, 10, 20, 40],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.2, 0.5, 0.8],
            "model__bootstrap": [True, False],
        }

        logging.info("Starting RF hyperparameter search for %s", target)
        search_start = time.time()
        search = RandomizedSearchCV(
            rf_pipeline,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            scoring="r2",
            cv=5,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbose=1,
        )
        search.fit(X_train, y_train)
        logging.info(
            "Finished RF search for %s in %.1f sec",
            target,
            time.time() - search_start,
        )

        best_rf = search.best_estimator_
        rf_pred = best_rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_pearson = _pearson_corr(y_test.values, rf_pred)

        _save_pred_plot(out_dir, target, y_test.values, rf_pred)

        feature_names = pd.Index(X.columns)
        selected_features = _feature_names_after_selection(best_rf, feature_names)
        importances = best_rf.named_steps["model"].feature_importances_
        importance_df = pd.DataFrame(
            {"feature": selected_features, "importance": importances}
        ).sort_values("importance", ascending=False)
        top_features = importance_df.head(50)
        top_path = out_dir / f"feature_importance_{target}.csv"
        top_features.to_csv(top_path, index=False)

        target_rna_in_top50 = target in set(top_features["feature"].astype(str))

        if args.save_models:
            model_path = out_dir / f"rf_{target}.joblib"
            joblib.dump(best_rf, model_path)

        metrics.append(
            {
                "target": target,
                "n_samples": int(len(y)),
                "baseline_r2": float(baseline_r2),
                "baseline_mae": float(baseline_mae),
                "baseline_pearson": float(baseline_pearson),
                "rf_r2": float(rf_r2),
                "rf_mae": float(rf_mae),
                "rf_pearson": float(rf_pearson),
                "rf_best_cv_r2": float(search.best_score_),
                "rf_best_params": json.dumps(search.best_params_),
                "target_rna_in_top50": bool(target_rna_in_top50),
            }
        )

        logging.info(
            "Target %s: baseline R2=%.3f, RF R2=%.3f",
            target,
            baseline_r2,
            rf_r2,
        )
        _write_progress(
            out_dir,
            {
                "stage": "target_done",
                "target": target,
                "elapsed_sec": round(time.time() - start_time, 1),
                "target_elapsed_sec": round(time.time() - target_start, 1),
            },
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logging.info("Saved metrics to %s", metrics_path)

    gap_rows = []
    gap_pairs = [("KRAS", "GAPDH"), ("TP53", "GAPDH")]
    for driver, control in gap_pairs:
        driver_row = metrics_df.loc[metrics_df["target"] == driver]
        control_row = metrics_df.loc[metrics_df["target"] == control]
        if driver_row.empty or control_row.empty:
            continue
        driver_r2 = float(driver_row["rf_r2"].iloc[0])
        control_r2 = float(control_row["rf_r2"].iloc[0])
        if np.isnan(driver_r2) or np.isnan(control_r2):
            continue
        gap_rows.append(
            {
                "driver": driver,
                "control": control,
                "accuracy_gap_rf_r2": driver_r2 - control_r2,
            }
        )

    gap_df = pd.DataFrame(gap_rows)
    gap_path = out_dir / "accuracy_gap.csv"
    gap_df.to_csv(gap_path, index=False)

    if not gap_df.empty:
        logging.info("Accuracy gap summary:\n%s", gap_df.to_string(index=False))
    else:
        logging.info("No accuracy gap results computed.")

    if missing_targets:
        logging.info("Missing targets: %s", ", ".join(missing_targets))

    _write_progress(
        out_dir,
        {
            "stage": "done",
            "elapsed_sec": round(time.time() - start_time, 1),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
