from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rf_rna_to_protein import (
    _clean_sample_ids,
    _median_impute_after_split,
    _pearson_corr,
    _prepare_feature_matrix,
    _resolve_target,
)
from utils_io import load_matrix_parquet


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an RF sweep from all features down to k=1000 by iteratively "
            "dropping the least important features and re-tuning RF at each k."
        ),
    )
    parser.add_argument("--rna", type=str, default="data/from_may/all_rna.parquet")
    parser.add_argument("--prot", type=str, default="data/from_may/all_protein.parquet")
    parser.add_argument("--target", type=str, default="KRAS")
    parser.add_argument("--outdir", type=str, default="elbow_from_may_kras_tuned")
    parser.add_argument(
        "--value-type",
        choices=["raw", "z"],
        default="z",
        help="Use normalized z-score values when available.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument(
        "--start-features",
        type=int,
        default=10000,
        help="Initial number of features to keep before the elbow sweep starts.",
    )
    parser.add_argument(
        "--retune-every",
        type=int,
        default=2,
        help="Retune RF every N elbow steps; reuse last best params in between.",
    )
    parser.add_argument(
        "--final-n-estimators",
        type=int,
        default=800,
        help="Refit each selected configuration with this many trees for evaluation.",
    )
    parser.add_argument(
        "--drop-step",
        type=int,
        default=1000,
        help="Number of least-important features to drop each round.",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=1000,
        help="Stop once this many features remain.",
    )
    parser.add_argument(
        "--descending-k",
        action="store_true",
        help="Plot larger k on the left and smaller k on the right.",
    )
    return parser.parse_args()


def _prepare_target_data(
    rna_path: str,
    prot_path: str,
    target: str,
    value_type: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logging.info("Loading RNA from %s", rna_path)
    rna = load_matrix_parquet(rna_path)
    logging.info("Loading proteomics from %s", prot_path)
    prot = load_matrix_parquet(prot_path)

    rna.columns = _clean_sample_ids(rna.columns)
    prot.columns = _clean_sample_ids(prot.columns)

    rna = _prepare_feature_matrix(rna, prefer_value_type=value_type)
    prot = _prepare_feature_matrix(prot, prefer_value_type=value_type)

    common_samples = sorted(set(rna.columns) & set(prot.columns))
    if not common_samples:
        raise ValueError("No overlapping sample IDs between RNA and proteomics.")

    rna = rna.loc[:, common_samples]
    prot = prot.loc[:, common_samples]

    missing_frac = rna.isna().mean(axis=0)
    keep_samples = missing_frac[missing_frac <= 0.2].index
    rna = rna.loc[:, keep_samples]
    prot = prot.loc[:, keep_samples]

    resolved = _resolve_target(prot.index, target)
    if resolved is None:
        raise ValueError(f"Target {target} not found in proteomics.")

    y = prot.loc[resolved].dropna()
    X = rna.loc[:, y.index].T
    y = y.loc[X.index]
    X = X.loc[~X.index.duplicated(keep="first")]
    y = y.loc[~y.index.duplicated(keep="first")]
    X, y = X.align(y, join="inner", axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )
    X_train, X_test = _median_impute_after_split(X_train, X_test)
    return X_train, X_test, y_train, y_test


def _fit_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    baseline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge()),
        ]
    )
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    return {
        "baseline_r2": float(r2_score(y_test, baseline_pred)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline_pred)),
        "baseline_pearson": float(_pearson_corr(y_test.values, baseline_pred)),
    }


def _initial_prefilter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    start_features: int,
) -> pd.Index:
    if start_features >= X_train.shape[1]:
        return pd.Index(X_train.columns.astype(str))

    scores, _ = f_regression(X_train, y_train)
    score_series = pd.Series(scores, index=X_train.columns).replace([np.inf, -np.inf], np.nan)
    score_series = score_series.fillna(-np.inf).sort_values(ascending=False)
    return pd.Index(score_series.head(start_features).index.astype(str))


def _fit_tuned_rf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_iter: int,
    cv: int,
    seed: int,
    n_jobs: int,
    best_params: Optional[dict] = None,
    final_n_estimators: int = 800,
    retune: bool = True,
) -> tuple[dict, pd.DataFrame, dict]:
    model = RandomForestRegressor(
        random_state=seed,
        n_jobs=n_jobs,
        oob_score=False,
    )
    best_cv_r2 = np.nan
    if retune or best_params is None:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions={
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 40],
                "min_samples_split": [5, 10],
                "min_samples_leaf": [4, 8],
                "max_features": [0.5, 0.8],
                "bootstrap": [True, False],
            },
            n_iter=n_iter,
            scoring="r2",
            cv=cv,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
        best_cv_r2 = float(search.best_score_)

    fit_params = dict(best_params)
    fit_params["n_estimators"] = final_n_estimators
    best_model = RandomForestRegressor(
        random_state=seed,
        n_jobs=n_jobs,
        oob_score=False,
        **fit_params,
    )
    best_model.fit(X_train, y_train)
    rf_pred = best_model.predict(X_test)

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns.astype(str),
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance_df["importance_pct"] = importance_df["importance"] * 100.0
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    metrics = {
        "rf_r2": float(r2_score(y_test, rf_pred)),
        "rf_mae": float(mean_absolute_error(y_test, rf_pred)),
        "rf_pearson": float(_pearson_corr(y_test.values, rf_pred)),
        "rf_best_cv_r2": best_cv_r2,
        "rf_best_params": json.dumps(best_params),
    }
    return metrics, importance_df, best_params


def _write_outputs(
    outdir: Path,
    metrics_rows: list[dict],
    feature_rows: list[pd.DataFrame],
    target: str,
    descending_k: bool,
) -> None:
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(outdir / "metrics.csv", index=False)

    if feature_rows:
        pd.concat(feature_rows, ignore_index=True).to_csv(
            outdir / "feature_importance.csv",
            index=False,
        )

    if metrics_df.empty:
        return

    plot_df = metrics_df.sort_values("k", ascending=not descending_k)
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["k"], plot_df["rf_r2"], marker="o")
    plt.xticks(plot_df["k"].tolist())
    plt.xlabel("k (remaining RNA features)")
    plt.ylabel("RF test R^2")
    plt.title(f"{target} RF performance vs k")
    plt.tight_layout()
    plt.savefig(outdir / f"elbow_{target}_rf_r2.png", dpi=150)
    plt.close()


def main() -> int:
    _setup_logging()
    args = _parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = _prepare_target_data(
        rna_path=args.rna,
        prot_path=args.prot,
        target=args.target,
        value_type=args.value_type,
        test_size=args.test_size,
        seed=args.seed,
    )
    baseline_metrics = _fit_baseline(X_train, X_test, y_train, y_test)

    feature_names = _initial_prefilter(
        X_train=X_train,
        y_train=y_train,
        start_features=args.start_features,
    )
    metrics_rows: list[dict] = []
    feature_rows: list[pd.DataFrame] = []
    best_params: Optional[dict] = None
    step_index = 0

    while len(feature_names) >= args.min_features:
        current_train = X_train.loc[:, feature_names]
        current_test = X_test.loc[:, feature_names]
        current_k = len(feature_names)
        retune = step_index == 0 or (step_index % args.retune_every == 0)
        logging.info("Fitting tuned RF for k=%s (retune=%s)", current_k, retune)

        rf_metrics, importance_df, best_params = _fit_tuned_rf(
            current_train,
            current_test,
            y_train,
            y_test,
            n_iter=args.n_iter,
            cv=args.cv,
            seed=args.seed,
            n_jobs=args.n_jobs,
            best_params=best_params,
            final_n_estimators=args.final_n_estimators,
            retune=retune,
        )

        target_in_features = args.target.upper() in set(importance_df["feature"].str.upper())
        metrics_rows.append(
            {
                "target": args.target,
                "k": current_k,
                "n_samples_train": int(len(y_train)),
                "n_samples_test": int(len(y_test)),
                **baseline_metrics,
                **rf_metrics,
                "target_rna_present": bool(target_in_features),
            }
        )

        kept_for_next = max(current_k - args.drop_step, args.min_features)
        importance_with_meta = importance_df.copy()
        importance_with_meta.insert(0, "target", args.target)
        importance_with_meta.insert(1, "k", current_k)
        importance_with_meta["kept_for_next_round"] = importance_with_meta["rank"] <= kept_for_next
        feature_rows.append(importance_with_meta)

        _write_outputs(
            outdir=outdir,
            metrics_rows=metrics_rows,
            feature_rows=feature_rows,
            target=args.target,
            descending_k=args.descending_k,
        )

        if current_k == args.min_features:
            break
        feature_names = pd.Index(importance_df.head(kept_for_next)["feature"].tolist())
        step_index += 1

    logging.info("Wrote sweep outputs to %s", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
