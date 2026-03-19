from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

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
        description="Run RF on the top-N features from a saved feature importance file.",
    )
    parser.add_argument("--rna", type=str, default="data/from_may/all_rna.parquet")
    parser.add_argument("--prot", type=str, default="data/from_may/all_protein.parquet")
    parser.add_argument("--feature-importance", type=str, required=True)
    parser.add_argument("--target", type=str, default="KRAS")
    parser.add_argument("--top-n", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--value-type", choices=["raw", "z"], default="z")
    return parser.parse_args()


def _save_pred_plot(out_dir: Path, target: str, y_true, y_pred) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.4)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
    plt.xlabel("True protein abundance")
    plt.ylabel("Predicted protein abundance")
    plt.title(f"{target} predicted vs true (top features)")
    plt.tight_layout()
    plt.savefig(out_dir / f"pred_vs_true_{target}.png", dpi=150)
    plt.close()


def main() -> int:
    _setup_logging()
    args = _parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_features_df = pd.read_csv(args.feature_importance)
    if "feature" not in top_features_df.columns:
        raise ValueError("feature importance CSV must contain a 'feature' column.")
    selected_features = top_features_df["feature"].astype(str).head(args.top_n).tolist()
    logging.info("Using top %s features for %s", len(selected_features), args.target)

    rna = load_matrix_parquet(args.rna)
    prot = load_matrix_parquet(args.prot)
    rna.columns = _clean_sample_ids(rna.columns)
    prot.columns = _clean_sample_ids(prot.columns)

    rna = _prepare_feature_matrix(rna, prefer_value_type=args.value_type)
    prot = _prepare_feature_matrix(prot, prefer_value_type=args.value_type)

    common_samples = sorted(set(rna.columns) & set(prot.columns))
    rna = rna.loc[:, common_samples]
    prot = prot.loc[:, common_samples]

    missing_frac = rna.isna().mean(axis=0)
    keep_samples = missing_frac[missing_frac <= 0.2].index
    rna = rna.loc[:, keep_samples]
    prot = prot.loc[:, keep_samples]

    resolved = _resolve_target(prot.index, args.target)
    if resolved is None:
        raise ValueError(f"Target {args.target} not found in proteomics.")

    y = prot.loc[resolved].dropna()
    feature_subset = [feature for feature in selected_features if feature in rna.index]
    if not feature_subset:
        raise ValueError("None of the selected features were found in the RNA matrix.")

    X = rna.loc[feature_subset, y.index].T
    y = y.loc[X.index]
    X = X.loc[~X.index.duplicated(keep="first")]
    y = y.loc[~y.index.duplicated(keep="first")]
    X, y = X.align(y, join="inner", axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    X_train, X_test = _median_impute_after_split(X_train, X_test)

    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(
            random_state=args.seed,
            n_jobs=args.n_jobs,
            oob_score=False,
        ),
        param_distributions={
            "n_estimators": [300, 500, 800, 1200],
            "max_depth": [None, 5, 10, 20, 40],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.2, 0.5, 0.8],
            "bootstrap": [True, False],
        },
        n_iter=args.n_iter,
        scoring="r2",
        cv=5,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_model: RandomForestRegressor = search.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = pd.DataFrame(
        [
            {
                "target": args.target,
                "top_n": len(feature_subset),
                "n_samples": int(len(y)),
                "rf_r2": float(r2_score(y_test, y_pred)),
                "rf_mae": float(mean_absolute_error(y_test, y_pred)),
                "rf_pearson": float(_pearson_corr(y_test.values, y_pred)),
                "rf_best_cv_r2": float(search.best_score_),
                "rf_best_params": json.dumps(search.best_params_),
            }
        ]
    )
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns.astype(str),
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(out_dir / f"feature_importance_{args.target}.csv", index=False)
    _save_pred_plot(out_dir, args.target, y_test.values, y_pred)
    logging.info("Saved results to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
