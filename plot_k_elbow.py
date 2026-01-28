from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate k-elbow plots from RF outputs.",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["outputs*"],
        help="Root output directories or glob patterns.",
    )
    parser.add_argument("--metric", type=str, default="r2")
    parser.add_argument("--outdir", type=str, default="elbow_plots")
    parser.add_argument("--logx", action="store_true", help="Use log scale on x-axis.")
    return parser.parse_args()


def _discover_roots(patterns: Iterable[str]) -> List[Path]:
    roots: List[Path] = []
    seen = set()
    for pattern in patterns:
        for path in Path(".").glob(pattern):
            if path.is_dir() and path.resolve() not in seen:
                roots.append(path)
                seen.add(path.resolve())
    return roots


def _parse_k_from_dir(path: Path) -> Optional[int]:
    name = path.name
    if name == "outputs":
        return 2000
    if name.startswith("outputs_k"):
        value = name.replace("outputs_k", "")
        if value.isdigit():
            return int(value)
    return None


def _find_metrics_files(root: Path) -> List[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower().startswith("metrics"):
            if path.suffix.lower() in {".csv", ".json"}:
                files.append(path)
    return files


def _extract_from_csv(path: Path) -> List[Dict]:
    df = pd.read_csv(path)
    if "target" not in df.columns:
        return []
    rows = []
    for idx, row in df.iterrows():
        target = str(row.get("target", "")).strip()
        if not target:
            continue
        rf_r2 = row.get("rf_r2")
        if pd.isna(rf_r2):
            rf_r2 = row.get("r2")
        rf_mae = row.get("rf_mae")
        if pd.isna(rf_mae):
            rf_mae = row.get("mae")
        rows.append(
            {
                "protein": target,
                "iteration": idx + 1,
                "r2": float(rf_r2) if rf_r2 is not None and not pd.isna(rf_r2) else np.nan,
                "mae": float(rf_mae) if rf_mae is not None and not pd.isna(rf_mae) else np.nan,
                "rmse": np.nan,
            }
        )
    return rows


def _extract_from_json(path: Path) -> List[Dict]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]
    if not isinstance(payload, list):
        return []
    rows = []
    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            continue
        target = str(row.get("target", "")).strip()
        if not target:
            continue
        rf_r2 = row.get("rf_r2", row.get("r2"))
        rf_mae = row.get("rf_mae", row.get("mae"))
        rows.append(
            {
                "protein": target,
                "iteration": idx,
                "r2": float(rf_r2) if rf_r2 is not None else np.nan,
                "mae": float(rf_mae) if rf_mae is not None else np.nan,
                "rmse": float(row.get("rmse")) if row.get("rmse") is not None else np.nan,
            }
        )
    return rows


def _load_metrics(root: Path) -> pd.DataFrame:
    k = _parse_k_from_dir(root)
    if k is None:
        return pd.DataFrame()
    records: List[Dict] = []
    for metrics_path in _find_metrics_files(root):
        if metrics_path.suffix.lower() == ".csv":
            records.extend(_extract_from_csv(metrics_path))
        elif metrics_path.suffix.lower() == ".json":
            records.extend(_extract_from_json(metrics_path))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["k"] = k
    return df


def _detect_knee(k_values: List[int], scores: List[float]) -> Optional[int]:
    if len(k_values) < 4:
        return None
    try:
        from kneed import KneeLocator  # type: ignore

        locator = KneeLocator(k_values, scores, curve="concave", direction="increasing")
        return int(locator.knee) if locator.knee is not None else None
    except Exception:
        pass

    deltas = []
    for i in range(1, len(k_values)):
        prev = scores[i - 1]
        curr = scores[i]
        if math.isnan(prev) or math.isnan(curr):
            deltas.append(np.nan)
        else:
            deltas.append(curr - prev)
    for i in range(len(deltas) - 2):
        window = deltas[i : i + 3]
        if all(not math.isnan(v) and v < 0.01 for v in window):
            return k_values[i + 1]
    return None


def _plot_elbow(df: pd.DataFrame, outdir: Path, metric: str, logx: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for protein, group in df.groupby("protein"):
        summary = (
            group.groupby("k")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary = summary[(summary["k"] >= 2) & (summary["k"] <= 20)]
        summary = summary.sort_values("k", ascending=False)
        if summary.empty:
            continue
        k_vals = summary["k"].tolist()
        means = summary["mean"].tolist()
        stds = summary["std"].fillna(0).tolist()

        plt.figure(figsize=(7, 5))
        plt.errorbar(k_vals, means, yerr=stds, fmt="o-", capsize=3)
        if logx:
            plt.xscale("log")
        else:
            plt.xticks(k_vals)
        plt.xlabel("k (selected features)")
        plt.ylabel(f"mean {metric}")
        plt.title(f"Elbow plot for {protein}")
        plt.tight_layout()
        plot_path = outdir / f"elbow_{protein}_{metric}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()


def main() -> int:
    _setup_logging()
    args = _parse_args()

    roots = _discover_roots(args.roots)
    if not roots:
        logging.error("No output directories found.")
        return 1

    frames = []
    for root in roots:
        df = _load_metrics(root)
        if df.empty:
            logging.info("Skipping %s (no metrics found).", root)
            continue
        frames.append(df)

    if not frames:
        logging.error("No valid metrics to plot.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=[args.metric], how="all")

    summary = (
        combined.groupby(["protein", "k"])[args.metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("k")
    )
    logging.info("Summary:\n%s", summary.to_string(index=False))

    _plot_elbow(combined, Path(args.outdir), args.metric, args.logx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
