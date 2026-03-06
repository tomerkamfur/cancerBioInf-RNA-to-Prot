"""
Create graphs from:
- step_3_results.txt
- step_3_results_chosen_proteins.txt (for RUN_ONLY_STEP_6 runs)

Outputs are written into the same folder as this script:
- step_3_parsed_metrics.csv
- step_3_cv_r2_by_protein.png
- step_3_selected_features_by_protein.png
- step_3_model_vs_baselines_r2.png
- step_3_feature_selection_comparison_r2.png
"""

from pathlib import Path
import ast
import re

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
PRIMARY_RESULTS_FILE = RESULTS_DIR / "step_3_results.txt"
CHOSEN_PROTEINS_RESULTS_FILE = RESULTS_DIR / "step_3_results_chosen_proteins.txt"


def parse_results_text(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()

    rows = []
    current = None
    current_mode = None
    current_fold = None

    mode_re = re.compile(r"^Modeling .*? \((.+)\):")
    header_re = re.compile(r"^--- Protein \d+/\d+: (.+) ---$")
    baseline_re = re.compile(r"^\s+(.+): mean-baseline R² = ([-+]?\d*\.?\d+)")
    perm_re = re.compile(r"^\s+(.+): permuted-label R² = ([-+]?\d*\.?\d+)")
    fold_start_re = re.compile(r"^\s*Fold (\d+) performance:")
    fold_r2_re = re.compile(r"^\s+R² = ([-+]?\d*\.?\d+)")
    fold_rmse_re = re.compile(r"^\s+RMSE = ([-+]?\d*\.?\d+)")
    fold_mae_re = re.compile(r"^\s+MAE = ([-+]?\d*\.?\d+)")
    fold_features_re = re.compile(r"^\s+(?:Selected features|Non-zero coefficients): (\d+) / (\d+) \(~")
    fold_alpha_re = re.compile(r"^\s+Best alpha: ([-+]?\d*\.?\d+)")
    fold_l1_re = re.compile(r"^\s+Best l1_ratio: ([-+]?\d*\.?\d+)")

    baseline_map = {}
    perm_map = {}

    for line in lines:
        m = header_re.match(line)
        if m:
            if current and current.get("cv_mean_r2") is not None:
                rows.append(current)
            current = {
                "protein_raw": m.group(1),
                "selection_mode": current_mode,
                "cv_mean_r2": None,
                "cv_std_r2": None,
                "cv_min_r2": None,
                "cv_max_r2": None,
                "cv_mean_features": None,
                "cv_std_features": None,
                "fold_r2": [],
                "fold_rmse": [],
                "fold_mae": [],
                "fold_selected_features": [],
                "fold_alpha": [],
                "fold_l1_ratio": [],
            }
            current_fold = None
            continue

        m_mode = mode_re.match(line)
        if m_mode:
            mode_token = m_mode.group(1).strip().lower()
            if "no pre-filter" in mode_token or "no prefilter" in mode_token:
                current_mode = "no_pre_filter"
            elif "as examples" in mode_token:
                current_mode = "variance_filter"
            else:
                current_mode = mode_token
            continue

        f = fold_start_re.match(line)
        if f:
            if current is not None:
                current_fold = int(f.group(1))
            continue

        if current is not None and current_fold is not None:
            r2_match = fold_r2_re.match(line)
            if r2_match:
                current["fold_r2"].append(float(r2_match.group(1)))
                continue
            rmse_match = fold_rmse_re.match(line)
            if rmse_match:
                current["fold_rmse"].append(float(rmse_match.group(1)))
                continue
            mae_match = fold_mae_re.match(line)
            if mae_match:
                current["fold_mae"].append(float(mae_match.group(1)))
                continue
            feat_match = fold_features_re.match(line)
            if feat_match:
                current["fold_selected_features"].append(int(feat_match.group(1)))
                continue
            alpha_match = fold_alpha_re.match(line)
            if alpha_match:
                current["fold_alpha"].append(float(alpha_match.group(1)))
                continue
            l1_match = fold_l1_re.match(line)
            if l1_match:
                current["fold_l1_ratio"].append(float(l1_match.group(1)))
                continue

        if current and "Mean R²:" in line:
            current["cv_mean_r2"] = float(line.split("Mean R²:")[1].strip())
            continue

        if current and "Std R²:" in line:
            current["cv_std_r2"] = float(line.split("Std R²:")[1].strip())
            continue

        if current and "Min R²:" in line:
            current["cv_min_r2"] = float(line.split("Min R²:")[1].strip())
            continue

        if current and ("Mean non-zero coefficients:" in line or "Mean features selected:" in line):
            if "Mean non-zero coefficients:" in line:
                current["cv_mean_features"] = float(line.split("Mean non-zero coefficients:")[1].strip())
            else:
                current["cv_mean_features"] = float(line.split("Mean features selected:")[1].strip())
            continue

        if current and ("Std non-zero coefficients:" in line or "Std features selected:" in line):
            if "Std non-zero coefficients:" in line:
                current["cv_std_features"] = float(line.split("Std non-zero coefficients:")[1].strip())
            else:
                current["cv_std_features"] = float(line.split("Std features selected:")[1].strip())
            continue

        b = baseline_re.match(line)
        if b:
            baseline_map[b.group(1)] = float(b.group(2))
            continue

        p = perm_re.match(line)
        if p:
            perm_map[p.group(1)] = float(p.group(2))
            continue

    if current and current.get("cv_mean_r2") is not None:
            rows.append(current)

    if not rows:
        return pd.DataFrame(
            columns=[
                "selection_mode",
                "protein_raw",
                "protein_label",
                "cv_mean_r2",
                "cv_std_r2",
                "cv_min_r2",
                "cv_max_r2",
                "cv_mean_features",
                "cv_std_features",
                "fold_r2",
                "fold_rmse",
                "fold_mae",
                "fold_selected_features",
                "fold_alpha",
                "fold_l1_ratio",
                "baseline_r2",
                "permuted_r2",
            ]
        )

    df = pd.DataFrame(rows)
    # Convert metric arrays into numeric columns to keep compatibility with plotting.
    for i in range(1, 6):
        df[f"cv_r2_fold_{i}"] = df["fold_r2"].apply(lambda vals: vals[i - 1] if isinstance(vals, list) and len(vals) >= i else None)
        df[f"cv_rmse_fold_{i}"] = df["fold_rmse"].apply(lambda vals: vals[i - 1] if isinstance(vals, list) and len(vals) >= i else None)
        df[f"cv_mae_fold_{i}"] = df["fold_mae"].apply(lambda vals: vals[i - 1] if isinstance(vals, list) and len(vals) >= i else None)
        df[f"cv_features_fold_{i}"] = df["fold_selected_features"].apply(lambda vals: vals[i - 1] if isinstance(vals, list) and len(vals) >= i else None)

    df["protein_label"] = df["protein_raw"].apply(make_protein_label)
    df["baseline_r2"] = df["protein_raw"].map(baseline_map)
    df["permuted_r2"] = df["protein_raw"].map(perm_map)
    return df


def make_protein_label(raw: str) -> str:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, tuple) and len(parsed) >= 2:
            return f"{parsed[0]}\n{parsed[1]}"
        if isinstance(parsed, tuple) and len(parsed) == 1:
            return str(parsed[0])
    except (SyntaxError, ValueError):
        pass
    return raw


def parse_multiple_result_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for order, path in enumerate(paths):
        if not path.exists():
            continue
        df = parse_results_text(path)
        if not df.empty:
            df["source_file"] = path.name
            df["source_order"] = order
            frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "protein_raw",
                "protein_label",
                "cv_mean_r2",
                "cv_std_r2",
                "cv_min_r2",
                "cv_max_r2",
                "cv_mean_features",
                "cv_std_features",
                "fold_r2",
                "fold_rmse",
                "fold_mae",
                "fold_selected_features",
                "fold_alpha",
                "fold_l1_ratio",
                "baseline_r2",
                "permuted_r2",
            ]
        )

    all_results = pd.concat(frames, ignore_index=True)
    all_results = all_results.sort_values("source_order", ascending=True)
    all_results = all_results.drop_duplicates(subset=["protein_raw", "selection_mode"], keep="last")
    return all_results.drop(columns=["source_order"])


def normalize_mode(mode: str) -> str:
    mode_l = str(mode).replace("_", " ").replace("-", " ").lower()
    if (
        ("no pre filter" in mode_l and "manual" not in mode_l)
        or ("no prefilter" in mode_l and "manual" not in mode_l)
        or ("no filter" in mode_l and "manual" not in mode_l)
    ):
        return "no_feature_selection"
    if "as examples" in mode_l or "variance filter" in mode_l or "filtered" in mode_l:
        return "manual_variance_filter"
    return "other"


def plot_feature_selection_comparison(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return

    comp = df.copy()
    comp["selection_mode_norm"] = comp["selection_mode"].apply(normalize_mode)
    comp = comp[comp["selection_mode_norm"].isin(["manual_variance_filter", "no_feature_selection"])]
    if comp.empty:
        return

    pivot = comp.pivot_table(
        index=["protein_raw", "protein_label"],
        columns="selection_mode_norm",
        values="cv_mean_r2",
        aggfunc="mean",
    )
    for col in ["manual_variance_filter", "no_feature_selection"]:
        if col not in pivot.columns:
            pivot[col] = pd.NA
    pivot = pivot.dropna(subset=["manual_variance_filter", "no_feature_selection"], how="any")

    if pivot.empty:
        return

    pivot = pivot.reset_index().sort_values("no_feature_selection", ascending=False)
    x = range(len(pivot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot)), 6))
    ax.bar([i - width / 2 for i in x], pivot["manual_variance_filter"], width=width, label="Manual variance filter")
    ax.bar([i + width / 2 for i in x], pivot["no_feature_selection"], width=width, label="No manual filter")
    ax.set_title("Mean CV R²: manual variance feature selection vs no manual feature selection")
    ax.set_ylabel("Mean R²")
    ax.set_xlabel("Protein target")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot["protein_label"], rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.4)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bar(df: pd.DataFrame, value_col: str, title: str, y_label: str, out_path: Path) -> None:
    if df.empty:
        return

    plot_df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    width = max(8, 0.75 * len(plot_df))
    fig, ax = plt.subplots(figsize=(width, 6))

    ax.bar(plot_df["protein_label"], plot_df[value_col], color="#2a9d8f")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Protein target")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model_vs_baseline(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.dropna(subset=["baseline_r2", "permuted_r2"]).copy()
    if plot_df.empty:
        return

    width = max(8, 0.9 * len(plot_df))
    fig, ax = plt.subplots(figsize=(width, 6))
    x = range(len(plot_df))
    bar_w = 0.25

    ax.bar([i - bar_w for i in x], plot_df["cv_mean_r2"], width=bar_w, label="Model CV mean R²", color="#2a9d8f")
    ax.bar(x, plot_df["baseline_r2"], width=bar_w, label="Mean baseline R²", color="#457b9d")
    ax.bar([i + bar_w for i in x], plot_df["permuted_r2"], width=bar_w, label="Permuted-label R²", color="#e76f51")

    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["protein_label"], rotation=45, ha="right")
    ax.set_ylabel("R²")
    ax.set_xlabel("Protein target")
    ax.set_title("Elastic Net vs Baseline/Permutation")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    result_files = [PRIMARY_RESULTS_FILE, CHOSEN_PROTEINS_RESULTS_FILE]
    df = parse_multiple_result_files(result_files)
    if df.empty:
        raise ValueError(
            f"No protein metrics were parsed from available result files: {', '.join(str(p) for p in result_files)}"
        )

    # Recompute labels after merging so any source-specific parsing stays consistent.
    df["protein_label"] = df["protein_raw"].apply(make_protein_label)
    # Keep step-6/user-selected + variance-filtered step-3 results for overview plots,
    # and reserve no-feature-selection mode for the dedicated comparison plot.
    df_primary = df.copy()
    df_main = df_primary[df_primary["selection_mode"] != "no_pre_filter"].copy()
    if df_main.empty:
        df_main = df_primary

    df.to_csv(RESULTS_DIR / "step_3_parsed_metrics.csv", index=False)

    plot_bar(
        df_main,
        value_col="cv_mean_r2",
        title="Cross-Validation Mean R² per Protein",
        y_label="Mean R² (5-fold CV)",
        out_path=RESULTS_DIR / "step_3_cv_r2_by_protein.png",
    )
    plot_bar(
        df_main,
        value_col="cv_mean_features",
        title="Mean Selected Features per Protein",
        y_label="Mean selected features",
        out_path=RESULTS_DIR / "step_3_selected_features_by_protein.png",
    )
    plot_model_vs_baseline(df_main, RESULTS_DIR / "step_3_model_vs_baselines_r2.png")

    plot_feature_selection_comparison(
        df_primary,
        RESULTS_DIR / "step_3_feature_selection_comparison_r2.png",
    )

    print(f"Saved parsed metrics and graphs to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
