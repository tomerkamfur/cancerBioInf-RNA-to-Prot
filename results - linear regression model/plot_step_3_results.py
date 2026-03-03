"""
Create graphs from step_3_results.txt.

Outputs are written into the same folder as this script:
- step_3_parsed_metrics.csv
- step_3_cv_r2_by_protein.png
- step_3_selected_features_by_protein.png
- step_3_model_vs_baselines_r2.png
"""

from pathlib import Path
import ast
import re

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
INPUT_FILE = RESULTS_DIR / "step_3_results.txt"


def parse_results_text(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()

    rows = []
    current = None

    header_re = re.compile(r"^--- Protein \d+/\d+: (.+) ---$")
    baseline_re = re.compile(r"^\s+(.+): mean-baseline R² = ([-+]?\d*\.?\d+)")
    perm_re = re.compile(r"^\s+(.+): permuted-label R² = ([-+]?\d*\.?\d+)")

    baseline_map = {}
    perm_map = {}

    for line in lines:
        m = header_re.match(line)
        if m:
            if current and current.get("cv_mean_r2") is not None:
                rows.append(current)
            current = {"protein_raw": m.group(1), "cv_mean_r2": None, "cv_mean_features": None}
            continue

        if current and "Mean R²:" in line:
            current["cv_mean_r2"] = float(line.split("Mean R²:")[1].strip())
            continue

        if current and "Mean features selected:" in line:
            current["cv_mean_features"] = float(line.split("Mean features selected:")[1].strip())
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
        return pd.DataFrame(columns=["protein_raw", "protein_label", "cv_mean_r2", "cv_mean_features"])

    df = pd.DataFrame(rows)
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
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = parse_results_text(INPUT_FILE)
    if df.empty:
        raise ValueError("No protein metrics were parsed from step_3_results.txt")

    df.to_csv(RESULTS_DIR / "step_3_parsed_metrics.csv", index=False)

    plot_bar(
        df,
        value_col="cv_mean_r2",
        title="Cross-Validation Mean R² per Protein",
        y_label="Mean R² (5-fold CV)",
        out_path=RESULTS_DIR / "step_3_cv_r2_by_protein.png",
    )
    plot_bar(
        df,
        value_col="cv_mean_features",
        title="Mean Selected Features per Protein",
        y_label="Mean selected features",
        out_path=RESULTS_DIR / "step_3_selected_features_by_protein.png",
    )
    plot_model_vs_baseline(df, RESULTS_DIR / "step_3_model_vs_baselines_r2.png")

    print(f"Saved parsed metrics and graphs to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
