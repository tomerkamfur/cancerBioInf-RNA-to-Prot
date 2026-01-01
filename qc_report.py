from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _sample_values(df: pd.DataFrame, max_rows: int = 200, max_cols: int = 2000) -> pd.Series:
    sampled = df
    if df.shape[0] > max_rows:
        sampled = sampled.sample(n=max_rows, axis=0, random_state=0)
    if df.shape[1] > max_cols:
        sampled = sampled.sample(n=max_cols, axis=1, random_state=0)
    values = sampled.to_numpy().ravel()
    return pd.Series(values)


def _safe_histogram(df: pd.DataFrame, title: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed; skipping histogram %s", out_path.name)
        return

    values = _sample_values(df).dropna().astype(float)
    if values.empty:
        return

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=40, color="#3b6ea8", edgecolor="white", linewidth=0.5)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _qc_for_table(df: pd.DataFrame) -> dict:
    missing = df.isna().sum().sum()
    total = df.shape[0] * df.shape[1]
    missing_frac = float(missing / total) if total else 0.0

    per_sample_missing = df.isna().mean(axis=1)
    per_gene_missing = df.isna().mean(axis=0)

    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_count": int(missing),
        "missing_fraction": missing_frac,
        "per_sample_missing_mean": float(per_sample_missing.mean()),
        "per_sample_missing_median": float(per_sample_missing.median()),
        "per_gene_missing_mean": float(per_gene_missing.mean()),
        "per_gene_missing_median": float(per_gene_missing.median()),
    }


def main() -> int:
    _setup_logging()
    project_root = Path(__file__).resolve().parent
    raw_root = project_root / "data" / "raw"
    qc_root = project_root / "data" / "processed" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)

    summary = {}

    for dataset_dir in sorted(raw_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        transcript_path = dataset_dir / "transcriptomics.parquet"
        proteomics_path = dataset_dir / "proteomics.parquet"

        if not transcript_path.exists() or not proteomics_path.exists():
            continue

        logging.info("QC for %s", dataset_name)
        transcriptomics = pd.read_parquet(transcript_path)
        proteomics = pd.read_parquet(proteomics_path)

        dataset_summary = {}
        dataset_summary["transcriptomics"] = _qc_for_table(transcriptomics)
        dataset_summary["proteomics"] = _qc_for_table(proteomics)

        rna_samples = set(transcriptomics.index.astype(str))
        prot_samples = set(proteomics.index.astype(str))
        overlap = rna_samples & prot_samples

        dataset_summary["sample_overlap"] = {
            "rna_samples": len(rna_samples),
            "protein_samples": len(prot_samples),
            "overlap_samples": len(overlap),
            "overlap_fraction_rna": float(len(overlap) / len(rna_samples)) if rna_samples else 0.0,
            "overlap_fraction_protein": float(len(overlap) / len(prot_samples)) if prot_samples else 0.0,
        }

        out_dir = qc_root / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        _safe_histogram(
            transcriptomics,
            f"{dataset_name} RNA value distribution",
            out_dir / "rna_value_hist.png",
        )
        _safe_histogram(
            proteomics,
            f"{dataset_name} Protein value distribution",
            out_dir / "protein_value_hist.png",
        )

        summary[dataset_name] = dataset_summary

    summary_path = qc_root / "qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info("Wrote QC summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
