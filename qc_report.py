from __future__ import annotations # for future compatibility with type hints in Python

import json
import logging
from pathlib import Path

import pandas as pd 


# validate data quality and overlap between RNA and protein tables after download, 
# and create a quick report + plots you can inspect.
# total rows/cols
# total missing values + missing fraction
# per‑sample and per‑gene missingness (mean/median)

# Sets up logging configuration
def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

# Samples values from a DataFrame for histogram plotting
def _sample_values(df: pd.DataFrame, max_rows: int = 200, max_cols: int = 2000) -> pd.Series:
    sampled = df
    if df.shape[0] > max_rows:
        sampled = sampled.sample(n=max_rows, axis=0, random_state=0)
    if df.shape[1] > max_cols:
        sampled = sampled.sample(n=max_cols, axis=1, random_state=0)
    values = sampled.to_numpy().ravel()
    return pd.Series(values)

# Creates and saves a histogram plot safely, handling missing matplotlib
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
    processed_root = project_root / "data" / "processed"
    qc_root = processed_root / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)

    summary = {}
# Iterate over each dataset directory in the raw data root
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

    # ============================================================
    # QC for PROCESSED (combined, normalized) data
    # ============================================================
    logging.info("QC for processed combined data...")
    all_rna_path = processed_root / "all_rna.parquet"
    all_protein_path = processed_root / "all_protein.parquet"
    
    if all_rna_path.exists() and all_protein_path.exists():
        all_rna = pd.read_parquet(all_rna_path)
        all_protein = pd.read_parquet(all_protein_path)
        
        # Extract cancer types and create histograms per cancer type
        cancer_type_col = ('meta', 'cancer_type', '')
        if cancer_type_col in all_rna.columns:
            cancer_types = all_rna[cancer_type_col].dropna().unique()
            
            for cancer_type in sorted(cancer_types):
                # Filter to this cancer type
                mask = all_rna[cancer_type_col] == cancer_type
                rna_cancer = all_rna[mask]
                
                # Find matching samples in protein table
                protein_cancer = all_protein.loc[all_protein.index.isin(rna_cancer.index)]
                
                if len(protein_cancer) == 0:
                    logging.warning("No samples for cancer type %s", cancer_type)
                    continue
                
                # Extract z-normalized RNA for this cancer type
                rna_z = rna_cancer[[col for col in rna_cancer.columns if col[0] == 'z']].iloc[:, 1:]
                
                out_dir = qc_root / cancer_type / "processed"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                logging.info("Creating histograms for %s (n=%d samples)", cancer_type, len(protein_cancer))
                
                _safe_histogram(
                    rna_z,
                    f"{cancer_type} RNA z-normalized (normalized)",
                    out_dir / "rna_z_hist.png",
                )
                _safe_histogram(
                    protein_cancer,
                    f"{cancer_type} Protein z-normalized (normalized)",
                    out_dir / "protein_z_hist.png",
                )
                
                # Compute stats for this cancer type
                rna_numeric = rna_z
                protein_numeric = protein_cancer
                
                cancer_summary = {
                    f"{cancer_type}_normalized": {
                        "samples": int(len(protein_cancer)),
                        "rna_cols": int(rna_numeric.shape[1]),
                        "protein_cols": int(protein_numeric.shape[1]),
                        "rna_mean": float(rna_numeric.mean(numeric_only=True).mean()),
                        "rna_std": float(rna_numeric.std(numeric_only=True).mean()),
                        "protein_mean": float(protein_numeric.mean(numeric_only=True).mean()),
                        "protein_std": float(protein_numeric.std(numeric_only=True).mean()),
                        "protein_missing_fraction": float(protein_numeric.isna().sum().sum() / (protein_numeric.shape[0] * protein_numeric.shape[1])),
                    }
                }
                
                cancer_summary_path = out_dir / "normalized_qc.json"
                cancer_summary_path.write_text(json.dumps(cancer_summary, indent=2))
                logging.info("  Saved normalized QC to %s", cancer_summary_path)
        else:
            logging.warning("Cancer type column not found in RNA table")
    else:
        logging.warning("Processed tables not found. Run build_processed_tables.py first.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
