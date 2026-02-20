from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

# Loads a parquet file into a DataFrame
def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

# Normalizes RNA table columns to ensure they are at gene-level
def _normalize_rna_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("RNA table columns must be a MultiIndex.")

    if df.columns.nlevels == 2:
        return df

    if df.columns.nlevels == 3:
        # Collapse transcript-level data to gene-level by averaging transcripts.
        # Use stack to avoid memory-intensive transpose on large DataFrames
        level_0 = df.columns.get_level_values(0)
        level_1 = df.columns.get_level_values(1)
        level_2 = df.columns.get_level_values(2)
        
        # Create grouping key and average
        result_data = {}
        for val_type in level_0.unique():
            for db_id in level_2[level_0 == val_type].unique():
                mask = (level_0 == val_type) & (level_2 == db_id)
                avg_col = df.iloc[:, mask].mean(axis=1)
                result_data[(val_type, db_id)] = avg_col
        
        collapsed = pd.DataFrame(result_data)
        collapsed.columns = pd.MultiIndex.from_tuples(
            collapsed.columns, names=["value_type", "Database_ID"]
        )
        return collapsed

    raise ValueError(f"Unsupported RNA column levels: {df.columns.nlevels}")

# Adds a cancer type column to the DataFrame
def _add_cancer_type_column(df: pd.DataFrame, cancer_type: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        col = ("cancer_type",) + ("",) * (df.columns.nlevels - 1)
        if col not in df.columns:
            df[col] = cancer_type
        return df
    if "cancer_type" not in df.columns:
        df["cancer_type"] = cancer_type
    return df

# Combines RNA tables from multiple datasets into a single DataFrame
def _combine_rna_tables(raw_root: Path, datasets: list[str]) -> pd.DataFrame:
    rna_tables_raw = []
    rna_tables_z = []
    cancer_types = []
    
    for name in datasets:
        transcript_path = raw_root / name / "transcriptomics.parquet"
        if not transcript_path.exists():
            logging.warning("Missing transcriptomics for %s", name)
            continue
        df = _load_parquet(transcript_path)
        df = _normalize_rna_columns(df)
        
        # Normalize this dataset's RNA independently (per-gene z-score, per-dataset)
        # Why: Different datasets may have different preprocessing, so we normalize each before combining
        mean = df.mean(axis=0, skipna=True)
        std = df.std(axis=0, skipna=True).replace(0, np.nan)
        df_z = ((df - mean) / std).astype("float32")
        
        rna_tables_raw.append(df)
        rna_tables_z.append(df_z)
        cancer_types.extend([name] * len(df))

    if not rna_tables_raw:
        raise RuntimeError("No RNA tables found to combine.")

    # Combine raw and z-normalized versions
    combined_raw = pd.concat(rna_tables_raw, axis=0, join="outer")
    combined_z = pd.concat(rna_tables_z, axis=0, join="outer")
    
    # Add cancer type metadata
    combined_raw[("cancer_type", "")] = cancer_types
    
    if isinstance(combined_raw.columns, pd.MultiIndex):
        # Create multi-index columns for raw and z
        raw_cols = pd.MultiIndex.from_arrays(
            [
                ["raw"] * (combined_raw.shape[1] - 1),  # -1 for cancer_type column
                combined_raw.columns.get_level_values(0)[:-1],
                combined_raw.columns.get_level_values(1)[:-1],
            ],
            names=["value_type", "Name", "Database_ID"],
        )
        z_cols = pd.MultiIndex.from_arrays(
            [
                ["z"] * combined_z.shape[1],
                combined_z.columns.get_level_values(0),
                combined_z.columns.get_level_values(1),
            ],
            names=["value_type", "Name", "Database_ID"],
        )
        
        # Create metadata column
        meta_cols = pd.MultiIndex.from_tuples(
            [("meta", "cancer_type", "")],
            names=["value_type", "Name", "Database_ID"],
        )
        meta_block = pd.DataFrame(
            combined_raw[("cancer_type", "")].values, 
            index=combined_raw.index, 
            columns=meta_cols
        )
        
        # Get raw and z blocks (exclude cancer_type)
        raw_block = combined_raw.iloc[:, :-1]
        raw_block.columns = raw_cols
        z_block = combined_z
        z_block.columns = z_cols
        
        return pd.concat([meta_block, raw_block, z_block], axis=1)
    
    return combined_raw

# Combines protein tables from multiple datasets into a single DataFrame
def _combine_protein_tables(raw_root: Path, datasets: list[str]) -> pd.DataFrame:
    protein_tables = []
    for name in datasets:
        protein_path = raw_root / name / "proteomics.parquet"
        if not protein_path.exists():
            logging.warning("Missing proteomics for %s", name)
            continue
        df = _load_parquet(protein_path)
        
        # Normalize each dataset independently (per-gene z-score)
        # Why: Different datasets may be on different scales (some raw counts, some normalized)
        # This ensures all datasets are comparable before combining
        mean = df.mean(axis=0, skipna=True)
        std = df.std(axis=0, skipna=True).replace(0, np.nan)
        df_normalized = ((df - mean) / std).astype("float32")
        
        protein_tables.append(df_normalized)

    if not protein_tables:
        raise RuntimeError("No protein tables found to combine.")

    return pd.concat(protein_tables, axis=0, join="outer")


def main() -> int:
    _setup_logging()
    project_root = Path(__file__).resolve().parent
    raw_root = project_root / "data" / "raw"
    processed_root = project_root / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    datasets = sorted([p.name for p in raw_root.iterdir() if p.is_dir()])
    logging.info("Found datasets: %s", ", ".join(datasets))

    rna_combined = _combine_rna_tables(raw_root, datasets)
    protein_combined = _combine_protein_tables(raw_root, datasets)

    rna_path = processed_root / "all_rna.parquet"
    protein_path = processed_root / "all_protein.parquet"
    rna_combined.to_parquet(rna_path)
    protein_combined.to_parquet(protein_path)

    summary = {
        "all_rna": {
            "rows": int(rna_combined.shape[0]),
            "cols": int(rna_combined.shape[1]),
            "path": str(rna_path.resolve()),
        },
        "all_protein": {
            "rows": int(protein_combined.shape[0]),
            "cols": int(protein_combined.shape[1]),
            "path": str(protein_path.resolve()),
        },
    }
    summary_path = processed_root / "processed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info("Wrote processed summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
