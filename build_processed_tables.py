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


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _normalize_rna_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("RNA table columns must be a MultiIndex.")

    if df.columns.nlevels == 2:
        return df

    if df.columns.nlevels == 3:
        # Collapse transcript-level data to gene-level by averaging transcripts.
        collapsed = df.groupby(level=[0, 2], axis=1).mean()
        collapsed.columns = pd.MultiIndex.from_tuples(
            collapsed.columns, names=["Name", "Database_ID"]
        )
        return collapsed

    raise ValueError(f"Unsupported RNA column levels: {df.columns.nlevels}")


def _add_cancer_type_column(df: pd.DataFrame, cancer_type: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        col = ("cancer_type",) + ("",) * (df.columns.nlevels - 1)
        if col not in df.columns:
            df[col] = cancer_type
        return df
    if "cancer_type" not in df.columns:
        df["cancer_type"] = cancer_type
    return df


def _combine_rna_tables(raw_root: Path, datasets: list[str]) -> pd.DataFrame:
    rna_tables = []
    for name in datasets:
        transcript_path = raw_root / name / "transcriptomics.parquet"
        if not transcript_path.exists():
            logging.warning("Missing transcriptomics for %s", name)
            continue
        df = _load_parquet(transcript_path)
        df = _normalize_rna_columns(df)
        df = _add_cancer_type_column(df, name)
        rna_tables.append(df)

    if not rna_tables:
        raise RuntimeError("No RNA tables found to combine.")

    combined = pd.concat(rna_tables, axis=0, join="outer")

    cancer_col = ("cancer_type", "") if isinstance(combined.columns, pd.MultiIndex) else "cancer_type"
    rna_meta = combined[[cancer_col]].copy()
    rna_numeric = combined.drop(columns=[cancer_col]).astype("float32")

    mean = rna_numeric.mean(axis=0, skipna=True)
    std = rna_numeric.std(axis=0, skipna=True).replace(0, np.nan)
    rna_z = ((rna_numeric - mean) / std).astype("float32")

    if isinstance(rna_numeric.columns, pd.MultiIndex):
        raw_cols = pd.MultiIndex.from_arrays(
            [
                ["raw"] * len(rna_numeric.columns),
                rna_numeric.columns.get_level_values(0),
                rna_numeric.columns.get_level_values(1),
            ],
            names=["value_type", *rna_numeric.columns.names],
        )
        z_cols = pd.MultiIndex.from_arrays(
            [
                ["z"] * len(rna_z.columns),
                rna_z.columns.get_level_values(0),
                rna_z.columns.get_level_values(1),
            ],
            names=["value_type", *rna_z.columns.names],
        )
        rna_numeric.columns = raw_cols
        rna_z.columns = z_cols
        meta_cols = pd.MultiIndex.from_tuples(
            [("meta", "cancer_type", "")],
            names=["value_type", "Name", "Database_ID"],
        )
        meta_block = pd.DataFrame(rna_meta.values, index=rna_meta.index, columns=meta_cols)
        return pd.concat([meta_block, rna_numeric, rna_z], axis=1)

    rna_z = rna_z.add_suffix("_z")
    return pd.concat([rna_meta, rna_numeric, rna_z], axis=1)


def _combine_protein_tables(raw_root: Path, datasets: list[str]) -> pd.DataFrame:
    protein_tables = []
    for name in datasets:
        protein_path = raw_root / name / "proteomics.parquet"
        if not protein_path.exists():
            logging.warning("Missing proteomics for %s", name)
            continue
        df = _load_parquet(protein_path)
        protein_tables.append(df)

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
