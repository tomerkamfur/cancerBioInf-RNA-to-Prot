from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

_ENSEMBL_VERSION_RE = re.compile(r"^(ENS[GPT]\d+)\.(\d+)$", re.IGNORECASE)


def standardize_gene_ids(ids: Iterable) -> pd.Index:
    """Normalize gene/protein identifiers and strip Ensembl version suffixes."""
    cleaned = []
    for raw in ids:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            cleaned.append("")
            continue
        value = str(raw).strip()
        match = _ENSEMBL_VERSION_RE.match(value)
        if match:
            value = match.group(1)
        cleaned.append(value)
    return pd.Index(cleaned)


def select_value_type(df: pd.DataFrame, prefer: str = "raw") -> pd.DataFrame:
    """Select a value_type level from a MultiIndex feature index if present."""
    if not isinstance(df.index, pd.MultiIndex):
        return df

    level_names = df.index.names or []
    if "value_type" in level_names:
        level = level_names.index("value_type")
        values = df.index.get_level_values(level)
        if prefer in values:
            return df.loc[values == prefer]
        if "meta" in values:
            return df.loc[values != "meta"]
        return df

    level0 = df.index.get_level_values(0)
    if prefer in level0:
        return df.loc[level0 == prefer]
    if "meta" in level0:
        return df.loc[level0 != "meta"]
    return df


def _find_column(columns: Iterable[str], options: Iterable[str]) -> Optional[str]:
    for option in options:
        if option in columns:
            return option
    return None


def _looks_like_samples_index(index: pd.Index) -> bool:
    name = str(index.name or "").lower()
    if name in {"sample", "sample_id", "patient_id"}:
        return True
    if len(index) == 0:
        return False
    sample = str(index[0])
    return any(char.isdigit() for char in sample) and any(char.isalpha() for char in sample)


def load_matrix_parquet(path: Path | str) -> pd.DataFrame:
    """Load parquet and return features x samples matrix."""
    df = pd.read_parquet(path)

    if isinstance(df.columns, pd.MultiIndex):
        wide = df.select_dtypes(include=[np.number])
        if wide.empty:
            wide = df.copy()
    else:
        lower_cols = [str(col).lower() for col in df.columns]
        sample_col = _find_column(lower_cols, ["sample", "sample_id", "patient_id"])
        feature_col = _find_column(lower_cols, ["gene", "protein", "feature", "id"])
        value_col = _find_column(lower_cols, ["value", "expression", "abundance", "intensity"])
        if sample_col and feature_col and value_col:
            sample_name = df.columns[lower_cols.index(sample_col)]
            feature_name = df.columns[lower_cols.index(feature_col)]
            value_name = df.columns[lower_cols.index(value_col)]
            wide = df.pivot_table(
                index=feature_name,
                columns=sample_name,
                values=value_name,
                aggfunc="mean",
            )
        else:
            wide = df

    if _looks_like_samples_index(wide.index) and not _looks_like_samples_index(wide.columns):
        return wide.T

    if _looks_like_samples_index(wide.columns) and not _looks_like_samples_index(wide.index):
        return wide

    if wide.shape[1] > wide.shape[0]:
        return wide.T

    return wide
