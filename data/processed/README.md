# Processed Tables

This folder contains combined CPTAC tables saved as parquet for fast reloads.

## Files

- `data/processed/all_rna.parquet`
- `data/processed/all_protein.parquet`
- `data/processed/processed_summary.json`

## `all_rna.parquet` layout

Rows are `Patient_ID` values. Columns are a 3-level MultiIndex:

- Level 0: `value_type` (`meta`, `raw`, `z`)
- Level 1: `Name` (gene symbol)
- Level 2: `Database_ID` (Ensembl gene ID)

The `meta` block contains one column:

- `("meta", "cancer_type", "")` = dataset name (e.g., `pdac`, `brca`)

The `raw` block holds original RNA values.  
The `z` block holds z-score values computed **per gene across all samples**, using:

```
z = (x - mean) / std
```

NaNs are ignored for mean/std. If a gene has zero std, its z-scores are NaN.

Transcript-level RNA tables are collapsed to gene-level by averaging transcripts
with the same `(Name, Database_ID)` before combining datasets.

## `all_protein.parquet` layout

Rows are `Patient_ID` values. Columns are a 2-level MultiIndex:

- Level 0: `Name` (gene symbol)
- Level 1: `Database_ID` (Ensembl gene ID)

## Notes

- Column sets are the union across datasets, so many values will be NaN.
- RNA tables can be very wide (hundreds of thousands of columns).
