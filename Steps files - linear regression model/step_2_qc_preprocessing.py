"""
Step 2: Quality Control and Preprocessing
Filter missing data, check normalization, detect batch effects, and handle outliers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

all_rna = pd.read_parquet("data/processed/all_rna.parquet")
all_protein = pd.read_parquet("data/processed/all_protein.parquet")
filtered_rna_path = "data/processed/all_rna_step2_filtered.parquet"
filtered_protein_path = "data/processed/all_protein_step2_filtered.parquet"

print("\n" + "="*60)
print("STEP 2: QUALITY CONTROL & PREPROCESSING")
print("="*60)

# ============================================================
# 1. MISSING VALUE ANALYSIS & FILTERING
# ============================================================
print("\n" + "="*60)
print("1. MISSING VALUE FILTERING")
print("="*60)

print(f"\nBefore filtering:")
print(f"  RNA shape: {all_rna.shape} samples (patients) × features (RNA genes/transcripts)") 
print(f"  Protein shape: {all_protein.shape} samples (patients) × features (protein measurements)") 
print(f"  Protein missingness: {100*all_protein.isna().sum().sum()/(all_protein.shape[0]*all_protein.shape[1]):.1f}%")
print(f"  This high missingness is because different datasets measured different proteins")

# Filter proteins: keep only those measured in at least 50% of samples
# Why? Low-coverage proteins have too much missing data to learn from.
protein_coverage = 1 - all_protein.isna().mean()  # Fraction non-missing per protein
coverage_threshold = 0.5
well_covered_proteins = protein_coverage[protein_coverage >= coverage_threshold].index

print(f"\nProtein coverage filter (>={int(coverage_threshold*100)}%):")
print(f"  Proteins before filter: {all_protein.shape[1]}")
print(f"  Proteins after filter: {len(well_covered_proteins)}")
print(f"  Proteins removed: {all_protein.shape[1] - len(well_covered_proteins)}")

all_protein_filtered = all_protein[well_covered_proteins]

# Filter samples: keep only those with measurements for at least 50% of remaining proteins
# Why? Samples with massive missing data may be poor quality or have technical issues.
sample_coverage = 1 - all_protein_filtered.isna().mean(axis=1)
well_covered_samples = sample_coverage[sample_coverage >= coverage_threshold].index

print(f"\nSample coverage filter (>={int(coverage_threshold*100)}%):")
print(f"  Samples before filter: {all_protein_filtered.shape[0]}")
print(f"  Samples after filter: {len(well_covered_samples)}")
print(f"  Samples removed: {all_protein_filtered.shape[0] - len(well_covered_samples)}")

all_protein_filtered = all_protein_filtered.loc[well_covered_samples]

# Filter RNA to matching samples
all_rna_filtered = all_rna.loc[all_rna.index.isin(well_covered_samples)]

print(f"\nAfter filtering:")
print(f"  RNA shape: {all_rna_filtered.shape}")
print(f"  Protein shape: {all_protein_filtered.shape}")
print(f"  Protein missingness: {100*all_protein_filtered.isna().sum().sum()/(all_protein_filtered.shape[0]*all_protein_filtered.shape[1]):.1f}%")

# ============================================================
# 2. BATCH EFFECT DETECTION (by cancer type)
# ============================================================
print("\n" + "="*60)
print("2. BATCH EFFECT DETECTION (Cancer Type Distribution)")
print("="*60)

# Extract cancer type from RNA meta column
cancer_type = all_rna_filtered[('meta', 'cancer_type', '')].dropna()
cancer_counts = cancer_type.value_counts().sort_values(ascending=False)

print(f"\nSamples per cancer type (after filtering):")
for cancer, count in cancer_counts.items():
    pct = 100*count/len(cancer_type)
    print(f"  {cancer}: {count} ({pct:.1f}%)")

# ============================================================
# 3. NORMALIZATION STATE CHECK
# ============================================================
print("\n" + "="*60)
print("3. NORMALIZATION STATE")
print("="*60)

# Check RNA: we have 'raw' and 'z' normalized versions
rna_raw = all_rna_filtered[[col for col in all_rna_filtered.columns if col[0] == 'raw']].iloc[:, 1:]
rna_z = all_rna_filtered[[col for col in all_rna_filtered.columns if col[0] == 'z']].iloc[:, 1:]

print(f"\nRNA normalization:")
print(f"  'raw' values:")
print(f"    Mean: {rna_raw.mean().mean():.2f}")
print(f"    Std: {rna_raw.std().mean():.2f}")
print(f"    Range: [{rna_raw.min().min():.2f}, {rna_raw.max().max():.2f}]")
print(f"  'z' values (per-gene standardized):")
print(f"    Mean: {rna_z.mean().mean():.4f} (should be ~0)")
print(f"    Std: {rna_z.std().mean():.4f} (should be ~1)")

# Protein normalization check
print(f"\nProtein values:")
print(f"  Mean: {all_protein_filtered.mean().mean():.2f}")
print(f"  Std: {all_protein_filtered.std().mean():.2f}")
print(f"  Range: [{all_protein_filtered.min().min():.2f}, {all_protein_filtered.max().max():.2f}]")

# ============================================================
# 4. OUTLIER DETECTION
# ============================================================
print("\n" + "="*60)
print("4. OUTLIER DETECTION")
print("="*60)

# Use Interquartile Range (IQR) method for outliers
def find_outlier_samples(df, iqr_multiplier=1.5):
    """Find samples with extreme values using IQR method."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    outliers = ((df < lower_bound) | (df > upper_bound)).any(axis=1)
    return outliers.sum()

rna_outliers = find_outlier_samples(rna_raw)
protein_outliers = find_outlier_samples(all_protein_filtered)

print(f"\nOutlier samples (IQR method, 1.5x multiplier):")
print(f"  RNA outlier samples: {rna_outliers} ({100*rna_outliers/len(rna_raw):.1f}%)")
print(f"  Protein outlier samples: {protein_outliers} ({100*protein_outliers/len(all_protein_filtered):.1f}%)")

if rna_outliers > 0 or protein_outliers > 0:
    print(f"  → Consider whether to remove or keep outliers for modeling")
    print(f"     (Outliers may represent real biological variation or technical issues)")

# ============================================================
# 5. CORRELATION CHECK (RNA vs Protein)
# ============================================================
print("\n" + "="*60)
print("\n" + "="*60)
print("5. RNA-PROTEIN CORRELATION (Sanity Check)")
print("="*60)

# Find genes that appear in both tables
# RNA: Level 0='raw'/'z', Level 1=gene_name, Level 2=ensembl_id
# Protein: Level 0=gene_name, Level 1=ensembl_id
rna_gene_names = rna_z.columns.get_level_values(1).unique()  # Level 1 has gene names!
protein_gene_names = all_protein_filtered.columns.get_level_values(0).unique()
shared_genes = list(set(rna_gene_names) & set(protein_gene_names))

print(f"\nShared genes available for correlation: {len(shared_genes)}")

if len(shared_genes) > 0:
    # Compute correlation for first 5 shared genes (as example)
    sample_correlations = []
    for gene in shared_genes[:5]:
        # Get RNA (use z-normalized version for fairness)
        # For RNA: look for columns where level 1 (gene name) matches
        rna_col = [col for col in rna_z.columns if col[1] == gene]
        protein_col = [col for col in all_protein_filtered.columns if col[0] == gene]
        
        if len(rna_col) > 0 and len(protein_col) > 0:
            rna_vals = rna_z[rna_col[0]].dropna()
            prot_vals = all_protein_filtered[protein_col[0]].dropna()
            
            # Find common indices
            common_idx = rna_vals.index.intersection(prot_vals.index)
            if len(common_idx) > 5:
                corr = rna_vals[common_idx].corr(prot_vals[common_idx])
                sample_correlations.append((gene, corr))
    
    if sample_correlations:
        print(f"\nSample RNA-protein correlations (first 5 shared genes):")
        for gene, corr in sample_correlations:
            print(f"  {gene}: {corr:.3f}")
        avg_corr = np.mean([c for _, c in sample_correlations])
print(f"  Average: {avg_corr:.3f}")
print(f"  → Moderate to weak correlation is typical (different biological processes)")

# ============================================================
# 6. EXPORT FILTERED TABLES FOR DOWNSTREAM MODELING
# ============================================================
print("\n" + "="*60)
print("6. EXPORT FILTERED TABLES")
print("="*60)

os.makedirs("data/processed", exist_ok=True)
all_rna_filtered.to_parquet(filtered_rna_path)
all_protein_filtered.to_parquet(filtered_protein_path)

print(f"\nSaved filtered RNA table: {filtered_rna_path}")
print(f"  Shape: {all_rna_filtered.shape}")
print(f"Saved filtered protein table: {filtered_protein_path}")
print(f"  Shape: {all_protein_filtered.shape}")
