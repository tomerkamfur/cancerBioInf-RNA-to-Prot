"""
Step 1: Orient to the Dataset
Inspect RNA and protein tables to understand sample matching, identifiers, and missingness.
"""

import pandas as pd
import numpy as np

# Load processed tables
all_rna = pd.read_parquet("data/processed/all_rna.parquet") 
all_protein = pd.read_parquet("data/processed/all_protein.parquet")

print("\n" + "="*60)
print("RNA TABLE OVERVIEW") # Print a header for the RNA table overview
print("="*60)
print(f"Shape: {all_rna.shape} (samples × features)") # The shape should be (number of samples, number of features), where features include both metadata and gene expression values.
print(f"Index name: {all_rna.index.name}") # The index name should be 'sample_id', indicating that rows are indexed by sample identifiers.
print(f"Index (first 5 samples): {all_rna.index[:5].tolist()}") # The first 5 sample IDs should be printed, giving us a sense of the sample naming convention (e.g., "TCGA-XX-XXXX").
print(f"\nColumn levels: {all_rna.columns.nlevels}") # The number of column levels should be 3, indicating a MultiIndex with levels for value type (e.g., 'meta' vs 'gene'), gene name, and possibly additional metadata.
print(f"Column level names: {all_rna.columns.names}") # The column level names should indicate what each level represents, such as ['value_type', 'gene_name', 'extra_info'] or similar. This helps us understand how the data is structured and how to access different types of information (e.g., metadata vs gene expression values).
print(f"Unique level 0 values (value_type): {all_rna.columns.get_level_values(0).unique().tolist()}") # The unique values in the first column level should indicate the types of data present, such as ['meta', 'gene']. This tells us that we have metadata columns (e.g., cancer type) and gene expression columns.
print(f"Unique cancer types: {all_rna[('meta', 'cancer_type', '')].dropna().unique().tolist()}") # The unique cancer types should be printed, giving us insight into the diversity of the dataset and potential batch effects. We might see a list of cancer types like ['BRCA', 'LUAD', 'KIRC', ...], which indicates that the dataset includes samples from multiple cancer types. This is important for understanding the heterogeneity of the data and planning downstream analyses (e.g., stratification by cancer type).
print(f"\nMissingness: {all_rna.isna().sum().sum()} NaN values ({100*all_rna.isna().sum().sum()/(all_rna.shape[0]*all_rna.shape[1]):.1f}%)") # The total number of missing values (NaNs) should be printed, along with the percentage of missing data relative to the total number of entries in the table. This gives us a sense of data quality and completeness. For example, if we see "10000 NaN values (5.0%)", it indicates that 5% of the data is missing, which may require imputation or filtering strategies in downstream analyses.

# for the protein table, we expect a similar structure but without the 'meta' level in the columns, since it should only contain gene expression values. The index should still be 'sample_id', and we should check for missingness and sample matching with the RNA table.
print("\n" + "="*60)
print("PROTEIN TABLE OVERVIEW")
print("="*60)
print(f"Shape: {all_protein.shape} (samples × features)")
print(f"Index name: {all_protein.index.name}")
print(f"Index (first 5 samples): {all_protein.index[:5].tolist()}")
print(f"Column levels: {all_protein.columns.nlevels}")
print(f"Column level names: {all_protein.columns.names}")
print(f"\nMissingness: {all_protein.isna().sum().sum()} NaN values ({100*all_protein.isna().sum().sum()/(all_protein.shape[0]*all_protein.shape[1]):.1f}%)")

# After inspecting both tables, we should have a good understanding of the sample identifiers, the structure of the data, the types of features available, and the extent of missingness. This will inform our next steps in terms of data preprocessing, feature selection, and modeling strategies.
print("\n" + "="*60)
print("SAMPLE MATCHING")
print("="*60)
rna_samples = set(all_rna.index) # The set of sample identifiers from the RNA table should be extracted, which will allow us to compare it with the set of sample identifiers from the protein table. This is crucial for understanding how many samples have both RNA and protein data available, which will impact our modeling approach. We expect that there will be some overlap between the two sets, but there may also be samples that are only present in one table (RNA-only or protein-only). The number of overlapping samples will determine how many samples we can use for integrative analyses that combine both RNA and protein data.
protein_samples = set(all_protein.index) 
overlap = rna_samples & protein_samples # The intersection of the RNA and protein sample sets should be calculated to determine how many samples have both types of data available. This is important for integrative analyses, as we can only use samples that have both RNA and protein measurements when building models that leverage both data types. The number of overlapping samples will give us insight into the potential sample size for our modeling efforts and may also highlight any discrepancies in sample collection or processing between the two datasets.
rna_only = rna_samples - protein_samples # The set of RNA-only samples should be calculated to identify how many samples have RNA data but do not have corresponding protein data. This is important for understanding the limitations of our dataset and for planning how to handle these samples in downstream analyses. For example, if there are a large number of RNA-only samples, we may need to consider whether to include them in certain analyses (e.g., RNA-only models) or whether to focus only on the overlapping samples for integrative modeling. The number of RNA-only samples can also provide insight into potential biases in sample collection or processing that may have led to missing protein data for certain samples.
protein_only = protein_samples - rna_samples # The set of protein-only samples should be calculated to identify how many samples have protein data but do not have corresponding RNA data. This is important for understanding the limitations of our dataset and for planning how to handle these samples in downstream analyses. For example, if there are a large number of protein-only samples, we may need to consider whether to include them in certain analyses (e.g., protein-only models) or whether to focus only on the overlapping samples for integrative modeling. The number of protein-only samples can also provide insight into potential biases in sample collection or processing that may have led to missing RNA data for certain samples.

# Finally, we should print out the results of our sample matching analysis, including the number of RNA samples, protein samples, overlapping samples, RNA-only samples, and protein-only samples. This will give us a clear picture of the sample composition of our dataset and will inform our decisions about how to proceed with data preprocessing and modeling.
print(f"RNA samples: {len(rna_samples)}") # The number of RNA samples should be printed, giving us insight into the size of the RNA dataset. This is important for understanding the potential sample size for RNA-based analyses and for comparing it to the number of protein samples. For example, if we see "RNA samples: 500".
print(f"Protein samples: {len(protein_samples)}") # The number of protein samples should be printed, giving us insight into the size of the protein dataset.
print(f"Overlapping samples: {len(overlap)} ({100*len(overlap)/max(len(rna_samples), len(protein_samples)):.1f}%)")
print(f"RNA-only samples: {len(rna_only)}")
print(f"Protein-only samples: {len(protein_only)}")

if len(overlap) > 0:
    print(f"\nFirst 5 overlapping samples: {list(overlap)[:5]}")

print("\n" + "="*60)
print("IDENTIFIER CONSISTENCY")
print("="*60)
# Check gene identifiers
rna_genes = all_rna.columns[1:]  # Skip 'meta' level
protein_genes = all_protein.columns

rna_gene_names = rna_genes.get_level_values(1).unique() # Level 1 contains gene names (Name); Level 0 is 'value_type'. Extract gene symbols to compare with protein table.
protein_gene_names = protein_genes.get_level_values(0).unique() # Level 0 contains gene names for protein table

# We should check the consistency of gene identifiers between the RNA and protein tables to understand how well the features align across the two datasets. This involves comparing the sets of gene names (or identifiers) present in each table to see how many genes are measured in both datasets, as well as how many are unique to each dataset. This is important for integrative analyses, as we can only directly compare or combine features that are present in both datasets. If there is a large discrepancy in gene identifiers (e.g., different naming conventions, missing genes in one dataset), we may need to perform additional preprocessing steps (e.g., mapping gene names, filtering genes) to ensure that we can effectively integrate the data for modeling.
print(f"Unique RNA genes: {len(rna_gene_names)}")
print(f"Unique protein genes: {len(protein_gene_names)}")
print(f"Overlapping gene names: {len(set(rna_gene_names) & set(protein_gene_names))}")
print(f"\nFirst 5 RNA genes: {rna_gene_names[:5].tolist()}")
print(f"First 5 protein genes: {protein_gene_names[:5].tolist()}")
