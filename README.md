# cancerBioInf-RNA-to-Prot
Project Overview

The Central Dogma of Molecular Biology (DNA → RNA → Protein) is often treated as a proportional and predictable process. In practice, however, protein abundance—the true functional driver of cellular behavior—is influenced by many regulatory layers beyond RNA expression.

While transcriptomic data are abundant, proteomic data remain limited. This raises a critical question in cancer biology:

Can RNA expression reliably predict protein abundance across cancers—and where does this relationship fail?

To address this, we use pan-cancer multi-omics data from CPTAC (Clinical Proteomic Tumor Analysis Consortium). Rather than focusing on a single cancer type, we aggregate all cancer types with available paired RNA-seq and proteomics data to increase statistical power and generalizability.
Cancer type is explicitly included as a model variable to account for tissue- and disease-specific effects.

Hypothesis
Part I: Establishing the Best Predictive Model

We hypothesize that the RNA–protein relationship is context-dependent, varying by gene function and cancer type.

Control (All Genes):
When using all genes and samples, RNA expression will predict protein abundance moderately well
(expected R² > 0.6).
This performance may be driven by stable housekeeping genes and potential overfitting.

Target (Selected Gene Subsets / Cancer Drivers):
When restricting the feature space via gene selection or focusing on cancer driver genes, RNA expression will perform poorly
(expected R² < 0.3).
This would indicate that RNA abundance alone is a weak proxy for protein activity in biologically critical pathways.

Objectives

Build a pan-cancer machine learning pipeline to predict protein abundance from RNA-seq data.

Evaluate whether increasing model complexity improves prediction accuracy.

Quantify how prediction performance varies across:

Gene categories

Cancer types

Identify systematic failures of RNA-based protein prediction.

Establish a scalable and reusable workflow for future protein activity predictors.

Workflow & Methodology
1. Data Acquisition

Download all available cancer datasets using the cptac Python library

Retain only cancer types with:

RNA-seq data

Mass spectrometry–based proteomics data

Combine all cancer types into a single unified dataset

Log cancer type as an explicit categorical variable

2. Multi-Omics Integration

Match patient/sample IDs to generate paired samples

Each sample contains:

RNA expression profile

Protein abundance measurements

Cancer type label

Model Selection (The Experiment)

We compare models of increasing complexity to test whether biological regulation requires non-linear or higher-order representations.

Model 1: Baseline — Linear Regression

Why:
Assumes a direct linear relationship between RNA and protein levels.
If this model performs well across cancers, it suggests broadly conserved and simple regulation.

Model 2: Non-Linear Models — Random Forest & Related Methods

Why:
Non-linear models can capture:

Conditional regulation

Threshold effects

Interaction between genes and cancer types

Improved performance over the baseline—especially for cancer driver genes—would imply complex regulatory mechanisms beyond simple transcriptional control.

Model 3: Unsupervised Feature Learning — PCA

Why:
Dimensionality reduction may uncover latent biological structure and reduce noise.
If PCA-based features outperform linear models, it suggests coordinated regulatory programs rather than independent gene effects.

Model Evaluation

Train/Test Split:

80% training

20% testing

Primary Metric:

R² (coefficient of determination)

Key Analysis:

Accuracy Gap:
Compare prediction performance between:

Housekeeping genes (e.g., GAPDH)

Cancer driver genes (e.g., KRAS)

This gap quantifies how RNA–protein coupling breaks down in oncogenic contexts across cancer types.

Optimization & Final Model

After selecting the best-performing model:

Tune hyperparameters

Incorporate cancer type effects explicitly

Retrain on target proteins

Evaluate performance after dimensionality reduction

Optimize computational efficiency

Our working hypothesis is that even with a reduced feature set, the final model will retain meaningful predictive power while lowering computational cost.

Tech Stack

Python

cptac

pandas

scikit-learn

matplotlib

Expected Impact

This project provides a pan-cancer, data-driven evaluation of the limits of transcriptomics as a proxy for protein abundance. By explicitly modeling cancer type, we aim to disentangle universal regulatory principles from cancer-specific effects and highlight where proteomics adds indispensable biological insight.
