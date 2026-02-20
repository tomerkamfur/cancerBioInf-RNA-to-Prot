"""
Step 3: Feature Selection & Elastic Net Modeling
Build RNA-to-protein prediction models using elastic net regression.
Use cross-validation to prevent overfitting on high-dimensional, small-sample data.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Write console output to a results file as well.
results_dir = "results - linear regression model"
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "step_3_results.txt")
RUN_ONLY_STEP_6 = True

# A simple class to duplicate console output to both stdout and a file.
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

# Redirect stdout to both console and results file.
_results_file = open(results_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _results_file)

all_rna = pd.read_parquet("data/processed/all_rna.parquet")
all_protein = pd.read_parquet("data/processed/all_protein.parquet")

print("\n" + "="*60)
print("STEP 3: ELASTIC NET MODELING FOR RNA→PROTEIN PREDICTION")
print("="*60)

# ============================================================
# 1. PREPARE DATA
# ============================================================
print("\n" + "="*60)
print("1. DATA PREPARATION")
print("="*60)

# Extract z-normalized RNA (skip meta and raw blocks, use only 'z' normalized)
rna_z = all_rna[[col for col in all_rna.columns if col[0] == 'z']].iloc[:, 1:]

# Get only overlapping samples
overlapping_samples = rna_z.index.intersection(all_protein.index)
rna_z = rna_z.loc[overlapping_samples].astype('float32')
all_protein = all_protein.loc[overlapping_samples].astype('float32')

# Drop duplicate indices and realign to guarantee matching rows.
rna_z = rna_z[~rna_z.index.duplicated(keep="first")]
all_protein = all_protein[~all_protein.index.duplicated(keep="first")]
rna_z, all_protein = rna_z.align(all_protein, join="inner", axis=0)

# Note: protein missingness is handled per protein inside the modeling loop.

print(f"\nTraining data:")
print(f"  Samples: {rna_z.shape[0]}")
print(f"  RNA features: {rna_z.shape[1]}")
print(f"  Protein targets: {all_protein.shape[1]}")

top_n_genes = 5000
print(f"  RNA features selected per fold (top {top_n_genes} by variance)")

# ============================================================
# 2. TRAIN/TEST DESIGN: CROSS-VALIDATION
# ============================================================
print("\n" + "="*60)
print("2. CROSS-VALIDATION DESIGN")
print("="*60)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"\nUsing {n_splits}-fold cross-validation:")
print(f"  Why cross-validation?")
print(f"    - Small sample size ({rna_z.shape[0]} samples) → risk of overfitting")
print(f"    - High dimensions ({rna_z.shape[1]} features) → very easy to overfit")
print(f"    - CV gives better estimate of generalization performance")
print(f"  Each fold size is computed per protein after filtering missing values")

# ============================================================
# 3. ELASTIC NET: FEATURE SELECTION & REGULARIZATION
# ============================================================
print("\n" + "="*60)
print("3. ELASTIC NET REGRESSION")
print("="*60)

print(f"""
Elastic Net combines L1 and L2 penalties:
  L1 (Lasso): Forces weak coefficients to exactly 0 → feature selection
  L2 (Ridge): Shrinks large coefficients → prevents overfitting)
  
Why elastic net?
  - Handles high-dimensional data (thousands of RNA genes → predict 1 protein)
  - Automatic feature selection (removes irrelevant genes)
  - Prevents overfitting via regularization
  - ElasticNetCV automatically tunes the regularization parameter (alpha)
  
Parameters:
  - l1_ratio: mix of L1 and L2 (grid searched in inner CV)
  - alphas: regularization strength (CV will find best value)
  - max_iter: iterations for convergence
""")

def run_model_for_proteins(protein_list, label):
    print(f"Modeling {len(protein_list)} proteins {label}: {protein_list.tolist()}")
    fold_results = [] # To store results for each protein and fold

    for protein_idx, protein in enumerate(protein_list):
        print(f"\n--- Protein {protein_idx+1}/{len(protein_list)}: {protein} ---")
        
        y = all_protein[protein]
        
        # Skip if too many NaNs in this protein
        valid_mask = y.notna() # Only use samples with valid protein measurements
        if valid_mask.sum() < 10: # Arbitrary threshold for minimum samples to model
            print(f"  Skipped: too few valid samples ({valid_mask.sum()})")
            continue
        
        X = rna_z.loc[valid_mask].values
        y = y.loc[valid_mask].values
        
        print(f"  Valid samples for this protein: {len(y)}")
        
        cv_r2_scores = [] # To store R² scores for each fold, r2 = 1 - (residual sum of squares / total sum of squares)
        cv_mse_scores = [] # To store MSE scores for each fold, MSE = mean squared error
        selected_features_list = [] # To store number of selected features for each fold
        
        fold_idx = 1
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Impute and select features using only the training fold to avoid leakage
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            # Feature selection: select top N genes by variance in the training fold
            variances = np.var(X_train_imp, axis=0)
            top_n = min(top_n_genes, X_train_imp.shape[1]) # Ensure we don't select more features than available
            top_idx = np.argsort(variances)[-top_n:] # Gets the indices of the top-variance genes (largest variances).
            X_train_sel = X_train_imp[:, top_idx] #Keeps only the selected genes in the training data.
            X_test_sel = X_test_imp[:, top_idx] #Keeps the same genes in the test data to ensure consistency.

            # Standardize features (important for regularization)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            model = ElasticNetCV(
                l1_ratio=np.linspace(0.1, 0.9, 9),  # Wider grid over L1/L2 mix
                alphas=np.logspace(-4, 1, 30),  # Slightly wider alpha grid
                cv=5,                  # 5-fold CV within training to tune alpha
                max_iter=1000,
                n_jobs=-1, # Use all cores for parallel CV
                random_state=42,
                verbose=0 # Suppress convergence warnings for demonstration
            )
            model.fit(X_train_scaled, y_train) # Fit on training data; imputation/selection from training fold only
            
            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Count selected features (non-zero coefficients)
            n_selected = np.sum(model.coef_ != 0)
            n_features_after_filter = X_train_sel.shape[1]
            
            cv_r2_scores.append(r2)
            cv_mse_scores.append(mse)
            selected_features_list.append(n_selected)
            
            if fold_idx == 1:
                print(f"  Fold 1 performance:")
                print(f"    R² = {r2:.4f} (fraction of variance explained)")
                print(f"    RMSE = {np.sqrt(mse):.4f}")
                print(f"    MAE = {mae:.4f}")
                print(f"    Selected features: {n_selected} / {n_features_after_filter} (~{100*n_selected/n_features_after_filter:.1f}%)")
                print(f"    Best alpha: {model.alpha_:.4f}")
                print(f"    Best l1_ratio: {model.l1_ratio_}")
            
            fold_idx += 1
        
        # Average across folds
        avg_r2 = np.mean(cv_r2_scores)
        avg_n_selected = np.mean(selected_features_list)
        
        print(f"\n  Cross-validation average (5 folds):")
        print(f"    Mean R²: {avg_r2:.4f}")
        print(f"    Mean features selected: {avg_n_selected:.0f}")
        
        fold_results.append({
            'protein': protein,
            'avg_r2': avg_r2,
            'avg_features': avg_n_selected,
            'n_samples': len(y)
        })
    return fold_results

fold_results = []
if not RUN_ONLY_STEP_6:
    # Select a few proteins to model (for demonstration)
    # In practice, we'd model all proteins
    target_proteins = all_protein.columns[:3]  # First 3 proteins
    fold_results = run_model_for_proteins(target_proteins, "(as examples)")

# ============================================================
# 4. INTERPRETATION
# ============================================================
if not RUN_ONLY_STEP_6:
    print("\n" + "="*60)
    print("4. MODEL INTERPRETATION")
    print("="*60)

    if fold_results:
        results_df = pd.DataFrame(fold_results)
        print(f"\nResults summary:")
        print(results_df.to_string(index=False))
        
        avg_r2_all = results_df['avg_r2'].mean()
        print(f"\n  Average R² across proteins: {avg_r2_all:.4f}")
        
        if avg_r2_all > 0.5:
            print(f"  Good: Model explains >50% of protein variance")
        elif avg_r2_all > 0.3:
            print(f"  Moderate: Model captures some signal (~{avg_r2_all*100:.0f}%)")
        else:
            print(f"  Weak: Limited predictive power")
        
        print(f"\n  Average features selected: {results_df['avg_features'].mean():.0f}")
        print(f"  This means elastic net reduces from {rna_z.shape[1]} features")
        print(f"  to ~{results_df['avg_features'].mean():.0f} predictive genes per protein")


# ============================================================
# 5. SANITY CHECKS (BASELINE + PERMUTATION)
# ============================================================
if not RUN_ONLY_STEP_6:
    print("\n" + "="*60)
    print("5. SANITY CHECKS")
    print("="*60)

    print("\nBaseline check: mean predictor on the same test folds")
    for protein_idx, protein in enumerate(target_proteins): # Loop through the target proteins
        y = all_protein[protein]
        valid_mask = y.notna()
        if valid_mask.sum() < 10: # Skip if too few valid samples to model
            continue
        y_valid = y.loc[valid_mask].values
        X_valid = rna_z.loc[valid_mask].values
        baseline_r2 = []
        for train_idx, test_idx in kfold.split(X_valid):
            y_train = y_valid[train_idx]
            y_test = y_valid[test_idx]
            y_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
            baseline_r2.append(r2_score(y_test, y_pred))
        print(f"  {protein}: mean-baseline R² = {np.mean(baseline_r2):.4f}") # R² can be negative if the model is worse than just predicting the mean, so we expect this to be around 0 or negative.

    print("\nPermutation check: shuffle labels within each fold (should drop to ~0)")
    for protein_idx, protein in enumerate(target_proteins):
        y = all_protein[protein]
        valid_mask = y.notna()
        if valid_mask.sum() < 10:
            continue
        X = rna_z.loc[valid_mask].values
        y = y.loc[valid_mask].values
        perm_r2_scores = []
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx].copy(), y[test_idx]

            # Shuffle labels in the training fold only
            rng = np.random.default_rng(42)
            rng.shuffle(y_train)

            # Same preprocessing as the main model
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            variances = np.var(X_train_imp, axis=0)
            top_n = min(top_n_genes, X_train_imp.shape[1])
            top_idx = np.argsort(variances)[-top_n:]

            X_train_sel = X_train_imp[:, top_idx]
            X_test_sel = X_test_imp[:, top_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            perm_model = ElasticNetCV(
                l1_ratio=np.linspace(0.1, 0.9, 9),
                alphas=np.logspace(-4, 1, 30),
                cv=5,
                max_iter=1000,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            perm_model.fit(X_train_scaled, y_train)
            y_pred = perm_model.predict(X_test_scaled)
            perm_r2_scores.append(r2_score(y_test, y_pred))

        print(f"  {protein}: permuted-label R² = {np.mean(perm_r2_scores):.4f}")

# ============================================================
# 6. OPTIONAL SINGLE-PROTEIN RUN
# ============================================================
print("\n" + "="*60)
print("6. OPTIONAL SINGLE-PROTEIN RUN")
print("="*60)

user_protein = input("\nEnter protein gene symbol to run (blank to skip): ").strip()
if user_protein:
    matching = [col for col in all_protein.columns if col[0] == user_protein]
    if matching:
        _ = run_model_for_proteins(pd.Index(matching), f"(user-selected: {user_protein})")
    else:
        print(f"  Protein '{user_protein}' not found in targets.")
