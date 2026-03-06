"""
Step 3: Feature Selection & Elastic Net Modeling
Build RNA-to-protein prediction models using elastic net regression.
Use cross-validation to prevent overfitting on high-dimensional, small-sample data.
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Write console output to a results file as well.
RUN_ONLY_STEP_6 = False # Set to True to run only the step-6 protein modeling flow.
USE_FILTERED_STEP2 = True # Read filtered outputs from step 2 when available.
RUN_STEP7_ALL_RNA = True # Set to False to skip the new no-feature-selection comparison.
Rna_PATH_STEP2 = "data/processed/all_rna_step2_filtered.parquet"
Protein_PATH_STEP2 = "data/processed/all_protein_step2_filtered.parquet"
Rna_PATH_ORIG = "data/processed/all_rna.parquet"
Protein_PATH_ORIG = "data/processed/all_protein.parquet"
STEP3_RESULTS_PATH = os.path.join("results - linear regression model", "step_3_results.txt")
CHOSEN_PROTEINS_RESULTS_PATH = os.path.join("results - linear regression model", "step_3_results_chosen_proteins.txt")
results_dir = "results - linear regression model"
os.makedirs(results_dir, exist_ok=True)

def _safe_token_for_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)

results_path = CHOSEN_PROTEINS_RESULTS_PATH if RUN_ONLY_STEP_6 else STEP3_RESULTS_PATH
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
_original_stdout = sys.stdout
_results_file = open(results_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _results_file)

if USE_FILTERED_STEP2 and os.path.exists(Rna_PATH_STEP2) and os.path.exists(Protein_PATH_STEP2):
    all_rna = pd.read_parquet(Rna_PATH_STEP2)
    all_protein = pd.read_parquet(Protein_PATH_STEP2)
    print(f"\nUsing Step 2 filtered inputs:\n  RNA: {Rna_PATH_STEP2}\n  Protein: {Protein_PATH_STEP2}")
else:
    all_rna = pd.read_parquet(Rna_PATH_ORIG)
    all_protein = pd.read_parquet(Protein_PATH_ORIG)
    if USE_FILTERED_STEP2:
        print("\nWarning: filtered step 2 files not found. Falling back to original processed files.")
        print(f"  Expected RNA: {Rna_PATH_STEP2}")
        print(f"  Expected Protein: {Protein_PATH_STEP2}")

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

def run_model_for_proteins(protein_list, label, use_top_variance_filter=True):
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
        
        cv_r2_scores = []
        cv_mse_scores = []
        cv_mae_scores = []
        selected_features_list = []
        input_features_list = []
        fold_alpha_values = []
        fold_l1_ratio_values = []
        
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Impute and (optionally) select features using only the training fold to avoid leakage
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            if use_top_variance_filter:
                # Feature selection: select top N genes by variance in the training fold
                variances = np.var(X_train_imp, axis=0)
                top_n = min(top_n_genes, X_train_imp.shape[1]) # Ensure we don't select more features than available
                top_idx = np.argsort(variances)[-top_n:] # Gets the indices of the top-variance genes (largest variances).
                X_train_sel = X_train_imp[:, top_idx] # Keeps only the selected genes in the training data.
                X_test_sel = X_test_imp[:, top_idx] # Keeps the same genes in the test data to ensure consistency.
            else:
                # No feature selection: use all available RNA features.
                X_train_sel = X_train_imp
                X_test_sel = X_test_imp

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
            n_features_before_filter = X_train_sel.shape[1]
            n_features_total = X_train.shape[1]
            
            cv_r2_scores.append(r2)
            cv_mse_scores.append(mse)
            cv_mae_scores.append(mae)
            selected_features_list.append(n_selected)
            input_features_list.append(n_features_before_filter)
            fold_alpha_values.append(model.alpha_)
            fold_l1_ratio_values.append(model.l1_ratio_)
            
            current_fold = len(cv_r2_scores)
            print(f"  Fold {current_fold} performance:")
            print(f"    R² = {r2:.4f} (fraction of variance explained)")
            print(f"    RMSE = {np.sqrt(mse):.4f}")
            print(f"    MAE = {mae:.4f}")
            if use_top_variance_filter:
                print(f"    Manual feature selection kept: {n_features_before_filter} of {n_features_total} RNAs")
            else:
                print(f"    Manual feature selection: disabled (using all {n_features_before_filter} RNAs)")
            if n_features_before_filter:
                nz_pct = 100 * n_selected / n_features_before_filter
            else:
                nz_pct = 0.0
            print(f"    Non-zero coefficients: {n_selected} / {n_features_before_filter} (~{nz_pct:.1f}%)")
            print(f"    Best alpha: {model.alpha_:.4f}")
            print(f"    Best l1_ratio: {model.l1_ratio_}")
        
        # Average across folds
        cv_r2_scores = np.array(cv_r2_scores, dtype=float)
        cv_mse_scores = np.array(cv_mse_scores, dtype=float)
        cv_mae_scores = np.array(cv_mae_scores, dtype=float)
        selected_features_list = np.array(selected_features_list, dtype=float)
        input_features_list = np.array(input_features_list, dtype=float)
        fold_alpha_values = np.array(fold_alpha_values, dtype=float)
        fold_l1_ratio_values = np.array(fold_l1_ratio_values, dtype=float)

        if cv_r2_scores.size == 0:
            print(f"  Skipping {protein}: no completed CV folds.")
            continue

        avg_r2 = cv_r2_scores.mean()
        std_r2 = cv_r2_scores.std(ddof=0)
        min_r2 = cv_r2_scores.min()
        max_r2 = cv_r2_scores.max()
        avg_n_selected = selected_features_list.mean()
        std_n_selected = selected_features_list.std(ddof=0)
        avg_n_input_features = input_features_list.mean()
        std_n_input_features = input_features_list.std(ddof=0)
        
        print(f"\n  Cross-validation summary (5 folds):")
        print(f"    Mean R²: {avg_r2:.4f}")
        print(f"    Std R²: {std_r2:.4f}")
        print(f"    Min R²: {min_r2:.4f}")
        print(f"    Max R²: {max_r2:.4f}")
        print(f"    Mean input RNAs per fold: {avg_n_input_features:.0f}")
        print(f"    Std input RNAs per fold: {std_n_input_features:.1f}")
        print(f"    Mean non-zero coefficients: {avg_n_selected:.0f}")
        print(f"    Std non-zero coefficients: {std_n_selected:.1f}")
        
        fold_results.append({
            'protein': protein,
            'avg_r2': avg_r2,
            'std_r2': std_r2,
            'min_r2': min_r2,
            'max_r2': max_r2,
            'avg_input_features': avg_n_input_features,
            'std_input_features': std_n_input_features,
            'avg_features': avg_n_selected,
            'std_features': std_n_selected,
            'n_samples': len(y),
            'fold_r2': cv_r2_scores.tolist(),
            'fold_rmse': np.sqrt(cv_mse_scores).tolist(),
            'fold_mae': cv_mae_scores.tolist(),
            'fold_selected_features': selected_features_list.tolist(),
            'fold_alpha': fold_alpha_values.tolist(),
            'fold_l1_ratio': fold_l1_ratio_values.tolist(),
            'used_variance_filter': bool(use_top_variance_filter),
        })
    return fold_results

# Choose a target explicitly by menu (even if only one match) so user controls selection.
def pick_protein_targets(user_symbol):
    symbol = user_symbol.strip().upper()
    if not symbol:
        return []

    matches = [col for col in all_protein.columns if str(col[0]).upper() == symbol]
    if not matches:
        return None

    print(f"  Found {len(matches)} targets for symbol '{user_symbol}':")
    for i, col in enumerate(matches, start=1):
        print(f"    {i}. {col}")

    while True:
        choice = input(f"Enter 1-{len(matches)}: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(matches):
                return [matches[idx - 1]]
        print("  Invalid choice. Try again.")

fold_results = []
if not RUN_ONLY_STEP_6:
    # Select a few proteins to model (for demonstration)
    # In practice, we'd model all proteins
    target_proteins = all_protein.columns[:3]  # First 3 proteins
    fold_results = run_model_for_proteins(target_proteins, "(as examples)")
    fold_results_no_filter = []
    if RUN_STEP7_ALL_RNA:
        print("\n" + "="*60)
        print("7. MODELING WITHOUT VARIANCE-BASED FEATURE SELECTION")
        print("="*60)
        fold_results_no_filter = run_model_for_proteins(
            target_proteins,
            "(using all RNA features, no pre-filter)",
            use_top_variance_filter=False
        )

        if fold_results and fold_results_no_filter:
            print("\nFeature-selection efficacy comparison (run 3-protein benchmark):")
            summary_fs = pd.DataFrame(fold_results)
            summary_no_filter = pd.DataFrame(fold_results_no_filter)
            fs_by_protein = dict(zip(summary_fs["protein"], summary_fs["avg_r2"]))
            no_filter_by_protein = dict(zip(summary_no_filter["protein"], summary_no_filter["avg_r2"]))
            target_set = list(target_proteins)
            common_proteins = [p for p in target_set if p in fs_by_protein and p in no_filter_by_protein]
            missing_fs = [p for p in target_set if p not in fs_by_protein]
            missing_nofilter = [p for p in target_set if p not in no_filter_by_protein]

            if missing_fs:
                print("  Warning: proteins missing from filtered run:")
                for p in missing_fs:
                    print(f"    - {p}")
            if missing_nofilter:
                print("  Warning: proteins missing from no-filter run:")
                for p in missing_nofilter:
                    print(f"    - {p}")

            if not common_proteins:
                print("  No common proteins available for comparison.")
            else:
                if len(common_proteins) != len(target_set):
                    print(f"  Comparing only common proteins ({len(common_proteins)}/{len(target_set)}).")

                print("  Protein | mean R² (top variance) | mean R² (all RNA) | delta")
                print("  " + "-"*68)
                common_fs_r2 = []
                common_nf_r2 = []
                common_input_fs = []
                common_input_nf = []
                common_nonzero_fs = []
                common_nonzero_nf = []
                for protein in common_proteins:
                    fs_r2 = fs_by_protein[protein]
                    nofilter_r2 = no_filter_by_protein[protein]
                    delta = nofilter_r2 - fs_r2
                    common_fs_r2.append(fs_r2)
                    common_nf_r2.append(nofilter_r2)
                    fs_input = summary_fs.loc[summary_fs["protein"] == protein, "avg_input_features"]
                    nf_input = summary_no_filter.loc[summary_no_filter["protein"] == protein, "avg_input_features"]
                    fs_nz = summary_fs.loc[summary_fs["protein"] == protein, "avg_features"]
                    nf_nz = summary_no_filter.loc[summary_no_filter["protein"] == protein, "avg_features"]

                    common_input_fs.append(float(fs_input.iloc[0]))
                    common_input_nf.append(float(nf_input.iloc[0]))
                    common_nonzero_fs.append(float(fs_nz.iloc[0]))
                    common_nonzero_nf.append(float(nf_nz.iloc[0]))

                    print(f"  {protein}: {fs_r2: .4f} | {nofilter_r2: .4f} | {delta: .4f}")

                print(f"\n  Average R² (with variance filter): {np.mean(common_fs_r2):.4f}")
                print(f"  Average R² (all RNA): {np.mean(common_nf_r2):.4f}")
                print(f"  Delta in average R² (all RNA - filtered): {np.mean(common_nf_r2) - np.mean(common_fs_r2):.4f}")
                print(f"  Average RNAs used per fold (manual variance filter): {np.mean(common_input_fs):.0f}")
                print(f"  Average RNAs used per fold (no manual filter): {np.mean(common_input_nf):.0f}")
                print(f"  Average non-zero coefficients (manual variance filter): {np.mean(common_nonzero_fs):.0f}")
                print(f"  Average non-zero coefficients (no manual filter): {np.mean(common_nonzero_nf):.0f}")

        print("\nNote: step 7 only changes the RNA feature preprocessing step.")
        print("      Model tuning, splits, and metrics remain identical for fair comparison.")

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
        std_r2_among_proteins = results_df['avg_r2'].std(ddof=0)
        print(f"\n  Average mean R² across proteins: {avg_r2_all:.4f}")
        print(f"  Std of mean R² across proteins: {std_r2_among_proteins:.4f}")
        
        if avg_r2_all > 0.5:
            print(f"  Good: Model explains >50% of protein variance")
        elif avg_r2_all > 0.3:
            print(f"  Moderate: Model captures some signal (~{avg_r2_all*100:.0f}%)")
        else:
            print(f"  Weak: Limited predictive power")
        
        print(f"\n  Average non-zero coefficients: {results_df['avg_features'].mean():.0f}")
        print(f"  Model is trained on average from ~{results_df['avg_input_features'].mean():.0f} input features")
        print(f"  and keeps ~{results_df['avg_features'].mean():.0f} non-zero coefficients per protein")


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

if RUN_ONLY_STEP_6:
    selected_targets = []
    selected_inputs = []
    while True:
        run_only_protein = input("\nRUN_ONLY_STEP_6=True. Enter protein gene symbol to run (blank to finish): ").strip()
        if not run_only_protein:
            break
        selected_targets_batch = pick_protein_targets(run_only_protein)
        if selected_targets_batch is None:
            print(f"  Protein '{run_only_protein}' not found in targets.")
            continue
        selected_targets.extend(selected_targets_batch)
        selected_inputs.append(run_only_protein)
        for target in selected_targets_batch:
            print(f"  Added: {target}")

    if not selected_targets:
        print("No protein entered. Exiting without running step 6.")
        raise SystemExit(0)

    if len(selected_inputs) == 1:
        print(f"\nUsing run-only input: {selected_inputs[0]}")
    else:
        print(f"\nRUN_ONLY_STEP_6 selected {len(selected_inputs)} proteins.")
else:
    selected_targets = []
    while True:
        user_protein = input("\nEnter protein gene symbol to run (blank to finish): ").strip()
        if not user_protein:
            break
        selected_targets_batch = pick_protein_targets(user_protein)
        if selected_targets_batch is None:
            print(f"  Protein '{user_protein}' not found in targets.")
            continue
        selected_targets.extend(selected_targets_batch)
        for target in selected_targets_batch:
            print(f"  Added: {target}")

if selected_targets:
    print(f"\nRunning user-selected proteins: {selected_targets}")
    _ = run_model_for_proteins(pd.Index(selected_targets), "(user-selected proteins)")
else:
    print("\nNo proteins selected for step 6 run.")

print(f"\nResults were written to: {results_path}")
sys.stdout = _original_stdout
_results_file.close()
