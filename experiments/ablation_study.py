#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ablation_study.py

1) Perform ablation study by removing one feature group at a time.
2) Train a single multi-output RandomForest model (not a cascade).
3) Save the model.
4) Evaluate on the test set (confusion matrix, evaluation report).
5) Include a baseline with all features.
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from evaluate_ec import evaluate_ec_predictions
import gc
import psutil

import sys
sys.modules["numpy._core.numeric"] = np.core.numeric

# ---------------------------
# Memory utilization monitoring
# ---------------------------
def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Current memory usage: {mem_info.rss / 1024 / 1024 / 1024:.2f} GB")

# ---------------------------
# 1. Helper: group features by name
# ---------------------------
def feature_group_map(col_name: str) -> str:
    if col_name.startswith("ACC_") or col_name.startswith("DC_"):
        return "Sequenctial"
    elif col_name in [
        "molecular_weight", "isoelectric_point", "gravy", 
        "aromaticity", "instability_index", "aliphatic_index", "boman_index"
    ]:
        return "Physiochemical"
    elif col_name in ["E-value", "score", "coverage", "num_domain"]:
        return "HMM"
    elif col_name.startswith("pssm_"):
        return "PSSM"
    elif col_name.startswith("embedded_"):
        return "Sequence Embedding"
    else:
        return "QSAR"

# ---------------------------
# 2. Load data (train/valid/test) with original columns
# ---------------------------
def load_data_with_full_columns(chunk_size=None):
    """
    Load train, validation, and test data from pickle files 
    with optional chunking to save memory.
    """
    print("Loading training data...")
    train_df = pd.read_pickle("../dataset/all_features/train.pkl")
    print("Loading validation data...")
    valid_df = pd.read_pickle("../dataset/all_features/valid.pkl")
    print("Loading test data...")
    test_df = pd.read_pickle("../dataset/all_features/test.pkl")
    
    print_memory_usage()
    
    # Convert to more memory-efficient dtypes where possible
    for df in [train_df, valid_df, test_df]:
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
    
    print("Data loaded and optimized.")
    print_memory_usage()
    
    return train_df, valid_df, test_df

def get_feature_and_label(df):
    """Split the feature columns and EC label columns without creating unnecessary copies."""
    ec_cols = [col for col in df.columns if col.startswith('ec_')]
    X_df = df.drop(columns=ec_cols)
    y_df = df[ec_cols]
    return X_df, y_df

# ---------------------------
# 3. Preprocessing
# ---------------------------
def prepare_train_val(X_train, X_valid, y_train, y_valid):
    """
    Concatenate train and validation sets and create a PredefinedSplit.
    Memory-optimized version.
    """
    # Use np.vstack to avoid intermediate copies
    X_trainval = np.vstack([X_train, X_valid])
    y_trainval = np.vstack([y_train, y_valid])
    
    # Using np.ones/zeros with the right dtype helps save memory
    split_index = np.hstack([
        np.full(X_train.shape[0], -1, dtype=np.int8),  # -1: training samples
        np.full(X_valid.shape[0], 0, dtype=np.int8)    # 0: validation samples
    ])
    
    pds = PredefinedSplit(test_fold=split_index)
    
    # Free intermediate memory
    del X_train, X_valid, y_train, y_valid
    gc.collect()
    
    return X_trainval, y_trainval, pds

# ---------------------------
# 4. Train and evaluate a model
# ---------------------------
def train_and_evaluate_model(X_trainval, y_trainval, X_test_scaled, y_test_np, y_test_df, model_name, models_folder, confusion_matrix_folder, reports_folder,batch_size=500):
    """Train and evaluate a RandomForest model with the given data."""
    print(f"Training RandomForest model: {model_name}...")
    best_model = RandomForestClassifier(
        random_state=42, 
        n_estimators=200,
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=6,
        verbose=1,
        oob_score=False, 
    )
    
    best_model.fit(X_trainval, y_trainval)
    
    print("Model training completed.")
    print_memory_usage()

    # Not saving model to save disk memory
    # model_filename = f"{model_name}_RF.pkl"
    # model_path = os.path.join(models_folder, model_filename)
    
    # print(f"Saving model to {model_path}...")
    # with open(model_path, "wb") as f:
    #     pickle.dump(best_model, f)
    # print(f"Model saved to {model_path}")

    # Evaluation on Test Set
    print(f"Evaluating model: {model_name}...")

    # Predict in batches if test set is large
    if len(X_test_scaled) > batch_size:
        test_pred_parts = []
        for i in range(0, len(X_test_scaled), batch_size):
            end_idx = min(i + batch_size, len(X_test_scaled))
            batch_pred = best_model.predict(X_test_scaled[i:end_idx])
            test_pred_parts.append(batch_pred)
            del batch_pred
            gc.collect()
        test_pred = np.vstack(test_pred_parts)
        del test_pred_parts
    else:
        # Single prediction if test set is small enough
        test_pred = best_model.predict(X_test_scaled)
    
    # Evaluation
    print("Generating evaluation report...")
    eval_report = evaluate_ec_predictions(
        test_pred,
        y_test_np,
        method_name=model_name
    )
    
    # Save individual report
    report_filename = f"eval_report_{model_name}.csv"
    report_path = os.path.join(reports_folder, report_filename)
    eval_report.to_csv(report_path, index=False)
    print(f"Saved evaluation report to {report_path}")

    # Confusion Matrix generation
    print("Generating confusion matrices...")
    model_cm_folder = os.path.join(confusion_matrix_folder, model_name)
    os.makedirs(model_cm_folder, exist_ok=True)

    num_outputs = y_test_df.shape[1]
    for i in range(num_outputs):
        y_true = y_test_np[:, i]
        y_pred = test_pred[:, i]
        mask = (y_true != -1)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        if len(y_true_filtered) == 0:
            print(f"No valid samples for EC_{i}, skipping confusion matrix.")
            continue

        unique_labels = np.unique(np.concatenate([y_true_filtered, y_pred_filtered]))
        unique_labels = np.sort(unique_labels)

        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=unique_labels)

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(
            cm.astype(np.float32), 
            row_sums,
            out=np.zeros_like(cm, dtype=np.float32),
            where=(row_sums != 0)
        )

        non_empty_rows = cm_normalized.sum(axis=1) != 0
        non_empty_cols = cm_normalized.sum(axis=0) != 0
        cm_normalized = cm_normalized[non_empty_rows][:, non_empty_cols]
        labels_row = unique_labels[non_empty_rows]
        labels_col = unique_labels[non_empty_cols]

        # Adjust figure size based on matrix dimensions
        fig_width = min(2 + 0.5 * len(labels_col), 25)
        fig_height = min(2 + 0.5 * len(labels_row), 25)

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels_col,
            yticklabels=labels_row
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {model_name}, EC_{i}")
        plt.tight_layout()

        cm_filename = os.path.join(model_cm_folder, f"ConfusionMatrix_EC_{i}.png")
        plt.savefig(cm_filename)
        plt.close()
        print(f"Saved confusion matrix to {cm_filename}")

    # Free memory
    del best_model
    gc.collect()
    
    return eval_report, test_pred

# ---------------------------
# 5. Ablation + Train + Save + Evaluate
# ---------------------------
def ablation_study():
    """
    Perform an ablation study by removing each group of features in turn,
    then train a multi-output RandomForest model, save it, and evaluate on test set.
    Memory-optimized version.
    """
    # ========== A) Load data ==========
    print("Starting ablation study...")
    print_memory_usage()
    
    train_df, valid_df, test_df = load_data_with_full_columns()
    print("Data loaded successfully.")
    
    # Identify all columns and group them (keep only references, no copies)
    all_feature_cols = [col for col in train_df.columns if not col.startswith('ec_')]
    col_to_group = {col: feature_group_map(col) for col in all_feature_cols}

    # Split into features & labels (creates views, not full copies)
    print("Splitting features and labels...")
    X_train_df_full, y_train_df = get_feature_and_label(train_df)
    X_valid_df_full, y_valid_df = get_feature_and_label(valid_df)
    X_test_df_full, y_test_df = get_feature_and_label(test_df)
    
    # Free memory from original DataFrames
    del train_df, valid_df
    gc.collect()
    print_memory_usage()

    # Unique groups
    unique_groups = sorted(list(set(col_to_group.values())))
    print("Feature groups identified:", unique_groups)

    # Create output folders
    models_folder = "Ablation_Models"
    results_base_folder = "../metrics/Ablation_Results"
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(results_base_folder, exist_ok=True)

    confusion_matrix_folder = os.path.join(results_base_folder, "confusion_matrices")
    reports_folder = os.path.join(results_base_folder, "evaluation_reports")
    for folder in [confusion_matrix_folder, reports_folder]:
        os.makedirs(folder, exist_ok=True)

    all_eval_reports = []  # Store as list instead of growing DataFrame
    batch_size = 500

    # ========== B) First train with ALL features as baseline ==========
    print("\n=== Training with ALL features (baseline) ===")
    print_memory_usage()
    
    # Scale data with all features
    print("Scaling features for baseline (all features)...")
    scaler_all = MinMaxScaler()
    
    print("Fitting scaler on all features...")
    # Fit in chunks
    for i in range(0, len(X_train_df_full), batch_size):
        end_idx = min(i + batch_size, len(X_train_df_full))
        batch = X_train_df_full.iloc[i:end_idx].values.astype(np.float32)
        if i == 0:  # First batch
            scaler_all.fit(batch)
        else:
            scaler_all.partial_fit(batch)
        del batch
        gc.collect()
    
    print("Transforming training data...")
    X_train_scaled_parts = []
    for i in range(0, len(X_train_df_full), batch_size):
        end_idx = min(i + batch_size, len(X_train_df_full))
        batch = X_train_df_full.iloc[i:end_idx].values.astype(np.float32)
        scaled_batch = scaler_all.transform(batch)
        X_train_scaled_parts.append(scaled_batch)
        del batch, scaled_batch
        gc.collect()
    
    X_train_scaled_all = np.vstack(X_train_scaled_parts)
    del X_train_scaled_parts
    gc.collect()
    
    print("Transforming validation data...")
    X_valid_scaled_all = scaler_all.transform(X_valid_df_full.values.astype(np.float32))
    
    print("Transforming test data...")
    X_test_scaled_all = scaler_all.transform(X_test_df_full.values.astype(np.float32))
    
    y_train_all = y_train_df.values
    y_valid_all = y_valid_df.values
    
    # Prepare train/val data
    print("Preparing train/val data...")
    X_trainval_all, y_trainval_all, _ = prepare_train_val(
        X_train_scaled_all, X_valid_scaled_all, y_train_all, y_valid_all
    )
    
    # Free original scaled data
    del X_train_scaled_all, X_valid_scaled_all, y_train_all, y_valid_all
    gc.collect()
    
    # Train and evaluate with all features
    baseline_report, baseline_pred = train_and_evaluate_model(
        X_trainval_all, y_trainval_all, 
        X_test_scaled_all, y_test_df.values, y_test_df,
        "AllFeatures", models_folder, confusion_matrix_folder, reports_folder,
        batch_size
    )
    
    all_eval_reports.append(baseline_report)
    
    # Free memory before ablation loop
    del X_trainval_all, y_trainval_all, baseline_pred
    gc.collect()
    print_memory_usage()

    # ========== C) Ablation Loop ==========
    for group_to_remove in unique_groups:
        print(f"\n=== Ablation: Removing group '{group_to_remove}' ===")
        print_memory_usage()
        
        # Keep columns that do NOT belong to the group_to_remove
        cols_to_keep = [c for c in all_feature_cols if col_to_group[c] != group_to_remove]
        
        # Index selection is more memory-efficient than creating full copies
        X_train_df_ablation = X_train_df_full[cols_to_keep]
        X_valid_df_ablation = X_valid_df_full[cols_to_keep]
        X_test_df_ablation = X_test_df_full[cols_to_keep]
        
        print(f"Feature subset selected after removing {group_to_remove}. Shape: {X_train_df_ablation.shape}")
        print_memory_usage()

        # Scale data - fit on smaller chunks if needed
        print("Scaling features...")
        scaler = MinMaxScaler()
        
        # Try to scale in chunks if the data is large
        if len(X_train_df_ablation) > batch_size:
            # Partial fit in chunks
            for i in range(0, len(X_train_df_ablation), batch_size):
                end_idx = min(i + batch_size, len(X_train_df_ablation))
                batch = X_train_df_ablation.iloc[i:end_idx].values.astype(np.float32)
                if i == 0:  # First batch - need to initialize scaler
                    scaler.fit(batch)
                else:  # Subsequent batches
                    scaler.partial_fit(batch)
                del batch
                gc.collect()
        else:
            # Fit directly if data is small enough
            scaler.fit(X_train_df_ablation.values.astype(np.float32))
        
        print("Scaling training data...")
        # Transform in chunks and collect results
        X_train_scaled_parts = []
        for i in range(0, len(X_train_df_ablation), batch_size):
            end_idx = min(i + batch_size, len(X_train_df_ablation))
            batch = X_train_df_ablation.iloc[i:end_idx].values.astype(np.float32)
            scaled_batch = scaler.transform(batch)
            X_train_scaled_parts.append(scaled_batch)
            del batch, scaled_batch
            gc.collect()
            
        X_train_scaled = np.vstack(X_train_scaled_parts)
        del X_train_scaled_parts
        gc.collect()
        
        print("Scaling validation data...")
        X_valid_scaled = scaler.transform(X_valid_df_ablation.values.astype(np.float32))
        
        print("Scaling test data...")
        X_test_scaled = scaler.transform(X_test_df_ablation.values.astype(np.float32))
        
        # Free memory from DataFrame slices
        del X_train_df_ablation, X_valid_df_ablation, X_test_df_ablation
        gc.collect()
        print_memory_usage()

        # Convert labels
        y_train = y_train_df.values  # shape (m_train, k)
        y_valid = y_valid_df.values  # shape (m_valid, k)
        
        # Prepare for training
        print("Preparing train/val data...")
        X_trainval, y_trainval, pds = prepare_train_val(X_train_scaled, X_valid_scaled, y_train, y_valid)
        
        # Free memory
        del X_train_scaled, X_valid_scaled, y_train, y_valid
        gc.collect()
        print_memory_usage()

        # Train and evaluate model
        ablation_tag = group_to_remove.replace(" ", "_")
        eval_report, _ = train_and_evaluate_model(
            X_trainval, y_trainval, 
            X_test_scaled, y_test_df.values, y_test_df,
            f"Remove_{ablation_tag}", models_folder, confusion_matrix_folder, reports_folder,
            batch_size
        )
        
        all_eval_reports.append(eval_report)

        # Free memory from large objects before next iteration
        del X_trainval, y_trainval, X_test_scaled
        gc.collect()
        print_memory_usage()

    # ========== D) Save aggregated reports ==========
    if all_eval_reports:
        # Concatenate the reports only at the end
        combined_reports = pd.concat(all_eval_reports, axis=0, ignore_index=True)
        all_reports_path = os.path.join(results_base_folder, "all_ablation_eval_reports.csv")
        combined_reports.to_csv(all_reports_path, index=False)
        print(f"\nAll ablation evaluation reports saved to {all_reports_path}")

# ---------------------------
# 6. Main
# ---------------------------
if __name__ == '__main__':
    np.set_printoptions(precision=4)
    
    # Print initial memory usage
    print("Initial memory usage:")
    print_memory_usage()
    
    # Run the ablation study
    ablation_study()
    
    # Final memory cleanup
    gc.collect()
    print("Final memory usage:")
    print_memory_usage()