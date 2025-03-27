#!/usr/bin/env python
import os
import sys
import gc
import pickle
import re
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, MinMaxScaler

# Import your evaluation function and SVM wrapper
from evaluate_ec import evaluate_ec_predictions
from svm_training import SVCWrapper

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
# Load datasets (adjust paths if needed)
train_df = pd.read_pickle("../dataset/all_features/train.pkl")
valid_df = pd.read_pickle("../dataset/all_features/valid.pkl")
test_df  = pd.read_pickle("../dataset/all_features/test.pkl")

# Define a helper function to separate features and labels.
def get_feature_and_label(df):
    # Assume columns starting with 'ec_' are labels.
    ec_cols = [col for col in df.columns if col.startswith('ec_')]
    return df.drop(columns=ec_cols), df[ec_cols]

train_X, train_Y = get_feature_and_label(train_df)
valid_X, valid_Y = get_feature_and_label(valid_df)
test_X, test_Y   = get_feature_and_label(test_df)

# Scale features using MinMaxScaler (as float32)
scaler = MinMaxScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X).astype(np.float32)
valid_X = scaler.transform(valid_X).astype(np.float32)
test_X  = scaler.transform(test_X).astype(np.float32)

# Free memory by deleting train and valid DataFrames and their labels
del train_df, valid_df, train_Y, valid_Y
gc.collect()

# -------------------------------
# Set Up Output Directories
# -------------------------------
models_folder = "Models"
results_base_folder = "../metrics"
os.makedirs(results_base_folder, exist_ok=True)

confusion_matrix_folder = os.path.join(results_base_folder, "confusion_matrices", "all_cascade_models_SVM")
roc_curves_folder = os.path.join(results_base_folder, "roc_curves", "all_cascade_models_SVM")
reports_folder = os.path.join(results_base_folder, "evaluation_reports")
for folder in [confusion_matrix_folder, roc_curves_folder, reports_folder]:
    os.makedirs(folder, exist_ok=True)

# Initialize storage for ROC comparison.
# This will collect ROC data (FPR, TPR, AUC scores, and model names) for each output.
roc_comparison_data = defaultdict(lambda: {
    'model_names': [],
    'fprs': [],
    'tprs': [],
    'auc_scores': []
})

# -------------------------------
# SVM Cascade Prediction
# -------------------------------
# Folder storing individual SVM cascade step models
cascade_steps_dir = os.path.join(models_folder, "SVM_cascade_steps")
if not os.path.exists(cascade_steps_dir):
    print("SVM cascade steps directory not found:", cascade_steps_dir)
    sys.exit(1)

# Get all cascade step files that match the expected pattern,
# e.g., cascade_step_0_model.pkl, cascade_step_1_model.pkl, etc.
pattern = re.compile(r'^cascade_step_(\d+)_model\.pkl$')
cascade_step_files = sorted(
    [f for f in os.listdir(cascade_steps_dir) if pattern.match(f)],
    key=lambda x: int(pattern.match(x).group(1))
)

# 用于存储每个级联模型的classes信息，避免后续重复加载模型
cascade_model_classes = []

# Prepare for cascade prediction
cascade_test_features = test_X.copy()
test_preds_list = []   # Will store predicted labels from each cascade step
roc_probs_list = []    # Will store probability matrices from each cascade step
num_outputs = test_Y.shape[1]

# Loop over each cascade step file, load the model, and generate predictions
for step_file in tqdm(cascade_step_files, desc="SVM Cascade Prediction", unit="step"):
    model_path = os.path.join(cascade_steps_dir, step_file)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # 如果模型没有 classes_ 属性，则根据 n_classes 或预测结果手动设置
    if not hasattr(model, 'classes_'):
        if hasattr(model, 'n_classes'):
            model.classes_ = np.arange(model.n_classes)
        else:
            try:
                proba = model.predict_proba(cascade_test_features[:1])
                model.classes_ = np.arange(proba.shape[1])
            except Exception as e:
                # 默认假设二分类
                model.classes_ = np.array([0, 1])
    # 保存当前模型的 classes 信息
    cascade_model_classes.append(model.classes_)
    
    # Get probability estimates
    try:
        prob_matrix = model.predict_proba(cascade_test_features)
    except Exception as e:
        print(f"Error in predict_proba for {step_file}: {e}")
        prob_matrix = np.zeros((len(cascade_test_features), 2))
    roc_probs_list.append(prob_matrix)
    
    # Get predicted labels
    try:
        y_pred_col = model.predict(cascade_test_features)
    except Exception as e:
        print(f"Error in predict for {step_file}: {e}")
        y_pred_col = -1 * np.ones(len(cascade_test_features), dtype=int)
    test_preds_list.append(y_pred_col)
    
    # Append current prediction as an additional feature for the next cascade step
    cascade_test_features = np.hstack([cascade_test_features, y_pred_col.reshape(-1, 1)])
    
    # Delete model to free memory
    del model
    gc.collect()

# Combine predictions from all cascade steps (each column corresponds to one step)
test_pred = np.column_stack(test_preds_list)

# -------------------------------
# Evaluation Report Generation
# -------------------------------
# Evaluate SVM cascade performance using your provided evaluation function.
eval_report = evaluate_ec_predictions(
    test_pred,
    test_Y.to_numpy(),
    method_name="all + SVM_Cascade"
)

# Save the evaluation report
report_path = os.path.join(reports_folder, "SVM_Cascade_eval_report.csv")
eval_report.to_csv(report_path, index=False)
print("Saved SVM cascade evaluation report to", report_path)

# -------------------------------
# Confusion Matrix Plotting
# -------------------------------
for i in range(num_outputs):
    y_true = test_Y.to_numpy()[:, i]
    y_pred = test_pred[:, i]
    mask = (y_true != -1)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    unique_labels = np.unique(np.concatenate([y_true_filtered, y_pred_filtered]))
    unique_labels = np.sort(unique_labels)

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=unique_labels)
    # Normalize the confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(float), row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=(row_sums != 0)
    )
    # Filter out rows/columns with all zeros
    non_empty_rows = cm_normalized.sum(axis=1) != 0
    non_empty_cols = cm_normalized.sum(axis=0) != 0
    cm_normalized = cm_normalized[non_empty_rows][:, non_empty_cols]
    labels_row = unique_labels[non_empty_rows]
    labels_col = unique_labels[non_empty_cols]

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
    plt.title(f"Confusion Matrix for SVM_Cascade - Output EC_{i}")
    plt.tight_layout()

    cm_filename = os.path.join(confusion_matrix_folder, f"ConfusionMatrix_OutputEC_{i}.png")
    plt.savefig(cm_filename)
    plt.close()
    print(f"Saved confusion matrix for SVM_Cascade, Output EC_{i} to {cm_filename}")

# -------------------------------
# ROC Curve Plotting and Saving ROC Data
# -------------------------------
roc_curves_folder = os.path.join(results_base_folder, "roc_curves", "all_cascade_models_SVM")
os.makedirs(roc_curves_folder, exist_ok=True)

for i in range(num_outputs):
    y_true_col = test_Y.iloc[:, i].to_numpy()
    mask = y_true_col != -1
    y_true_filtered = y_true_col[mask]
    
    # Skip if there are insufficient samples for ROC computation
    if len(y_true_filtered) == 0 or len(np.unique(y_true_filtered)) < 2:
        print(f"Skipping ROC for SVM_Cascade, Output EC_{i}: insufficient samples.")
        continue

    # Use the probability matrix corresponding to the cascade step
    # 注意：确保 roc_probs_list 的长度与 cascade_model_classes 一致
    prob_matrix = roc_probs_list[i][mask, :]

    # 直接使用在预测阶段保存的 classes 信息
    classes = cascade_model_classes[i]

    valid_labels = np.isin(y_true_filtered, classes)
    if not np.all(valid_labels):
        y_true_filtered = y_true_filtered[valid_labels]
        prob_matrix = prob_matrix[valid_labels]
    if len(y_true_filtered) < 2:
        print(f"Skipping ROC for SVM_Cascade, Output EC_{i}: insufficient valid samples.")
        continue

    # Binarize true labels
    y_true_bin = label_binarize(y_true_filtered, classes=classes)
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), prob_matrix.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Save ROC data for later combined plotting
    roc_comparison_data[i]['model_names'].append("SVM_Cascade")
    roc_comparison_data[i]['fprs'].append(fpr_micro)
    roc_comparison_data[i]['tprs'].append(tpr_micro)
    roc_comparison_data[i]['auc_scores'].append(roc_auc_micro)

    # Plot individual ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-avg (AUC = {roc_auc_micro:.2f})',
             linestyle=':', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC for SVM_Cascade - Output EC_{i}")
    plt.legend(loc="lower right")
    
    roc_filename = os.path.join(roc_curves_folder, f"ROC_EC_{i}.png")
    plt.savefig(roc_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve for SVM_Cascade, Output EC_{i} to {roc_filename}")

# -------------------------------
# Save ROC Comparison Data
# -------------------------------
roc_info_file = os.path.join(results_base_folder, "roc_comparison_data.pkl")
roc_comparison_data_dict = dict(roc_comparison_data)
with open(roc_info_file, "wb") as f:
    pickle.dump(roc_comparison_data_dict, f)
print("Saved ROC comparison data to", roc_info_file)