#!/usr/bin/env python
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import PredefinedSplit
from sklearn.base import clone
import gc
import psutil  # for monitoring memory usage
from thundersvm import SVC  # base model

# For pickle compatibility with older numpy versions
import sys
sys.modules["numpy._core.numeric"] = np.core.numeric

class SVCWrapper:
    def __init__(self, svc):
        self.svc = svc
        self._estimator_type = "classifier"
        self.classes_ = svc.classes_
    def fit(self, X, y):
        return self
    def predict(self, X):
        return self.svc.predict(X)
    def decision_function(self, X):
        if hasattr(self.svc, "decision_function"):
            return self.svc.decision_function(X)
        else:
            return self.svc.predict(X)

def print_memory_usage(note=""):
    """Print current memory usage in MB with an optional note."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"{note} - Memory usage: {mem:.2f} MB")

# ---------------------------
# Data Loading and Preparation
# ---------------------------
def load_data(feature_method):
    """Load train, validation, and test data from pickle files."""
    train_df = pd.read_pickle(f"../dataset/all_features/train.pkl")
    valid_df = pd.read_pickle(f"../dataset/all_features/valid.pkl")
    test_df  = pd.read_pickle(f"../dataset/all_features/test.pkl")
    
    def get_feature_and_label(df):
        ec_cols = [col for col in df.columns if col.startswith('ec_')]
        return df.drop(columns=ec_cols), df[ec_cols]
    
    train_X, train_Y = get_feature_and_label(train_df)
    valid_X, valid_Y = get_feature_and_label(valid_df)
    test_X,  test_Y  = get_feature_and_label(test_df)
    
    if feature_method == "selected":
        with open("../metrics/feature_selection_results/final_indices.pkl", "rb") as f:
            final_indices = pickle.load(f)
        train_X = train_X.iloc[:, final_indices]
        valid_X = valid_X.iloc[:, final_indices]
        test_X = test_X.iloc[:, final_indices]
    
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X).astype(np.float32)
    valid_X = scaler.transform(valid_X).astype(np.float32)
    test_X  = scaler.transform(test_X).astype(np.float32)
    
    del train_df, valid_df, test_df
    gc.collect()
    
    # Merge training and validation sets
    X_trainval = np.concatenate([train_X, valid_X], axis=0)
    y_trainval = np.concatenate([train_Y, valid_Y], axis=0)
    return X_trainval, y_trainval

# ---------------------------
# Batch Prediction
# ---------------------------
def batch_predict(model, X, batch_size=1024):
    """Predict in batches to avoid large memory allocations."""
    preds = []
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        batch = X[i:i+batch_size]
        preds.append(model.predict(batch))
    return np.concatenate(preds, axis=0)

# ---------------------------
# Update and Save Cascade Features Layer by Layer
# ---------------------------
def update_cascade_features(cascade_dir, total_steps, X_trainval, init_feature_count):
    """
    Sequentially load models for each layer, predict on training data,
    and concatenate to create new features, then save the updated complete
    cascade features to the corresponding file.
    
    Parameters:
      cascade_dir: Directory for saving models and features (e.g., "Models/SVM_cascade_steps").
      total_steps: Total number of cascade layers to process (e.g., 3).
      X_trainval: Original training data (merged train and validation sets).
      init_feature_count: Initial number of features.
    
    Returns:
      cascade_features: Updated complete cascade feature matrix.
      current_features: Final number of features (initial features + cascade layers).
    """
    cascade_features = X_trainval.copy()
    current_features = init_feature_count

    for i in range(total_steps):
        model_path = os.path.join(cascade_dir, f"cascade_step_{i}_model.pkl")
        features_path = os.path.join(cascade_dir, f"cascade_step_{i}_features.pkl")
        if os.path.exists(model_path) and os.path.exists(features_path):
            with open(features_path, "rb") as f:
                saved_dict = pickle.load(f)
            cascade_features[:, :saved_dict["current_features"]] = saved_dict["cascade_features"]
            current_features = saved_dict["current_features"]
            print(f"Step {i}: Loaded saved cascade features (current_features={current_features}).")
        else:
            # Load current layer model
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                print(f"Step {i}: Loaded model from {model_path}.")
            else:
                print(f"Step {i}: Model file not found! Cannot update cascade features.")
                break
            # Batch predict on existing features
            pred = batch_predict(model, cascade_features[:, :current_features]).reshape(-1, 1).astype(np.float32)
            # Concatenate prediction results to feature matrix
            cascade_features = np.hstack([cascade_features, pred])
            current_features += 1
            # Save updated complete features
            save_dict = {
                "current_features": current_features,
                "cascade_features": cascade_features[:, :current_features].copy()
            }
            with open(features_path, "wb") as f:
                pickle.dump(save_dict, f)
            print(f"Step {i}: Saved updated cascade features to {features_path}.")
        # Optional: release prediction variables
        gc.collect()
    return cascade_features, current_features

# ---------------------------
# Main Script
# ---------------------------
def main():
    X_trainval, y_trainval = load_data("all")
    print_memory_usage("After data load")
    init_feature_count = X_trainval.shape[1]
    
    # Specify cascade model save directory and total cascade layers
    cascade_dir = os.path.join("Models", "SVM_cascade_steps")
    total_steps = 3  # Assuming 3 layers of models exist
    # Update cascade features: sequentially load models for each layer and predict
    cascade_features, current_features = update_cascade_features(cascade_dir, total_steps, X_trainval, init_feature_count)
    print(f"Final cascade features shape: {cascade_features.shape}, current_features: {current_features}")
    
    # Subsequently can use updated cascade_features to train next layer model
    # For example, train fourth cascade model:
    model_base = SVC(random_state=42, probability=True)
    # Assuming target column is the 4th column (index 3) in y_trainval
    target_col = 3
    # Train fourth model, input is complete cascade_features, target is y_trainval[:, target_col]
    from sklearn.base import clone
    fourth_model = clone(model_base)
    mask = y_trainval[:, target_col] != -1
    X_input = cascade_features[mask, :]
    y_target = y_trainval[mask, target_col]
    fourth_model.fit(X_input, y_target)
    
    fourth_save_path = os.path.join(cascade_dir, f"cascade_step_{target_col}_model.pkl")
    with open(fourth_save_path, "wb") as f:
        pickle.dump(fourth_model, f)
    print(f"Saved fourth cascade model to {fourth_save_path}")

if __name__ == '__main__':
    main()