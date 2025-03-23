#!/usr/bin/env python
"""
cascade_model_training_only.py

This script trains cascade models using a custom list of classifiers and saves
the trained cascade models to disk without performing any evaluation.
You can run this script in the background (e.g., in a tmux session) and later load
the saved models for prediction or further analysis.

Usage:
    nohup python cascade_model_training_only.py > training.log 2>&1 &
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import PredefinedSplit
from sklearn.base import clone

# Candidate models for custom cascade: two CatBoost and two SVMs.
from catboost import CatBoostClassifier
from sklearn.svm import SVC

# For pickle version
import sys
sys.modules["numpy._core.numeric"] = np.core.numeric

# ---------------------------
# Helper Functions
# ---------------------------
def load_data():
    """Load train, validation, and test data from pickle files."""
    train_df = pd.read_pickle("../dataset/all_features/train.pkl")
    valid_df = pd.read_pickle("../dataset/all_features/valid.pkl")
    test_df  = pd.read_pickle("../dataset/all_features/test.pkl")
    
    def get_feature_and_label(df):
        ec_cols = [col for col in df.columns if col.startswith('ec_')]
        return df.drop(columns=ec_cols), df[ec_cols]
    
    train_X, train_Y = get_feature_and_label(train_df)
    valid_X, valid_Y = get_feature_and_label(valid_df)
    test_X,  test_Y  = get_feature_and_label(test_df)

    # apply selected features
    # with open("final_indices.pkl", "rb") as f:
    #     final_indices = pickle.load(f)
    # train_X = train_X.iloc[:, final_indices]
    # valid_X = valid_X.iloc[:, final_indices]
    # test_X = test_X.iloc[:, final_indices]
    
    # Scale features
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    valid_X = scaler.transform(valid_X)
    test_X  = scaler.transform(test_X)
    
    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y

def prepare_train_val(train_X, valid_X, train_Y, valid_Y):
    """Concatenate train and validation sets and create a PredefinedSplit."""
    X_trainval = np.concatenate([train_X, valid_X], axis=0)
    y_trainval = np.concatenate([train_Y, valid_Y], axis=0)
    split_index = np.concatenate([
        np.full(len(train_X), -1),  # -1: training samples
        np.full(len(valid_X), 0)      # 0: validation samples
    ])
    pds = PredefinedSplit(test_fold=split_index)
    return X_trainval, y_trainval, pds

def cascade_custom_models(model_list, X_trainval, y_trainval):
    """
    Train a cascade of models using a custom list of models.
    
    Parameters:
        model_list (list): List of model objects for the cascade.
                           The order corresponds to each output position.
        X_trainval (np.array): Feature matrix of shape (n_samples, n_features).
        y_trainval (np.array): Label matrix of shape (n_samples, n_outputs).
                               Invalid labels should be marked as -1.
                               
    Returns:
        List of trained models for each output.
    """
    cascade_features = X_trainval.copy()
    trained_models = []
    num_outputs = y_trainval.shape[1]
    
    if len(model_list) != num_outputs:
        raise ValueError("The number of models in model_list must equal the number of outputs in y_trainval.")
    
    for i in tqdm(range(num_outputs), desc="Cascade Training with Custom Models"):
        # Select samples with valid label for the current output
        valid_indices = y_trainval[:, i] != -1
        X_train_i = cascade_features[valid_indices]
        y_train_i = y_trainval[valid_indices, i]
        
        # Clone and train the model for the current output
        model = clone(model_list[i])
        model.fit(X_train_i, y_train_i)
        trained_models.append(model)
        
        # Append predictions as new features for subsequent steps
        pred = model.predict(cascade_features).reshape(-1, 1)
        cascade_features = np.hstack([cascade_features, pred])
    
    return trained_models

def run_model_training():
    """Train cascade models using a custom list of classifiers and save them."""
    # Load and prepare data
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = load_data()
    X_trainval, y_trainval, pds = prepare_train_val(train_X, valid_X, train_Y, valid_Y)
    
    # all feature
    custom_models = [
        CatBoostClassifier(random_state=42, iterations=200, verbose=0),
        CatBoostClassifier(random_state=42, iterations=200, verbose=0),
        SVC(random_state=42, probability=True),
        SVC(random_state=42, probability=True)
    ]
    
    # Train the custom cascade models
    print("\n==== Training custom cascade models ====")
    models = cascade_custom_models(custom_models, X_trainval, y_trainval)
    
    # Create a directory to save models if it does not exist
    models_dir = "Models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the trained cascade models to disk
    model_path = os.path.join(models_dir, "cascade_models_custom.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"Saved custom cascade models to {model_path}")

# ---------------------------
# Main entry point
# ---------------------------
if __name__ == '__main__':
    run_model_training()
