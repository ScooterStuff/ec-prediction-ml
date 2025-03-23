#!/usr/bin/env python
"""
cascade_model_training_only.py

This script trains cascade models using several candidate classifiers and saves
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

# Candidate models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

# For pickle version
import sys
sys.modules["numpy._core.numeric"] = np.core.numeric

# ---------------------------
# Helper Functions
# ---------------------------
def load_data(feature_method):
    """Load train, validation, and test data from pickle files."""
    train_df = pd.read_pickle(f"../dataset/{feature_method}_features/train.pkl")
    valid_df = pd.read_pickle(f"../dataset/{feature_method}_features/valid.pkl")
    test_df  = pd.read_pickle(f"../dataset/{feature_method}_features/test.pkl")
    
    def get_feature_and_label(df):
        ec_cols = [col for col in df.columns if col.startswith('ec_')]
        return df.drop(columns=ec_cols), df[ec_cols]
    
    train_X, train_Y = get_feature_and_label(train_df)
    valid_X, valid_Y = get_feature_and_label(valid_df)
    test_X,  test_Y  = get_feature_and_label(test_df)

    # apply selected features
    if feature_method == "selected":
        with open("final_indices.pkl", "rb") as f:
            final_indices = pickle.load(f)
        train_X = train_X.iloc[:, final_indices]
        valid_X = valid_X.iloc[:, final_indices]
        test_X = test_X.iloc[:, final_indices]
    
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

def cascade_model_training(model_base, X_trainval, y_trainval, feature_method):
    """
    Train a cascade of models (one per output) using the given base model.
    Returns the list of trained models.
    """
    models = []
    cascade_features = X_trainval.copy()
    
    for i in tqdm(range(y_trainval.shape[1]), desc="Cascade Training"):
        valid_indices = y_trainval[:, i] != -1  # assume -1 indicates invalid label
        X_trainval_i = cascade_features[valid_indices]
        y_trainval_i = y_trainval[valid_indices, i]
        
        model = clone(model_base)
        model.fit(X_trainval_i, y_trainval_i)
        models.append(model)
        
        # Append predictions to features for next cascade step
        pred = model.predict(cascade_features).reshape(-1, 1)
        cascade_features = np.hstack([cascade_features, pred])
        
    return models

def run_model_training(feature_method="selected"):
    """Train cascade models for different candidate models and save them."""
    # Load and prepare data
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = load_data(feature_method)
    X_trainval, y_trainval, pds = prepare_train_val(train_X, valid_X, train_Y, valid_Y)
    
    # Dictionary of candidate models to test
    model_constructors = {
        "KNN": KNeighborsClassifier(),
        "LogisticRegression": LogisticRegression(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None),
        "CatBoost": CatBoostClassifier(random_state=42, iterations=200, verbose=0),
        "SVM": SVC(random_state=42, probability=True)
    }
    
    # Create a directory to save models if not exists
    models_dir = "Selected_Feature_Models" if feature_method == "selected" else "Models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Train cascade models for each candidate and save them
    for model_name, model_base in model_constructors.items():
        print(f"\n==== Training cascade model for: {model_name} ====")
        models = cascade_model_training(model_base, X_trainval, y_trainval, feature_method)
        
        model_path = os.path.join(models_dir, f"{feature_method}_cascade_models_{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(models, f)
        print(f"Saved cascade models for {model_name} to {model_path}")

# ---------------------------
# Main entry point
# ---------------------------
if __name__ == '__main__':
    run_model_training(feature_method="all")
