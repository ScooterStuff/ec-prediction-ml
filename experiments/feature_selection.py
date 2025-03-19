import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.feature_selection import RFECV, mutual_info_classif, VarianceThreshold
from evaluate_ec import evaluate_ec_predictions
import matplotlib.pyplot as plt
import pickle
import sys
sys.modules["numpy._core.numeric"] = np.core.numeric

train_df = pd.read_pickle("../dataset/all_features/train.pkl")
valid_df = pd.read_pickle("../dataset/all_features/valid.pkl")
test_df = pd.read_pickle("../dataset/all_features/test.pkl")

def get_feature_and_label(df):
    ec_cols = [col for col in df.columns if col.startswith('ec_')]
    return df.drop(columns=ec_cols), df[ec_cols]

train_X, train_Y = get_feature_and_label(train_df)
valid_X, valid_Y = get_feature_and_label(valid_df)
test_X, test_Y = get_feature_and_label(test_df)

scaler = MinMaxScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
valid_X = scaler.transform(valid_X)
test_X = scaler.transform(test_X)

def multioutput_f1_score(y_true, y_pred):
    # Flatten all outputs and compute micro-F1
    return f1_score(y_true.ravel(), y_pred.ravel(), average='micro', zero_division=0)

X_trainval = np.concatenate([train_X, valid_X], axis=0)
y_trainval = np.concatenate([train_Y, valid_Y], axis=0)
split_index = np.concatenate([
    np.full(len(train_X), -1),  # Training samples (-1 means they are used for training)
    np.full(len(valid_X), 0)    # Validation samples (0 means they are used for validation)
])
pds = PredefinedSplit(test_fold = split_index)
f1_scorer = make_scorer(multioutput_f1_score)

# --- Step 1: Variance Filtering ---
print("Step 1: Variance Filtering...")
var_selector = VarianceThreshold(threshold=0.01)
X_var_filtered = var_selector.fit_transform(X_trainval)
var_mask = var_selector.get_support()
var_indices = np.where(var_mask)[0]

# --- Step 2: Multi-output Feature Selection ---
print("Step 2: Multi-output Feature Selection...")
k_best = min(500, X_var_filtered.shape[1])
num_outputs = y_trainval.shape[1]
feature_scores = np.zeros((X_var_filtered.shape[1],))
for i in range(num_outputs):
    y_single = y_trainval[:, i]
    scores = mutual_info_classif(X_var_filtered, y_single, random_state=42)
    feature_scores += scores
top_indices = np.argsort(feature_scores)[-k_best:]
X_filtered = X_var_filtered[:, top_indices]
filtered_indices = var_indices[top_indices]

# --- Step 3: Final Feature Selection using RFECV ---
print(f"Step 3: Running RFECV on {X_filtered.shape[1]} features...")
rfecv = RFECV(
    estimator=RandomForestClassifier(random_state=42, n_estimators=100),
    step=5,
    cv=pds,
    scoring=f1_scorer,
    n_jobs=6,
    verbose=2
)
rfecv.fit(X_filtered, y_trainval)
final_indices = filtered_indices[rfecv.support_]
print("Optimal number of features:", rfecv.n_features_)
print("Optimal feature indices:", final_indices.tolist())

with open("final_indices.pkl", "wb") as f:
    pickle.dump(final_indices, f)
with open("grid_scores.pkl", "wb") as f:
    pickle.dump(rfecv.cv_results_, f)
print("Final indices and grid scores have been saved.")


