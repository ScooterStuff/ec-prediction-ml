import os
import gc
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import PredefinedSplit
from sklearn.feature_selection import RFECV, mutual_info_classif, VarianceThreshold
import pickle
import sys
sys.modules["numpy._core.numeric"] = np.core.numeric
output_dir = "../metrics/feature_selection_results"
os.makedirs(output_dir, exist_ok=True)

# ----------------- Data Loading and Preprocessing -----------------
train_df = pd.read_pickle("../dataset/all_features/train.pkl")
valid_df = pd.read_pickle("../dataset/all_features/valid.pkl")
test_df = pd.read_pickle("../dataset/all_features/test.pkl")

def get_feature_and_label(df):
    ec_cols = [col for col in df.columns if col.startswith('ec_')]
    return df.drop(columns=ec_cols), df[ec_cols]

train_X, train_Y = get_feature_and_label(train_df)
valid_X, valid_Y = get_feature_and_label(valid_df)
test_X, test_Y = get_feature_and_label(test_df)
del train_df, valid_df, test_df
gc.collect()

# Normalize the data; converting to float32 helps reduce memory usage
scaler = MinMaxScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X).astype(np.float32)
valid_X = scaler.transform(valid_X).astype(np.float32)
test_X = scaler.transform(test_X).astype(np.float32)

# ----------------- Step 1: Variance Filtering -----------------
print("Step 1: Variance Filtering...")

# Compute the variance of each feature (using training data)
variances = np.var(train_X, axis=0)
variance_ranking_all = np.argsort(variances)

variance_info = {
    "feature_indices": np.arange(train_X.shape[1]),
    "variances": variances,
    "ranking": variance_ranking_all
}
with open(os.path.join(output_dir, "variance_info.pkl"), "wb") as f:
    pickle.dump(variance_info, f)

# Fit VarianceThreshold on training set and transform both train and valid sets
var_selector = VarianceThreshold(threshold=0.001)
X_var_filtered_train = var_selector.fit_transform(train_X)
X_var_filtered_valid = var_selector.transform(valid_X)

var_mask = var_selector.get_support()
var_indices = np.where(var_mask)[0]

# Re-rank the filtered features by variance
variances_filtered = variances[var_indices]
variance_ranking_filtered = var_indices[np.argsort(variances_filtered)]
variance_info_filtered = {
    "filtered_feature_indices": var_indices,
    "filtered_variances": variances_filtered,
    "filtered_ranking": variance_ranking_filtered
}
with open(os.path.join(output_dir, "variance_info_filtered.pkl"), "wb") as f:
    pickle.dump(variance_info_filtered, f)

# ----------------- Step 2: Multi-output Feature Selection -----------------
print("Step 2: Multi-output Feature Selection...")
k_best = min(1000, X_var_filtered_train.shape[1])
num_outputs = train_Y.shape[1]
n_features = X_var_filtered_train.shape[1]
feature_scores = np.zeros(n_features)

# Use chunking to compute mutual information to reduce peak memory usage
chunk_size = 500 
for i in tqdm(range(num_outputs)):
    y_single = train_Y.iloc[:, i]
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        # Compute mutual information on a chunk of features
        scores_chunk = mutual_info_classif(X_var_filtered_train[:, start:end], y_single, random_state=42)
        feature_scores[start:end] += scores_chunk
    del y_single
    gc.collect()

# Save cumulative mutual information scores
with open(os.path.join(output_dir, "mutual_info_scores.pkl"), "wb") as f:
    pickle.dump(feature_scores, f)

sorted_indices = np.argsort(feature_scores)
ranking_info = {
    "sorted_indices": sorted_indices,
    "scores": feature_scores[sorted_indices]
}
with open(os.path.join(output_dir, "mutual_info_ranking_all.pkl"), "wb") as f:
    pickle.dump(ranking_info, f)

top_indices = np.argsort(feature_scores)[-k_best:]
X_filtered_train = X_var_filtered_train[:, top_indices]
filtered_indices = var_indices[top_indices]

top_scores = feature_scores[top_indices]
top_ranking_info = {
    "top_indices": top_indices,
    "original_feature_indices": filtered_indices,
    "top_scores": top_scores
}
with open(os.path.join(output_dir, "mutual_info_top_ranking.pkl"), "wb") as f:
    pickle.dump(top_ranking_info, f)

X_filtered_valid = X_var_filtered_valid[:, top_indices]

# Free memory from variables that are no longer needed
del X_var_filtered_train, X_var_filtered_valid, feature_scores
gc.collect()

# ----------------- Step 3: Final Feature Selection using RFECV ----------------
def multioutput_f1_score(y_true, y_pred):
    return f1_score(y_true.ravel(), y_pred.ravel(), average='micro', zero_division=0)

X_trainval = np.concatenate([X_filtered_train, X_filtered_valid], axis=0)
y_trainval = np.concatenate([train_Y.to_numpy(), valid_Y.to_numpy()], axis=0)

# Create PredefinedSplit based on actual sample sizes
n_train = X_filtered_train.shape[0]
n_valid = X_filtered_valid.shape[0]
split_index = np.concatenate([
    np.full(n_train, -1),  # Training samples marked as -1
    np.full(n_valid, 0)    # Validation samples marked as 0
])
pds = PredefinedSplit(test_fold=split_index)
f1_scorer = make_scorer(multioutput_f1_score)

# Free memory from original (unfiltered) arrays
del train_X, valid_X, variances
gc.collect()

print(f"Step 3: Running RFECV on {X_filtered_train.shape[1]} features...")
rfecv = RFECV(
    estimator=RandomForestClassifier(random_state=42, n_estimators=100),
    step=5,
    cv=pds,
    scoring=f1_scorer,
    n_jobs=8,
    verbose=2,
    min_features_to_select=100
)
rfecv.fit(X_trainval, y_trainval)

final_indices = filtered_indices[rfecv.support_]
print("Optimal number of features:", rfecv.n_features_)
print("Optimal feature indices:", final_indices.tolist())

with open(os.path.join(output_dir, "final_indices.pkl"), "wb") as f:
    pickle.dump(final_indices, f)

with open(os.path.join(output_dir, "grid_scores.pkl"), "wb") as f:
    pickle.dump(rfecv.cv_results_, f)

rfecv_ranking_info = {"rfecv_ranking": rfecv.ranking_, "support": rfecv.support_}
with open(os.path.join(output_dir, "rfecv_ranking_info.pkl"), "wb") as f:
    pickle.dump(rfecv_ranking_info, f)

print("Feature ranking information from all steps has been saved in ../metrics/feature_selection_results.")

# Clean up remaining large variables
del top_ranking_info, ranking_info, rfecv, X_filtered_train, X_filtered_valid, filtered_indices, y_trainval
gc.collect()
