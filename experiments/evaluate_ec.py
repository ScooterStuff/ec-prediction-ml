import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from collections import defaultdict

def load_data(ec40_file, ec_results_file):
    """Loads EC ground truth and predicted EC numbers."""
    ec40 = pd.read_csv(ec40_file)
    ec_results = pd.read_csv(ec_results_file)
    
    # Convert EC list strings into actual lists
    ec40['ec'] = ec40['ec'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    
    return ec40, ec_results

def match_predictions(ec40, ec_results):
    """Matches predicted EC numbers with ground truth."""
    results = []
    
    for _, row in tqdm(ec40.iterrows(), total=len(ec40), desc="Matching Predictions"):
        query = row['accession']
        true_ecs = row['ec']
        
        pred_row = ec_results[ec_results['Query'] == query]
        if pred_row.empty:
            pred_ec = 'No Prediction'
        else:
            pred_ec = pred_row.iloc[0]['EC Number']
        
        first_number_match = any(pred_ec.split('.')[0] == ec.split('.')[0] for ec in true_ecs if pred_ec != 'No Prediction')
        
        results.append({
            'Query': query,
            'True EC': true_ecs,
            'Predicted EC': pred_ec,
            'Exact Match': pred_ec in true_ecs if pred_ec != 'No Prediction' else False,
            'First Number Match': first_number_match,
            'First Number': pred_ec.split('.')[0] if pred_ec != 'No Prediction' else 'No Prediction'
        })
    
    return pd.DataFrame(results)

def compute_metrics(results):
    """Computes Accuracy, Precision, Recall, and F1-Score for both exact and first-number matches, including per first-number accuracy."""
    exact_matches = results['Exact Match'].sum()
    total_queries = len(results)
    accuracy_exact = exact_matches / total_queries * 100
    
    first_number_matches = results['First Number Match'].sum()
    accuracy_first_number = first_number_matches / total_queries * 100
    
    # Compute per first-number accuracy
    first_number_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for _, row in results.iterrows():
        pred_ec = row['Predicted EC']
        if pred_ec in ["No EC number found", "No Prediction"]:
            continue
        first_num = row['First Number']
        first_number_accuracy[first_num]['total'] += 1
        if row['First Number Match']:
            first_number_accuracy[first_num]['correct'] += 1
    
    print(f"Exact Match Accuracy: {accuracy_exact:.2f}%")
    print(f"First Number Match Accuracy: {accuracy_first_number:.2f}%")
    
    print("Per First-Number Accuracy:")
    first_num_acc = {
        "first_num": [],
        "accuracy (%)": []
    }
    for first_num, counts in first_number_accuracy.items():
        acc = (counts['correct'] / counts['total']) * 100 if counts['total'] > 0 else 0
        print(f"  EC {first_num}: {acc:.2f}%")
        first_num_acc["first_num"].append(first_num)
        first_num_acc["accuracy (%)"].append(acc)
    
    y_true = [1 if row['True EC'] else 0 for _, row in results.iterrows()]
    y_pred_exact = [1 if row['Exact Match'] else 0 for _, row in results.iterrows()]
    y_pred_first = [1 if row['First Number Match'] else 0 for _, row in results.iterrows()]
    
    precision_exact = precision_score(y_true, y_pred_exact, zero_division=0)
    recall_exact = recall_score(y_true, y_pred_exact, zero_division=0)
    f1_exact = f1_score(y_true, y_pred_exact, zero_division=0)
    
    precision_first = precision_score(y_true, y_pred_first, zero_division=0)
    recall_first = recall_score(y_true, y_pred_first, zero_division=0)
    f1_first = f1_score(y_true, y_pred_first, zero_division=0)
    
    print(f"Exact Precision: {precision_exact:.2f}, Recall: {recall_exact:.2f}, F1-Score: {f1_exact:.2f}")
    print(f"First Number Precision: {precision_first:.2f}, Recall: {recall_first:.2f}, F1-Score: {f1_first:.2f}")

    no_ec_count = results[results['Predicted EC'] == "No EC number found"].shape[0] / total_queries
    no_pred_count = results[results['Predicted EC'] == "No Prediction"].shape[0] / total_queries

    cols = [
        f"EC {num}" for num in first_num_acc["first_num"]
    ]
    sorted_cols = sorted(
        cols,
        key=lambda x: int(x.split()[1]) if x.startswith("EC ") else float('inf')
    )

    accuracy_df = pd.DataFrame({
        col: [acc] for col, acc in zip(cols, first_num_acc["accuracy (%)"])
    }, columns=sorted_cols)

    metrics_df = pd.DataFrame({
        "Method": ["DIMOND Benchmark"],
        "Exact Match Accuracy": [accuracy_exact],
        "First Number Match Accuracy": [accuracy_first_number],
        "Exact Precision": [precision_exact],
        "Exact Recall": [recall_exact],
        "Exact F1-Score": [f1_exact],
        "First Number Precision": [precision_first],
        "First Number Recall": [recall_first],
        "First Number F1-Score": [f1_first],
        "No EC number found": [no_ec_count],
        "No Prediction": [no_pred_count]
    })

    merged_df = pd.concat([metrics_df, accuracy_df], axis=1)

    return merged_df
    

def evaluate_ec(ec_results_file, metrics_file, evaluate_file, ec40_file):
    '''
    ec_results_file: EC given by DIAMOND
    metrics_file: Generated metrics
    evaluate_file: EC evaluate file
    ec40_file: EC40 file with true EC numberss
    '''
    
    print("Loading data...")
    ec40, ec_results = load_data(ec40_file, ec_results_file)
    
    print("Matching predictions...")
    results = match_predictions(ec40, ec_results)
    
    print("Computing evaluation metrics...")
    metrics = compute_metrics(results)
    
    # Save results for further analysis
    metrics.to_csv(metrics_file, index=False)
    results.to_csv(evaluate_file, index=False)
    print(f"Saved results to {metrics_file} and {evaluate_file}")


def evaluate_ec_predictions(true_labels: np.ndarray, pred_labels: np.ndarray) -> pd.DataFrame:
    """
    Evaluate n×4 EC prediction results and return a DataFrame with main evaluation metrics.

    Parameters
    ----------
    true_labels : np.ndarray, shape (n, 4)
        Ground truth EC numbers, each row is [EC1, EC2, EC3, EC4]. -1 indicates no value.
    pred_labels : np.ndarray, shape (n, 4)
        Predicted EC numbers, each row is [EC1, EC2, EC3, EC4]. -1 indicates no prediction.

    Returns
    -------
    result_df : pd.DataFrame
        A DataFrame containing the following columns:
        [
          Method, Exact Match Accuracy, First Number Match Accuracy,
          Exact Precision, Exact Recall, Exact F1-Score,
          First Number Precision, First Number Recall, First Number F1-Score,
          No EC number found, No Prediction,
          EC 1, EC 2, EC 3, EC 4, EC 5, EC 6, EC 7
        ]
    """

    n = true_labels.shape[0]
    
    # Flags for different match conditions
    exact_match_flags = []
    first_num_match_flags = []
    no_ec_found_flags = []  # True if the prediction is all -1

    # For accuracy per first number (pred[0])
    first_number_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

    for i in range(n):
        true_ec = true_labels[i]
        pred_ec = pred_labels[i]

        # (a) Check if prediction is entirely -1
        no_ec_found = np.all(pred_ec == -1)
        no_ec_found_flags.append(no_ec_found)

        # (b) Exact match if all four positions match
        exact_match = np.all(pred_ec == true_ec)
        exact_match_flags.append(exact_match)

        # (c) First number match if pred_ec[0] == true_ec[0] and not -1
        first_match = (pred_ec[0] != -1 and pred_ec[0] == true_ec[0])
        first_num_match_flags.append(first_match)

        # (d) Record accuracy for the predicted first number
        if pred_ec[0] != -1:
            fnum_str = str(int(pred_ec[0]))  # convert 1,2,... to '1','2',...
            first_number_accuracy[fnum_str]['total'] += 1
            if first_match:
                first_number_accuracy[fnum_str]['correct'] += 1

    # ---------------- 2. Compute overall metrics ----------------
    exact_match_count = sum(exact_match_flags)
    first_num_match_count = sum(first_num_match_flags)

    accuracy_exact = exact_match_count / n * 100
    accuracy_first = first_num_match_count / n * 100

    # Ratio of rows predicted entirely as -1
    no_ec_found_ratio = sum(no_ec_found_flags) / n

    # “No Prediction” is set to 0.0 here (modify as needed)
    no_prediction_ratio = 0.0

    # ---------------- 3. Accuracy per first number (1~7) ----------------
    ec_accuracies = {}
    for ec_num in range(1, 8):
        key = str(ec_num)
        if first_number_accuracy[key]['total'] > 0:
            acc = (first_number_accuracy[key]['correct'] /
                   first_number_accuracy[key]['total']) * 100
        else:
            acc = 0.0
        ec_accuracies[f"EC {ec_num}"] = acc

    # ---------------- 4. Precision, Recall, F1 ----------------
    # Example: y_true = 1 if there's at least one EC position != -1; else 0
    y_true = [1 if not np.all(t == -1) else 0 for t in true_labels]
    # exact match
    y_pred_exact = [1 if flag else 0 for flag in exact_match_flags]
    # first number match
    y_pred_first = [1 if flag else 0 for flag in first_num_match_flags]

    precision_exact = precision_score(y_true, y_pred_exact, zero_division=0)
    recall_exact = recall_score(y_true, y_pred_exact, zero_division=0)
    f1_exact = f1_score(y_true, y_pred_exact, zero_division=0)

    precision_first = precision_score(y_true, y_pred_first, zero_division=0)
    recall_first = recall_score(y_true, y_pred_first, zero_division=0)
    f1_first = f1_score(y_true, y_pred_first, zero_division=0)

    # ---------------- 5. Assemble result DataFrame ----------------
    result_dict = {
        "Method": ["Physiochemical Features"],
        "Exact Match Accuracy": [accuracy_exact],
        "First Number Match Accuracy": [accuracy_first],
        "Exact Precision": [precision_exact],
        "Exact Recall": [recall_exact],
        "Exact F1-Score": [f1_exact],
        "First Number Precision": [precision_first],
        "First Number Recall": [recall_first],
        "First Number F1-Score": [f1_first],
        "No EC number found": [no_ec_found_ratio],
        "No Prediction": [no_prediction_ratio]
    }

    # Add columns EC 1 ~ EC 7
    for ec_num in range(1, 8):
        col_name = f"EC {ec_num}"
        result_dict[col_name] = [ec_accuracies[col_name]]

    result_df = pd.DataFrame(result_dict)
    return result_df

# if __name__ == "__main__":
#     ec_results_file = "dataset/test_sequences/test_sequences_ec_results.csv"
#     evaluate_ec(ec_results_file)
