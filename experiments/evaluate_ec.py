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
        
        true_ec_split = [ec.split('.') for ec in true_ecs]
        pred_ec_split = pred_ec.split('.') if pred_ec != 'No Prediction' else []
        
        # Pad EC numbers to ensure they have four parts
        true_ec_split = [ec + ['-'] * (4 - len(ec)) for ec in true_ec_split]
        pred_ec_split += ['-'] * (4 - len(pred_ec_split))
        
        # Compare each position
        match_flags = []
        for i in range(4):
            position_matches = [ec[i] == pred_ec_split[i] for ec in true_ec_split if ec[i] != '-']
            match = any(position_matches)
            match_flags.append(match)
        
        exact_match = all(match_flags)
        no_ec_found = pred_ec == 'No Prediction' or pred_ec == 'No EC number found'
        
        results.append({
            'Query': query,
            'True EC': true_ecs,
            'Predicted EC': pred_ec,
            'Exact Match': exact_match,
            'No EC number found': no_ec_found,
            'Position Matches': match_flags
        })
        
    return pd.DataFrame(results)

def compute_metrics(results, method_name):
    """Computes metrics and returns a DataFrame similar to evaluate_ec_predictions."""
    n_samples = len(results)
    n_positions = 4  # Four parts in EC numbers
    
    exact_match_flags = results['Exact Match'].tolist()
    no_ec_found_flags = results['No EC number found'].tolist()
    
    position_match_flags = [[] for _ in range(n_positions)]
    for flags in results['Position Matches']:
        for i, match in enumerate(flags):
            position_match_flags[i].append(match)
    
    exact_match_accuracy = sum(exact_match_flags) / n_samples * 100
    no_ec_found_ratio = sum(no_ec_found_flags) / n_samples * 100
    no_prediction_ratio = no_ec_found_ratio  # Assuming 'No Prediction' and 'No EC number found' are equivalent
    
    position_accuracies = []
    position_precisions = []
    position_recalls = []
    position_f1_scores = []
    
    for pos in range(n_positions):
        y_true = []
        y_pred = []
        for idx, row in results.iterrows():
            true_ecs = row['True EC']
            pred_ec = row['Predicted EC']
            if pred_ec in ["No EC number found", "No Prediction"]:
                pred_part = '-'
            else:
                pred_split = pred_ec.split('.')
                pred_part = pred_split[pos] if pos < len(pred_split) else '-'
            for true_ec in true_ecs:
                true_split = true_ec.split('.')
                true_part = true_split[pos] if pos < len(true_split) else '-'
                if true_part != '-':
                    y_true.append(true_part)
                    y_pred.append(pred_part)
        
        accuracy = sum(position_match_flags[pos]) / n_samples * 100
        position_accuracies.append(accuracy)
        
        if y_true:
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            precision = recall = f1 = 0.0
        
        position_precisions.append(precision * 100)
        position_recalls.append(recall * 100)
        position_f1_scores.append(f1 * 100)
    
    result_dict = {
        "Method": [method_name],
        "Exact Match Accuracy": [exact_match_accuracy],
        "No EC number found": [no_ec_found_ratio],
        "No Prediction": [no_prediction_ratio],
    }
    
    for pos in range(n_positions):
        pos_idx = pos + 1
        result_dict[f"Position {pos_idx} Accuracy"] = [position_accuracies[pos]]
        result_dict[f"Position {pos_idx} Precision"] = [position_precisions[pos]]
        result_dict[f"Position {pos_idx} Recall"] = [position_recalls[pos]]
        result_dict[f"Position {pos_idx} F1-Score"] = [position_f1_scores[pos]]
    
    result_df = pd.DataFrame(result_dict)
    return result_df

def evaluate_ec(ec_results_file, evaluate_file, ec40_file, method_name):
    '''
    ec_results_file: EC numbers predicted by the model
    evaluate_file: Path to save detailed evaluation results
    ec40_file: EC40 file with true EC numbers
    method_name: Name of the method for reporting
    '''
    
    print("Loading data...")
    ec40, ec_results = load_data(ec40_file, ec_results_file)
    
    print("Matching predictions...")
    results = match_predictions(ec40, ec_results)
    
    print("Computing evaluation metrics...")
    metrics = compute_metrics(results, method_name)
    
    results.to_csv(evaluate_file, index=False)
    print(f"Saved results to and {evaluate_file}")
    
    return metrics



def evaluate_ec_predictions(true_labels: np.ndarray, pred_labels: np.ndarray, method_name: str) -> pd.DataFrame:
    n_samples = true_labels.shape[0]
    n_positions = true_labels.shape[1]

    exact_match_flags = []
    no_ec_found_flags = []

    position_match_flags = [[] for _ in range(n_positions)]

    for i in range(n_samples):
        true_ec = true_labels[i]
        pred_ec = pred_labels[i]

        no_ec_found = np.all(pred_ec == -1)
        no_ec_found_flags.append(no_ec_found)

        exact_match = np.all(pred_ec == true_ec)
        exact_match_flags.append(exact_match)

        for pos in range(n_positions):
            true_val = true_ec[pos]
            pred_val = pred_ec[pos]

            if true_val != -1:
                if pred_val != -1 and pred_val == true_val:
                    position_match_flags[pos].append(True)
                else:
                    position_match_flags[pos].append(False)
            else:
                position_match_flags[pos].append(False)

    exact_match_accuracy = sum(exact_match_flags) / n_samples * 100

    no_ec_found_ratio = sum(no_ec_found_flags) / n_samples * 100

    no_prediction_ratio = no_ec_found_ratio

    position_accuracies = []
    position_precisions = []
    position_recalls = []
    position_f1_scores = []

    for pos in range(n_positions):
        match_flags = position_match_flags[pos]

        valid_indices = true_labels[:, pos] != -1
        y_true = true_labels[valid_indices, pos]
        y_pred = pred_labels[valid_indices, pos]

        y_pred_processed = np.where(y_pred != -1, y_pred, -99)

        accuracy = sum(match_flags) / n_samples * 100
        position_accuracies.append(accuracy)

        if len(y_true) > 0:
            precision = precision_score(y_true, y_pred_processed, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred_processed, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred_processed, average='macro', zero_division=0)
        else:
            precision = recall = f1 = 0.0

        position_precisions.append(precision)
        position_recalls.append(recall)
        position_f1_scores.append(f1)

    result_dict = {
        "Method": [method_name],
        "Exact Match Accuracy": [exact_match_accuracy],
        "No EC number found": [no_ec_found_ratio],
        "No Prediction": [no_prediction_ratio],
    }

    for pos in range(n_positions):
        pos_idx = pos + 1
        result_dict[f"Position {pos_idx} Accuracy"] = [position_accuracies[pos]]
        result_dict[f"Position {pos_idx} Precision"] = [position_precisions[pos]]
        result_dict[f"Position {pos_idx} Recall"] = [position_recalls[pos]]
        result_dict[f"Position {pos_idx} F1-Score"] = [position_f1_scores[pos]]

    result_df = pd.DataFrame(result_dict)
    return result_df

# if __name__ == "__main__":
#     ec_results_file = "dataset/test_sequences/test_sequences_ec_results.csv"
#     evaluate_ec(ec_results_file)
