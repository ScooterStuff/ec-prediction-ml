import pandas as pd
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

    cols = [
        f"EC {num}" if num.isdigit() else num
        for num in first_num_acc["first_num"]
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
        "First Number F1-Score": [f1_first]
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

# if __name__ == "__main__":
#     ec_results_file = "dataset/test_sequences/test_sequences_ec_results.csv"
#     evaluate_ec(ec_results_file)
