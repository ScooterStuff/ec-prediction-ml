import os
import pickle
import re
from svm_training import SVCWrapper
from tqdm import tqdm
import concurrent.futures


def aggregate_cascade_models(cascade_dir, output_file="Models/all_cascade_models_SVM.pkl"):
    """
    Consolidate step-by-step models in the cascade directory into a single list file

    :param cascade_dir: Directory storing models (e.g., Models/SVM_cascade_steps)
    :param output_file: Output file path (e.g., Models/all_cascade_models_SVM.pkl)
    """
    pattern = re.compile(r'^cascade_step_(\d+)_model\.pkl$')
    # Only include files that fully match the pattern
    model_files = [f for f in os.listdir(cascade_dir) if pattern.match(f)]
    # Sort the files numerically based on the captured number
    model_files = sorted(model_files, key=lambda x: int(pattern.match(x).group(1)))
    
    all_models = []
    for model_file in model_files:
        with open(os.path.join(cascade_dir, model_file), "rb") as f:
            all_models.append(pickle.load(f))
    
    # Ensure the directory for the output exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the consolidated model list
    with open(output_file, "wb") as f:
        pickle.dump(all_models, f)
    print(f"Saved aggregated models to {output_file} (count={len(all_models)})")

# Example usage
aggregate_cascade_models(
    cascade_dir="Models/SVM_cascade_steps",
    output_file="Models/all_cascade_models_SVM.pkl"
)
