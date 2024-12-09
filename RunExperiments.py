import pandas as pd
import numpy as np
from Model import predict_result
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os

def run_experiment(subset_path, experiment_name, output_dir):
    """
    Runs a single experiment on the given dataset subset.
    
    Args:
        subset_path (str): Path to the subset CSV file.
        experiment_name (str): Name of the experiment.
        output_dir (str): Directory to save experiment results.
    """
    #Load the subset
    subset = pd.read_csv(subset_path)
    features = subset.drop(columns=['Label'])  # Features
    true_labels = subset['Label']  # True labels
    
    #Predict using the model
    predictions = predict_result(features)
    predicted_labels = (predictions >= 0.5).astype(int)  #Convert probabilities to binary labels
    
    #Evaluate performance
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    
    #Generate classification report
    report = classification_report(true_labels, predicted_labels, zero_division=0)
    
    #Save results
    results_path = os.path.join(output_dir, f"{experiment_name}_results.txt")
    with open(results_path, "w") as result_file:
        result_file.write(f"Experiment: {experiment_name}\n")
        result_file.write(f"Accuracy: {accuracy:.4f}\n")
        result_file.write(f"Precision: {precision:.4f}\n")
        result_file.write(f"Recall: {recall:.4f}\n")
        result_file.write(f"F1-Score: {f1:.4f}\n\n")
        result_file.write("Classification Report:\n")
        result_file.write(report)
    
    print(f"Experiment {experiment_name} completed. Results saved to {results_path}.")
    
    return {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def run_all_experiments(subsets_dir, output_dir):
    """
    Runs all experiments on the subsets generated.
    
    Args:
        subsets_dir (str): Directory containing subset CSV files.
        output_dir (str): Directory to save all experiment results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = []
    subset_files = {
        "Subset_A_Normal": "subset_a_normal.csv",
        "Subset_B_Mixed": "subset_b_mixed.csv",
        "Subset_C_DDoS": "subset_c_ddos.csv",
        "Subset_D_Edge": "subset_d_edge_cases.csv"
    }
    
    for experiment_name, subset_file in subset_files.items():
        subset_path = os.path.join(subsets_dir, subset_file)
        if os.path.exists(subset_path):
            result = run_experiment(subset_path, experiment_name, output_dir)
            results.append(result)
        else:
            print(f"Subset {subset_file} not found. Skipping {experiment_name}.")
    
    #Generate summary results
    summary_path = os.path.join(output_dir, "summary_results.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"All experiments completed. Summary saved to {summary_path}.")

if __name__ == "__main__":
    subsets_dir = "experiment_subsets"  #Directory containing subsets
    output_dir = "experiment_results"   #Directory to save results
    run_all_experiments(subsets_dir, output_dir)
