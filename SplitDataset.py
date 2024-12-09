import pandas as pd
import numpy as np
import os

def create_subsets(dataset_path, output_dir):
    """
    Splits the dataset into subsets A, B, C, and D for experiments.
    
    Subset A: Normal traffic only.
    Subset B: Mixed normal and DDoS traffic in varying proportions.
    Subset C: Only DDoS traffic.
    Subset D: Rare cases (e.g., very small or large flows).
    
    Args:
        dataset_path (str): Path to the preprocessed dataset.
        output_dir (str): Directory to save the subsets.
    """
    #Load the dataset
    df = pd.read_csv(dataset_path)
    
    #Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #Subset A: Normal traffic only
    subset_a = df[df['Label'] == 0]
    subset_a_path = os.path.join(output_dir, "subset_a_normal.csv")
    subset_a.to_csv(subset_a_path, index=False)
    print(f"Subset A (Normal traffic only) saved to {subset_a_path}.")
    
    #Subset B: Mixed normal and DDoS traffic
    normal_samples = df[df['Label'] == 0].sample(n=5000, random_state=42)  # 5000 normal samples
    ddos_samples = df[df['Label'] == 1].sample(n=5000, random_state=42)    # 5000 DDoS samples
    subset_b = pd.concat([normal_samples, ddos_samples]).sample(frac=1, random_state=42)  # Shuffle
    subset_b_path = os.path.join(output_dir, "subset_b_mixed.csv")
    subset_b.to_csv(subset_b_path, index=False)
    print(f"Subset B (Mixed normal and DDoS traffic) saved to {subset_b_path}.")
    
    #Subset C: Only DDoS traffic
    subset_c = df[df['Label'] == 1]
    subset_c_path = os.path.join(output_dir, "subset_c_ddos.csv")
    subset_c.to_csv(subset_c_path, index=False)
    print(f"Subset C (Only DDoS traffic) saved to {subset_c_path}.")
    
    #Subset D: Rare cases or edge scenarios
    #Example: Very small flows (Flow Duration < 1000) or very large flows (Flow Duration > 1e6)
    subset_d = df[(df['Flow Duration'] < 1000) | (df['Flow Duration'] > 1e6)]
    subset_d_path = os.path.join(output_dir, "subset_d_edge_cases.csv")
    subset_d.to_csv(subset_d_path, index=False)
    print(f"Subset D (Rare cases or edge scenarios) saved to {subset_d_path}.")

if __name__ == "__main__":
    dataset_path = "preprocessed_dataset.csv"  #Path to your preprocessed dataset
    output_dir = "experiment_subsets"          #Directory to save subsets
    create_subsets(dataset_path, output_dir)
