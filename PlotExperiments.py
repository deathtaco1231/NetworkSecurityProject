import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#Data for charts
results = pd.DataFrame({
    "experiment": ["Subset_A_Normal", "Subset_B_Mixed", "Subset_C_DDoS", "Subset_D_Edge"],
    "accuracy": [0.999682761, 0.9412, 0.888726597, 0.929763874],
    "precision": [0, 0.999773448, 1, 0.999595235],
    "recall": [0, 0.8826, 0.888726597, 0.873688059],
    "f1_score": [0, 0.937539834, 0.941085489, 0.932410408]
})

#Prepare data 
metrics = ["accuracy", "precision", "recall", "f1_score"]
values = results[metrics].values
categories = metrics

#Create a radar chart for each experiment
for i, experiment in enumerate(results["experiment"]):
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    #Radar plot values
    values_exp = values[i].tolist()
    values_exp += values_exp[:1]  

    ax.plot(angles, values_exp, color='skyblue', linewidth=2, label=experiment)
    ax.fill(angles, values_exp, color='skyblue', alpha=0.25)

    #Add labels
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    plt.title(f"Performance Metrics for {experiment}", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


    #Data for the heatmap
    heatmap_data = results.set_index("experiment").T

    #Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt=".3f", linewidths=0.5)
    plt.title("Performance Metrics Across Experiments", fontsize=14)
    plt.xlabel("Experiment", fontsize=12)
    plt.ylabel("Metrics", fontsize=12)
    plt.tight_layout()
    plt.show()

    #Data for the line plot
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(results["experiment"], results[metric], marker="o", label=metric)

    plt.title("Performance Metrics Across Experiments", fontsize=14)
    plt.xlabel("Experiment", fontsize=12)
    plt.ylabel("Metric Values", fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


    #Data for the stacked bar chart
    bar_width = 0.5
    indices = np.arange(len(results["experiment"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(results["experiment"]))

    for metric in metrics:
        ax.bar(indices, results[metric], bar_width, label=metric, bottom=bottoms, alpha=0.7)
        bottoms += results[metric]

    ax.set_xticks(indices)
    ax.set_xticklabels(results["experiment"], rotation=45)
    ax.set_ylim(0, 4)
    plt.title("Stacked Performance Metrics Across Experiments", fontsize=14)
    plt.xlabel("Experiment", fontsize=12)
    plt.ylabel("Cumulative Metric Score", fontsize=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()
