import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#File paths
log_file = "Mitigation_Results.csv"                  #File created during the mitigation test
test_file = "experiment_subsets/subset_b_mixed.csv"  #Original dataset with true labels

#Load the datasets
log_data = pd.read_csv(log_file)
test_data = pd.read_csv(test_file)

#Ensure correct columns are present
if 'Label' not in test_data.columns:
    raise ValueError("The test dataset must contain a 'Label' column.")

if 'Action' not in log_data.columns:
    raise ValueError("The log file must contain an 'Action' column.")

#Map 'Action' to binary (0 = Allowed, 1 = Blocked)
log_data['Predicted_Label'] = log_data['Action'].apply(lambda x: 1 if x == 'Blocked' else 0)

#Compare predictions with actual labels
y_true = test_data['Label']
y_pred = log_data['Predicted_Label']

#Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

#Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Benign", "DDoS"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

#Visualize actions
action_counts = log_data['Action'].value_counts()
action_counts.plot(kind='bar', color=['green', 'red'], title='Mitigation Actions')
plt.xlabel("Action")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()
