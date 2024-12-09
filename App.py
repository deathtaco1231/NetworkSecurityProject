from Model import predict_result, loadmodel
from CreateModel import preprocess, retrain_model
import csv
import numpy as np
import pandas as pd
import os
import shutil
import string
import matplotlib.pyplot as plt

def main():
    experiment_1()
    experiment_2()

def duplicate_file(oldfile, newfile): # Creates the result CSV file, and adds the predicted label column to format it in preparation to have AI model results inserted
    shutil.copyfile(oldfile, newfile)
    csv_input = pd.read_csv(newfile)
    csv_input['Predicted_Label'] = string.whitespace # Just empty filler for now, until these slots are filled in later
    csv_input.to_csv(newfile, index=False)

def write_results_to_result_file(pred, resultfile, resulttxt):
    contents = pd.read_csv(resultfile)
    txtfile = open(resulttxt, "w") # Creating (or overwriting if one exists) the result text file
    normal = 0
    sus = 0
    for i, row in contents.iterrows():
        result = 1 if pred[i] >= 0.5 else 0 # Because our results from the model are long floats, and we only have two options, a 0.5 (50%) certainty of DDoS packet gives us the best possible chances of an accurate result
        contents.at[i, 'Predicted_Label'] = result
        txtfile.write(str(i) + ": " + str(result) + "\n")
        if result == 1:
            sus += 1
        else:
            normal += 1
            
    contents.to_csv(resultfile, index=False) # Writes results to label column of new result csv file
    txtfile.write("Model predicted benign packets: " + str(normal) + " , DDOS packets: " + str(sus))
    txtfile.close()
    return normal, sus

def plot_prediction_dataframe(pred):
    plt.plot(pred)
    plt.title('Model Certainty of DDoS Packet')
    plt.xlabel('Packet')
    plt.ylabel('Certainty')
    plt.legend()
    plt.show()

def experiment_2(): # Each experiment will have its own method, main just needs to be updated to call each but the structure remains the same except for file names/paths
    print("Experiment 2: LOIT DDoS")
    preprocess(r"Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    retrain_model()
    loadmodel()
    testfile = "test1.csv" # Just follow this file structure for each experiment, with test data (unlabeled) going here
    resultfile = "results1.csv" # Name/path to result CSV file (which will be created for you)
    resulttxt = "result1.txt" # Name/path to the result column ONLY (along with information printed to console being logged here), also created for you.
    duplicate_file(testfile, resultfile)
    pred = predict_packet_types_from_test_data(testfile)
    normal, sus = write_results_to_result_file(pred, resultfile, resulttxt)
    print("Expected Benign: " + str(0) + ", DDoS: " + str(2437) + "\nResult Benign: " + str(normal) + ", DDoS: " + str(sus))
    plot_prediction_dataframe(pred)

def experiment_1():
    print("Experiment 1: Portscan")
    preprocess(r"Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    retrain_model()
    loadmodel()
    testfile = "test2.csv" 
    resultfile = "results2.csv" 
    resulttxt = "result2.txt"
    duplicate_file(testfile, resultfile)
    pred = predict_packet_types_from_test_data(testfile)
    normal, sus = write_results_to_result_file(pred, resultfile, resulttxt)
    print("Expected Benign: " + str(13) + ", DDoS: " + str(1978) + "\nResult Benign: " + str(normal) + ", DDoS: " + str(sus))
    plot_prediction_dataframe(pred)

def predict_packet_types_from_test_data(testfile): # Feed model experiment data and obtain result, no longer responsible for writing to result file as it was before
    contents = pd.read_csv(testfile)
    pred = predict_result(contents)
    return pred
    
if __name__ == "__main__":
    main()