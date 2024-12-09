from Model import predict_result
import csv
import numpy as np
import pandas as pd
import os
import shutil
import string

def main():
    experiment_1()

def duplicate_file(oldfile, newfile):
    shutil.copyfile(oldfile, newfile)
    csv_input = pd.read_csv(newfile)
    csv_input['Predicted_Label'] = string.whitespace
    csv_input.to_csv(newfile, index=False)
    #f = open(newfile, "w")
    #with open(oldfile, "r") as f:
        #reader = csv.reader(f, delimiter="\t")
        #for i, line in enumerate(reader):
            #f.write(line)
    #f.close()

def write_results_to_result_file(pred, resultfile, resulttxt):
    contents = pd.read_csv(resultfile)
    txtfile = open(resulttxt, "w")
    normal = 0
    sus = 0
    for i, row in contents.iterrows():
        result = 1 if pred[i] >= 0.5 else 0
        contents.at[i, 'Predicted_Label'] = result
        txtfile.write(str(i) + ": " + str(result) + "\n")
        if result == 1:
            sus += 1
        else:
            normal += 1
            
    contents.to_csv(resultfile, index=False)  
    txtfile.write("Model predicted benign packets: " + str(normal) + " , DDOS packets: " + str(sus))
    txtfile.close()
    return normal, sus

def experiment_1():
    print("Experiment 1: All packets DDoS (port 80 brute force attack)")
    testfile = "test1.csv"
    resultfile = "results1.csv"
    resulttxt = "result1.txt"
    duplicate_file(testfile, resultfile)
    pred = predict_packet_types_from_test_data(testfile, resultfile)
    normal, sus = write_results_to_result_file(pred, resultfile, resulttxt)
    for i in pred:
        print(i)

    print("Benign: " + str(normal) + ", DDoS: " + str(sus))

def predict_packet_types_from_test_data(testfile, resultfile):
    contents = pd.read_csv(testfile)
    pred = predict_result(contents)
    return pred
    
if __name__ == "__main__":
    main()