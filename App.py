from Model import predict_result
import csv
import numpy as np
import pandas as pd

def main():
    experiment_1()

def experiment_1():
    testfile = "test1.csv"
    resultfile = "results1.csv"
    predict_packet_types_from_test_data(testfile, resultfile)

def predict_packet_types_from_test_data(testfile, resultfile):
    contents = pd.read_csv(testfile)
    pred = predict_result(contents)
    for i in pred:
        print(i)
    
if __name__ == "__main__":
    main()