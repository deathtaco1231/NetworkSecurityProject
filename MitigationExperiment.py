import pandas as pd
from Model import predict_result
import time

### WARNING: RUNNING THIS TEST TAKES LIKE 6 MINUTES ###
def simulate_real_time_input(file_path):
    """
    Simulates packets being analyzed in real-time.
    """
    df = pd.read_csv(file_path)
    for i, row in df.iterrows():
        yield row

def mitigation_logic(packet, prediction):
    """
    Simulates a mitigation action based on the model's prediction.
    """
    if prediction >= 0.5:  #DDoS detected
        print(f"Packet ID {packet.name}: DDoS detected. Blocking traffic...")
        return "Blocked"
    else:
        print(f"Packet ID {packet.name}: Normal traffic. Allowing...")
        return "Allowed"

def run_mitigation_experiment(test_file, log_file):
    """
    Runs the mitigation experiment.
    """
    print("Starting mitigation experiment...")
    start_time = time.time()

    #Prep logging
    with open(log_file, "w") as log:
        log.write("PacketID,Prediction,Action\n")

        #Simulate real-time input
        for packet in simulate_real_time_input(test_file):
            features = packet.drop(["Label"]).values.reshape(1, -1)  #Extract features
            prediction = predict_result(features)                    #Predict using the model
            action = mitigation_logic(packet, prediction[0])         #Mitigation action
            log.write(f"{packet.name},{prediction[0]},{action}\n")   #Log

    end_time = time.time()
    print(f"Mitigation experiment completed in {end_time - start_time:.2f} seconds.")

#Run experiment
if __name__ == "__main__":
    test_file = "experiment_subsets/subset_b_mixed.csv"  
    log_file = "Mitigation_Results.csv"
    run_mitigation_experiment(test_file, log_file)

