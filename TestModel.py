from keras.models import load_model

#Test if model loads in correctly or has to be created


try:
    model = load_model("ddos_detection_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
