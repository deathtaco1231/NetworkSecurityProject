# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

def loadmodel():
    global model
    model = load_model("ddos_detection_model.h5")

def predict_result(line):
    pred = model.predict(line)
    return pred