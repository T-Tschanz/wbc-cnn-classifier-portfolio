
import numpy as np
import tensorflow as tf

def classify_image(model, image_tensor):
    prediction = model.predict(image_tensor, verbose=0)[0]
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index])
    return class_index, confidence, prediction
