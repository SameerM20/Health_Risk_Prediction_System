import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

heart_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "heart_disease_model.sav"), "rb")
)

def test_heart_prediction_valid():
    sample_input = np.array([[52, 1, 2, 140, 250, 0, 1, 150, 0, 1.2, 2, 0, 2]])
    prediction = heart_model.predict(sample_input)
    
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
