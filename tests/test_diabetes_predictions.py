import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

diabetes_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "diabetes_model.sav"), "rb")
)

def test_diabetes_prediction_valid():
    sample_input = np.array([[2, 120, 70, 20, 79, 25.0, 0.5, 33]])
    prediction = diabetes_model.predict(sample_input)
    
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
