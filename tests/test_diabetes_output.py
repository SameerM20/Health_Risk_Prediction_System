import os
import pickle
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#  LOAD DIABETES MODEL 
diabetes_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "diabetes_model.sav"), "rb")
)


def test_prediction_output_type():
    sample_input = np.array([[2, 120, 70, 20, 79, 25.0, 0.5, 33]])
    prediction = diabetes_model.predict(sample_input)
    
    assert isinstance(prediction[0], (int, np.integer))

