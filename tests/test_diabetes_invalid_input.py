import os
import pickle
import numpy as np
import pytest

#  BASE_DIR
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

diabetes_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "diabetes_model.sav"), "rb")
)

def test_diabetes_invalid_input():
    invalid_input = np.array([[1, 2, 3]])  # Wrong feature count

    with pytest.raises(ValueError):
        diabetes_model.predict(invalid_input)
