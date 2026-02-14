import os
import pickle
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

heart_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "heart_disease_model.sav"), "rb")
)

def test_heart_invalid_input():
    invalid_input = np.array([[1, 2, 3]])  # ❌ Wrong feature count

    with pytest.raises(ValueError):
        heart_model.predict(invalid_input)
