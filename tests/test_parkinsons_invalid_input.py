import os
import pickle
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parkinsons_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "parkinsons_model.sav"), "rb")
)

def test_parkinsons_invalid_input():
    invalid_input = np.array([[1, 2, 3]])  # ❌ Wrong feature count

    with pytest.raises(ValueError):
        parkinsons_model.predict(invalid_input)
