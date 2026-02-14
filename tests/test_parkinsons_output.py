import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parkinsons_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "parkinsons_model.sav"), "rb")
)

def test_parkinsons_prediction_output():
    sample_input = np.random.rand(1, 22)  # ✅ Assuming 22 features
    prediction = parkinsons_model.predict(sample_input)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
    assert isinstance(prediction[0], (int, np.integer))
