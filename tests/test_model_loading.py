import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(model_name):
    model_path = os.path.join(BASE_DIR, "saved_models", model_name)
    assert os.path.exists(model_path), f"{model_name} not found!"
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    assert model is not None, f"{model_name} failed to load!"
    return model

def test_diabetes_model_loading():
    load_model("diabetes_model.sav")

def test_heart_model_loading():
    load_model("heart_disease_model.sav")

def test_parkinsons_model_loading():
    load_model("parkinsons_model.sav")
