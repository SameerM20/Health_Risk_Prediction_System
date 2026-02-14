import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_dataset_files_exist():
    datasets = ["diabetes.csv", "heart.csv", "parkinsons.csv"]
    
    for file in datasets:
        path = os.path.join(BASE_DIR, "dataset", file)
        assert os.path.exists(path), f"{file} missing!"
