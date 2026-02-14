import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_app_file_exists():
    app_path = os.path.join(BASE_DIR, "app.py")
    assert os.path.exists(app_path), "app.py not found!"
