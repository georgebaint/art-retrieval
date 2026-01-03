from backend.src import api_calls
import os 

def test_fetch_and_save_wrapper(tmp_path):
    """Wrapper so pytest discovers and runs the API call test in `api_calls.py`."""
    api_calls.test_fetch_and_save(tmp_path)


if __name__ == "__main__":
    tmp_path = "temp_test_output"
    os.makedirs(tmp_path, exist_ok=True)
    test_fetch_and_save_wrapper(tmp_path)
