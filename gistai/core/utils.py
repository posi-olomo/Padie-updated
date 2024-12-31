import json
from pathlib import Path


def load_dataset(file_name):
    """
    Load a dataset from the datasets folder.

    Args:
        file_name (str): Name of the dataset file (e.g., 'language_detection.json').

    Returns:
        list: List of dataset entries.
    """
    # Adjust the path to the root directory
    file_path = Path(__file__).resolve().parent.parent.parent / "datasets" / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file {file_name} not found at {file_path}")

    with open(file_path, "r") as file:
        return json.load(file)
