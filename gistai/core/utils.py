from collections import Counter
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

from gistai.core.constants import LANGUAGES


def load_file(file_name):
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


def load_and_inspect_dataset(data_path, key_name):
    file_path = Path(__file__).resolve().parent.parent.parent / "datasets" / data_path

    # Initialize an empty list to collect datasets
    datasets_list = []

    # Iterate over languages and load datasets
    for language in LANGUAGES:
        # Load dataset for the current language
        dataset = load_dataset(
            "json", data_files={language: f"{file_path}/{language}.json"}
        )[language]
        datasets_list.append(dataset)  # Append to the list

    # Combine all datasets into one
    combined_dataset = concatenate_datasets(datasets_list)

    # Shuffle the combined dataset
    shuffled_combined_dataset = combined_dataset.shuffle(seed=42)

    print(f"Total samples: {len(shuffled_combined_dataset)}")
    counts = Counter(shuffled_combined_dataset[key_name])

    print(f"{key_name.title()} distribution:")
    for intent, count in counts.items():
        print(f"  {intent}: {count}")

    return shuffled_combined_dataset
