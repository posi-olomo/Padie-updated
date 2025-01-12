from collections import Counter
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

from padie.core.constants import LANGUAGES


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


def load_and_inspect_dataset(data_path, key_name, merge_notonal=False):
    file_path = Path(__file__).resolve().parent.parent.parent / "datasets" / data_path

    # Initialize an empty list to collect datasets
    datasets_list = []

    # Iterate over languages and load datasets
    for language in LANGUAGES:
        if language in ["yoruba", "igbo"] and merge_notonal:
            # Load both tonal and notonal datasets
            tonal_dataset = load_dataset(
                "json", data_files={"tonal": f"{file_path}/{language}.json"}
            )["tonal"]

            notonal_dataset = load_dataset(
                "json", data_files={"notonal": f"{file_path}/{language}_notonal.json"}
            )["notonal"]

            # Take 90% from tonal and 10% from notonal
            tonal_sample_size = int(len(tonal_dataset) * 0.9)
            notonal_sample_size = int(len(notonal_dataset) * 0.1)

            tonal_sample = tonal_dataset.shuffle(seed=42).select(
                range(min(tonal_sample_size, 9000))
            )
            notonal_sample = notonal_dataset.shuffle(seed=42).select(
                range(min(notonal_sample_size, 1000))
            )

            # Combine tonal and notonal samples
            combined_language_dataset = concatenate_datasets(
                [tonal_sample, notonal_sample]
            )
        else:
            # Load dataset for other languages
            combined_language_dataset = load_dataset(
                "json", data_files={language: f"{file_path}/{language}.json"}
            )[language]

            # Limit to a maximum of 10k samples per language
            combined_language_dataset = combined_language_dataset.shuffle(
                seed=42
            ).select(range(min(len(combined_language_dataset), 10000)))

        datasets_list.append(combined_language_dataset)  # Append to the list

    # Combine all datasets into one
    combined_dataset = concatenate_datasets(datasets_list)

    # Shuffle the combined dataset
    shuffled_combined_dataset = combined_dataset.shuffle(seed=42)

    print(f"Total samples: {len(shuffled_combined_dataset)}")
    counts = Counter(shuffled_combined_dataset[key_name])

    print(f"{key_name.title()} distribution:")
    for intent, count in counts.items():
        print(f"  {intent}: {count}")

    split_dataset = shuffled_combined_dataset.train_test_split(
        test_size=0.2,
        # stratify_by_column=key_name,
        seed=42,
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    return train_dataset, eval_dataset
