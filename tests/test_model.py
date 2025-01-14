import torch
from transformers import pipeline
from collections import defaultdict

from train_language_detection import MODEL_OUTPUT_DIR


def load_trained_model(model_path=MODEL_OUTPUT_DIR):
    """
    Loads the trained model and tokenizer from the specified path.

    Args:
        model_path (str): Path to the saved model and tokenizer.

    Returns:
        Pipeline: Hugging Face pipeline for text classification.
    """
    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        # Change device selection if you want to use CUDA when available:
        device=0 if torch.cuda.is_available() else -1,
    )
    return classifier


def test_model(test_dataset):
    """
    Loads the trained model and tests it on a provided test dataset.

    Each example in test_dataset should be a dictionary with keys:
      - "text": the text input to classify.
      - "expected_label": the expected label (string) for that input.

    The function will print the prediction and if it was correct, and compute overall accuracy.
    """
    # Load classifier using the helper function
    classifier = load_trained_model()

    total_samples = len(test_dataset)
    correct_predictions = 0

    # For detailed metrics per label, we can use a dictionary.
    detailed_results = defaultdict(lambda: {"correct": 0, "total": 0})

    print("Starting testing on {} samples...\n".format(total_samples))
    for i, sample in enumerate(test_dataset):
        text = sample["text"]
        expected = sample["expected_label"]

        # The pipeline returns a list of dictionaries. The first result is the most likely.
        prediction = classifier(text)[0]["label"].lower()
        # The model might return the label with capital letters or with some prefixes
        # (like "LABEL_0" if it wasn't properly mapped); adjust as needed.

        # Check prediction against expected
        is_correct = prediction == expected.lower()
        correct_predictions += int(is_correct)
        detailed_results[expected.lower()]["total"] += 1
        if is_correct:
            detailed_results[expected.lower()]["correct"] += 1

        print(f"Sample {i+1}:")
        print(f"Text: {text}")
        print(f"Expected: {expected}")
        print(f"Predicted: {prediction}")
        print(f"Correct? {'Yes' if is_correct else 'No'}\n")

    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_samples * 100

    print("Testing Complete!")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%\n")

    # Show detailed results per label
    print("Detailed Results:")
    for label, stats in detailed_results.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(
            f"  {label.capitalize()}: {stats['correct']}/{stats['total']} ({acc:.2f}%)"
        )
