import pytest
from gistai.core.utils import load_dataset


def test_load_language_detection_dataset():
    dataset = load_dataset("language_detection.json")
    assert len(dataset) > 0
    assert "text" in dataset[0]
    assert "label" in dataset[0]


def test_load_intent_recognition_dataset():
    dataset = load_dataset("intent_recognition.json")
    assert len(dataset) > 0
    assert "text" in dataset[0]
    assert "intent" in dataset[0]
    assert "language" in dataset[0]


def test_load_response_generation_dataset():
    dataset = load_dataset("response_generation.json")
    assert len(dataset) > 0
    assert "input" in dataset[0]
    assert "response" in dataset[0]


def test_load_nonexistent_dataset():
    with pytest.raises(FileNotFoundError) as excinfo:
        load_dataset("nonexistent.json")
    assert "nonexistent.json" in str(excinfo.value)
