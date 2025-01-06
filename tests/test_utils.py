from random import choice
import pytest
from mypadi.core.constants import LANGUAGES
from mypadi.core.utils import load_file


def test_load_language_detection_dataset():
    dataset = load_file(f"language_detection/{choice(LANGUAGES)}.json")
    assert len(dataset) > 0
    assert "text" in dataset[0]
    assert "label" in dataset[0]


def test_load_intent_recognition_dataset():
    dataset = load_file(f"intent_recognition/{choice(LANGUAGES)}.json")
    assert len(dataset) > 0
    assert "text" in dataset[0]
    assert "intent" in dataset[0]
    assert "language" in dataset[0]


def test_load_nonexistent_dataset():
    with pytest.raises(FileNotFoundError) as excinfo:
        load_file("nonexistent.json")
    assert "nonexistent.json" in str(excinfo.value)
