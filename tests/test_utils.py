import pytest
from gistai.core.utils import load_dataset


def test_load_dataset():
    dataset = load_dataset("language_detection.json")
    assert len(dataset) > 0
    assert "text" in dataset[0]
    assert "label" in dataset[0]


def test_load_nonexistent_dataset():
    with pytest.raises(FileNotFoundError) as excinfo:
        load_dataset("nonexistent.json")
    assert "nonexistent.json" in str(excinfo.value)
