from gistai.core.intent import detect_intent


def test_detect_intent():
    text = "I need help"
    intent = detect_intent(text)
    assert intent == "greeting"  # Adjust based on actual function implementation
