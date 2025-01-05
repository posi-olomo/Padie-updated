from gistai.core.intent import detect_intent


def test_detect_intent():
    text = "Wetin dey happen?"
    intent = detect_intent(text)
    assert intent == "greeting"
