from mypadi.core.language import detect_language


def test_detect_language():
    text = "How far?"
    language = detect_language(text)
    assert language == "pidgin"
