from gistai.core.response import generate_response


def test_generate_response():
    intent = "greeting"
    language = "pidgin"
    response = generate_response(intent, language)
    assert (
        response == "Response for intent 'greeting' in language 'pidgin'"
    )  # Adjust based on actual logic
