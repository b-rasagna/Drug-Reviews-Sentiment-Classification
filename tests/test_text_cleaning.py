from app.text_cleaning import clean_text


def test_clean_text_removes_html():
    raw = "This <b>drug</b> works!"
    cleaned = clean_text(raw)
    assert "drug" in cleaned and "<b>" not in cleaned


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""


def test_clean_text_lowercases_text():
    assert "good" in clean_text("GOOD MEDICINE")
