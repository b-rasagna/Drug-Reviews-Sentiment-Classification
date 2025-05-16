import re

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources if not already present
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Globals
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(tag: str) -> str:
    """
    Convert POS tag from nltk to WordNet format.

    Parameters:
    - tag (str): POS tag from nltk.pos_tag

    Returns:
    - str: Corresponding WordNet POS tag
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # Default to noun


def clean_text(text: str) -> str:
    """
    Cleans input text using the following pipeline:
    - Lowercasing
    - HTML tag removal
    - Non-alphabetic character removal
    - Stopword removal
    - Lemmatization with POS tagging

    Parameters:
    - text (str): Raw user input

    Returns:
    - str: Cleaned and lemmatized string
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase the text
    text = text.lower()

    # Remove HTML tags and non-alphabetic characters
    text = re.sub(r"<.*?>", " ", text)  # Remove HTML
    text = re.sub(r"[^a-z\s]", "", text)  # Keep only letters and whitespace

    # Tokenization and stopword removal
    tokens = [t for t in word_tokenize(text) if t not in stop_words]

    # POS tagging and lemmatization
    tagged_tokens = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(
            word,
            get_wordnet_pos(tag)) for word,
        tag in tagged_tokens]

    return " ".join(lemmatized)
