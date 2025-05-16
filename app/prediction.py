import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.text_cleaning import clean_text


def predict_sentiment(review_text: str, model_objects: dict) -> dict:
    """
    Predicts the sentiment of a given drug review using a loaded model.

    Parameters:
    - review_text (str): The raw review text input from the user.
    - model_objects (dict): A dictionary containing the loaded model,
    tokenizer/vectorizer, encoder, and type.

    Returns:
    - dict: Contains the predicted sentiment label.
    """

    if not review_text or not isinstance(review_text, str):
        return {"error": "Invalid input. Please provide a non-empty string."}

    # Step 1: Clean the input text
    cleaned_text = clean_text(review_text)

    # Step 2: ML pipeline
    if model_objects["type"] == "ml":
        # TF-IDF vectorization
        X = model_objects["vectorizer"].transform([cleaned_text])
        y_pred = model_objects["model"].predict(X)

    # Step 3: DL pipeline
    elif model_objects["type"] == "dl":
        tokenizer = model_objects["tokenizer"]
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding="post")
        probabilities = model_objects["model"].predict(padded_sequence)
        y_pred = np.argmax(probabilities, axis=1)

    else:
        return {"error": "Model type is not set or invalid."}

    # Step 4: Inverse transform to label
    label = model_objects["encoder"].inverse_transform(y_pred)[0]
    return {"prediction": label}
