import os
import pickle

import joblib
from tensorflow.keras.models import load_model


def load_model_by_name(model_name: str, model_objects: dict):
    """
    Loads a machine learning or deep learning model based on the provided name.

    Parameters:
    - model_name (str): UI-displayed model name (mapped to filename)
    - model_objects (dict): In-memory store to update with the loaded
                            model and processors

    Raises:
    - ValueError: If an unknown model name is passed
    - FileNotFoundError: If the expected model file does not exist
    """

    # Reset model state before loading
    model_objects.update(
        {
            "model": None,
            "vectorizer": None,
            "tokenizer": None,
            "encoder": None,
            "type": None,
        }
    )

    # Mapping from UI name to actual saved file
    model_file_map = {
        "Logistic Regression (Imbalanced)":
        "logistic_regression_model_imbal.pkl",
        "Random Forest (Imbalanced)":
        "random_forest_model_imbal.pkl",
        "Linear SVC (Imbalanced)":
        "linear_svc_model_imbal.pkl",
        "LightGBM (Imbalanced)":
        "lightgbm_model_imbal.pkl",
        "Logistic Regression (SMOTE)":
        "logistic_regression_model_bal.pkl",
        "Random Forest (SMOTE)":
        "random_forest_model_bal.pkl",
        "Linear SVC (SMOTE)":
        "linear_svc_model_bal.pkl",
        "LightGBM (SMOTE)":
        "lightgbm_model_bal.pkl",
        "Bi-LSTM + CNN (Imbalanced)":
        "bilstm_cnn_model_imbalanced.h5",
        "Bi-LSTM + CNN (Class Weights)":
        "bilstm_cnn_model_balanced.h5",
    }

    # Lookup model file
    model_file = model_file_map.get(model_name)
    if not model_file:
        raise ValueError(f"Unknown model name: '{model_name}'")

    file_path = os.path.join("models", model_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at path: {file_path}")

    # Load Deep Learning Model
    if model_name.startswith("Bi-LSTM"):
        model_objects["model"] = load_model(file_path)
        with open("models/tokenizer.pkl", "rb") as f:
            model_objects["tokenizer"] = pickle.load(f)
        model_objects["type"] = "dl"

    # Load Traditional ML Model
    else:
        model_objects["model"] = joblib.load(file_path)
        model_objects["vectorizer"] = joblib.load(
            "models/tfidf_vectorizer.pkl")
        model_objects["type"] = "ml"

    # Load Common Label Encoder
    encoder_path = os.path.join("models", "label_encoder.pkl")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError("Label encoder file not found.")

    model_objects["encoder"] = joblib.load(encoder_path)
