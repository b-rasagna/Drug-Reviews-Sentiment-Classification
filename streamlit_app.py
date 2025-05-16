import requests
import streamlit as st

# Streamlit Configuration
st.set_page_config(
    page_title="Drug Review Sentiment Predictor",
    layout="centered")
st.title("Drug Review Sentiment Predictor (JWT Secured)")

# Sidebar Login Panel
st.sidebar.header("User Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

# Session token management
if "access_token" not in st.session_state:
    st.session_state.access_token = None

# Step 1: Authenticate with FastAPI
if login_button:
    login_url = "http://localhost:8000/login"
    try:
        response = requests.post(
            login_url, data={"username": username, "password": password}
        )
        if response.status_code == 200:
            st.session_state.access_token = response.json()["access_token"]
            st.sidebar.success("Login successful.")
        else:
            error_message = response.json().get("detail", "Unknown error")
            st.sidebar.error(f"Login failed: {error_message}")
    except Exception as e:
        st.sidebar.error(f"Login error: {e}")

# Block further interaction until logged in
if not st.session_state.access_token:
    st.warning("Please log in using the sidebar to access the model.")
    st.stop()

# Model Options
model_list = [
    "Logistic Regression (Imbalanced)",
    "Random Forest (Imbalanced)",
    "Linear SVC (Imbalanced)",
    "LightGBM (Imbalanced)",
    "Logistic Regression (SMOTE)",
    "Random Forest (SMOTE)",
    "Linear SVC (SMOTE)",
    "LightGBM (SMOTE)",
    "Bi-LSTM + CNN (Imbalanced)",
    "Bi-LSTM + CNN (Class Weights)",
]

st.markdown(
    """
**Best Performing Model:** Random Forest (Imbalanced)
**F1 Macro Score:** 0.8518
**Accuracy:** 90.38%
"""
)

# Model Selection
model_choice = st.selectbox(
    "Choose a model:",
    model_list,
    index=model_list.index("Random Forest (Imbalanced)"))

# User Input Area
user_input = st.text_area("Enter your drug review below:", height=150)

# Prediction Button
if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review to continue.")
    else:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}

        # Step 2: Select model
        select_url = "http://localhost:8000/select_model"
        select_response = requests.post(
            select_url, json={"model_name": model_choice}, headers=headers
        )

        if select_response.status_code != 200:
            st.error
            (f"Model selection failed: {select_response.json().get('detail')}")
            st.stop()

        # Step 3: Predict
        predict_url = "http://localhost:8000/predict"
        predict_response = requests.post(
            predict_url, json={"review": user_input}, headers=headers
        )

        if predict_response.status_code == 200:
            sentiment = predict_response.json().get("prediction")
            st.success(f"Predicted Sentiment: {sentiment.upper()}")
        else:
            try:
                error_detail = predict_response.json().get("detail",
                                                           "Unknown error")
            except Exception:
                error_detail = predict_response.text
            st.error(f"Prediction failed: {error_detail}")
