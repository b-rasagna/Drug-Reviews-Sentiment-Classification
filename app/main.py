import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from app.auth import create_token, decode_token
from app.model_selector import load_model_by_name
from app.prediction import predict_sentiment

# Load environment variables from .env
load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# Initialize FastAPI App
app = FastAPI()

# In-memory store to hold the selected model and components
model_objects = {
    "model": None,
    "vectorizer": None,
    "tokenizer": None,
    "encoder": None,
    "type": None,  # 'ml' or 'dl'
}

# OAuth2 Configuration for JWT Token Authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


def verify_jwt_token(token: str = Depends(oauth2_scheme)):
    """
    Verify and decode the provided JWT token.
    Raises 403 error if the token is invalid or expired.
    """
    user = decode_token(token)
    if user is None:
        raise HTTPException(
            status_code=403,
            detail="Invalid or "
            "expired JWT token.")
    return user


# Simple in-memory user store
# For production, replace this with hashed credentials or a database.
users = {"admin": "pass123"}


# Endpoint: /login
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticates a user and returns a JWT access token upon success.
    """
    username = form_data.username
    password = form_data.password

    if users.get(username) != password:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or "
            "password")

    token = create_token(username)
    return {"access_token": token, "token_type": "bearer"}


# Pydantic Models for API Request Bodies
class ModelRequest(BaseModel):
    model_name: str


class PredictionRequest(BaseModel):
    review: str


# Endpoint: /select_model
@app.post("/select_model")
def select_model(request: ModelRequest, user=Depends(verify_jwt_token)):
    """
    Loads the specified machine learning or deep learning model into memory.
    Requires JWT authentication.
    """
    try:
        load_model_by_name(request.model_name, model_objects)
        return {"message": f"Model '{request.model_name}' loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint: /predict
@app.post("/predict")
def predict(request: PredictionRequest, user=Depends(verify_jwt_token)):
    """
    Predicts sentiment from a drug review using the previously selected model.
    Requires JWT authentication.
    """
    if model_objects["model"] is None:
        raise HTTPException(
            status_code=400,
            detail="No model selected." "Please select the " "model first.",
        )

    return predict_sentiment(request.review, model_objects)
