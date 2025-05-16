# Drug Review Sentiment Classification

This project is an end-to-end machine learning and deep learning pipeline designed to classify drug reviews into **Positive**, **Negative**, or **Neutral** categories. It provides:

- A FastAPI backend with JWT authentication
- A secure Streamlit dashboard for real-time predictions
- A complete training notebook covering EDA, preprocessing, and model evaluation
- Comparison of 10 models trained on both imbalanced and balanced datasets

---

## Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Folder Structure](#folder-structure)
- [Testing](#testing)

---

## Overview

The system allows users to:

- Train and evaluate models using a Jupyter notebook
- Save trained models, tokenizers, and vectorizers
- Dynamically load saved models using FastAPI
- Make authenticated predictions via API or Streamlit interface
- Choose between ML and DL models trained on imbalanced or balanced data

---

## Model Performance

| Model                             | Accuracy | F1 (Macro) | ROC-AUC | Log Loss |
|----------------------------------|----------|------------|---------|----------|
| Random Forest (Imbalanced)       | 0.9041   | 0.8519     | 0.9634  | 0.3514   |
| Random Forest (SMOTE)            | 0.9008   | 0.8481     | 0.9638  | 0.3608   |
| Bi-LSTM + CNN (Class Weights)    | 0.7496   | 0.6617     | 0.9133  | 0.5729   |
| Bi-LSTM + CNN (Imbalanced)       | 0.8417   | 0.6164     | 0.9115  | 0.4283   |
| Logistic Regression (SMOTE)      | 0.7409   | 0.6160     | 0.8495  | 0.6387   |
| Linear SVC (SMOTE)               | 0.7410   | 0.6152     | N/A     | N/A      |
| Linear SVC (Imbalanced)          | 0.7980   | 0.5730     | N/A     | N/A      |
| Logistic Regression (Imbalanced) | 0.7984   | 0.5708     | 0.8678  | 0.5213   |
| LightGBM (SMOTE)                 | 0.7329   | 0.5672     | 0.8184  | 0.6876   |
| LightGBM (Imbalanced)            | 0.7724   | 0.5107     | 0.8454  | 0.5851   |

> **Recommended model for deployment**: Random Forest (Imbalanced)

---

## End-to-End Pipeline

The pipeline consists of two phases:

### 1. Notebook Phase

- Conducts EDA and preprocessing
- Trains 10 models (ML and DL)
- Applies SMOTE and class weights to handle imbalance
- Saves models, vectorizers, encoders, and tokenizers in the `models/` directory

### 2. Application Phase

- The FastAPI app loads selected models on demand
- Tokenizers and vectorizers are reused without retraining
- Sentiment predictions are made on cleaned input reviews

---

## Getting Started

### Requirements

- Python 3.11+
- pip
- Linux/macOS or WSL (Windows Subsystem for Linux)

### Quick Start

```bash
chmod +x run_app.sh
./run_app.sh
```

This will:

* Set up a virtual environment
* Install dependencies
* Launch FastAPI at `http://localhost:8000`
* Launch Streamlit at `http://localhost:8501`

---

## Usage

### Streamlit Dashboard

* Visit `http://localhost:8501`

* Login with:

  ```
  Username: admin
  Password: pass123
  ```

* Choose a model

* Paste a review

* View predicted sentiment

---

## API Reference

All endpoints require JWT-based authentication.

### 1. Login

`POST /login`

**Form Data:**

```
username=admin&password=pass123
```

**Response:**

```json
{ "access_token": "<JWT>", "token_type": "bearer" }
```

---

### 2. Select Model

`POST /select_model`

**Headers:**

```
Authorization: Bearer <JWT>
```

**Body:**

```json
{ "model_name": "Random Forest (Imbalanced)" }
```

---

### 3. Predict Sentiment

`POST /predict`

**Headers:**

```
Authorization: Bearer <JWT>
```

**Body:**

```json
{ "review": "This medicine worked very well for my symptoms." }
```

**Response:**

```json
{ "prediction": "positive" }
```

---

## Folder Structure

```
Drug_Reviews_Sentiment_Analysis/
│
├── app/                  # FastAPI source code
│   ├── main.py           # API routes
│   ├── auth.py           # JWT logic
│   ├── model_selector.py # Load models
│   ├── prediction.py     # Inference logic
│   └── text_cleaning.py  # Text preprocessing
│
├── models/               # Saved model files (pkl, h5)
├── tests/                # Unit tests
├── streamlit_app.py      # Frontend dashboard
├── run_app.sh            # Setup script
├── requirements.txt      # Dependencies
├── .env                  # API secrets
├── Drug_Reviews_Sentiment_Analysis.ipynb # Training + EDA
├── README.md             # This file
```

---

## Testing

```bash
pytest tests/
```

Unit tests include:

* API endpoint coverage
* Text cleaning logic
* Model loading

---
