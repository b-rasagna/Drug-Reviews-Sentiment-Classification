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
- [Docker Support](#docker-support)
- [CI/CD Support](#cicd-support)

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

The complete system consists of two core phases, designed to deliver a robust end-to-end machine learning solution:

---

### 1. Notebook Phase - Data Exploration, Model Training & Artifact Preparation

* Conducts thorough Exploratory Data Analysis (EDA) and preprocessing on the drug reviews dataset to identify trends, clean data, and prepare features.
* Trains **10 distinct models**, spanning classical Machine Learning algorithms (e.g., Random Forest, Logistic Regression) and Deep Learning architectures (e.g., Bi-LSTM + CNN).
* Implements strategies such as **SMOTE** for oversampling minority classes and **class weighting** to mitigate data imbalance and improve model generalization.
* Saves all trained models and essential preprocessing artifacts—including vectorizers, encoders, and tokenizers—into a centralized `models/` directory for streamlined loading during inference.

---

### 2. Application Phase — Model Deployment & Real-Time Prediction

* A FastAPI backend dynamically loads the selected model upon client requests, optimizing memory use and enabling easy model switching without downtime.
* Preprocessing tools like tokenizers and vectorizers are reused as-is, ensuring consistency between training and inference data processing.
* Provides secure, JWT-authenticated API endpoints for model selection and sentiment prediction on user-provided drug reviews.
* Integrates with a Streamlit frontend that allows users to authenticate, select models, enter reviews, and view predicted sentiment in real-time.

---

This modular two-phase approach enables efficient model development, management, and deployment, supporting a scalable and maintainable sentiment classification system.

---

## Getting Started

### Requirements

- Python 3.11+
- pip
- Linux/macOS or WSL (Windows Subsystem for Linux)

### .env Setup

Before running the app, create a `.env` file in the root directory and add JWT secret:

```env
JWT_SECRET=super_secret_key
JWT_ALGORITHM=HS256
```

* `JWT_SECRET` is used to sign and verify JWT tokens. Use a strong, random string.
* `JWT_ALGORITHM` defaults to `HS256` if not specified (optional).

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

## Docker Support

Application can be built and served using Docker with a single command.

### Prerequisites

* [Docker](https://docs.docker.com/get-docker/) must be installed and running.
* Ensure user is part of the `docker` group to avoid using `sudo`:

  ```bash
  sudo usermod -aG docker $USER
  newgrp docker
  ```

---

### 1. Add `.env` File

Make sure project root contains a `.env` file like this:

```env
JWT_SECRET=jwt_secret_key
JWT_ALGORITHM=HS256
```

---

### 2. Build the Docker Image

```bash
docker build -t drug-review-app .
```

---

### 3. Run the Container

```bash
docker run --env-file .env -p 8000:8000 -p 8501:8501 drug-review-app
```

* FastAPI API: [http://localhost:8000](http://localhost:8000)
* Streamlit App: [http://localhost:8501](http://localhost:8501)

---

### One-Liner Script

Helper script can also be used:

```bash
chmod +x docker_run.sh
./docker_run.sh
```

---

## CI/CD Support

### Continuous Integration (CI)

This project includes GitHub Actions CI workflow:

* Linting using `flake8`
* Unit testing using `pytest`
* Dependency installation from `requirements.txt`

The CI workflow automatically runs on every push and pull request to the `main` branch.
CI configuration is defined in:

```
.github/workflows/python-app.yml
```

---

### Continuous Deployment (CD)

While CD is not set up by default, we can extend this project with **Docker-based deployment**:

* Containerize the app using the provided `Dockerfile`
* Use services like **GitHub Actions + DockerHub**, **GitLab CI/CD**, or **AWS/GCP Cloud Build** to push containers
* Deploy to **cloud platforms** (e.g., AWS ECS, GCP Cloud Run, Heroku) or **Kubernetes clusters**

---
