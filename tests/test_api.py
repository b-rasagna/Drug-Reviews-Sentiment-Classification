from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_login_success():
    response = client.post(
        "/login",
        data={
            "username": "admin",
            "password": "pass123"})
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_fail():
    response = client.post(
        "/login",
        data={
            "username": "admin",
            "password": "wrong"})
    assert response.status_code == 401


def test_select_model_and_predict():
    login = client.post(
        "/login",
        data={
            "username": "admin",
            "password": "pass123"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Select model
    select = client.post(
        "/select_model",
        json={"model_name": "Random Forest (Imbalanced)"},
        headers=headers,
    )
    assert select.status_code == 200

    # Predict
    predict = client.post(
        "/predict",
        json={
            "review": "This medicine helped me a lot."},
        headers=headers)
    assert predict.status_code == 200
    assert "prediction" in predict.json()
