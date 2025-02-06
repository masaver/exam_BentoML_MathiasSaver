import pytest
import requests
from datetime import datetime, timedelta
import jwt

# Define API endpoints
LOGIN_URL = "http://localhost:3000/login"
PREDICT_URL = "http://localhost:3000/v1/models/rf_regressor/predict"

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "super_secret_key_obtained_from_somewhere_!@#$"
JWT_ALGORITHM = "HS256"

# ---- HELPER FUNCTION TO CREATE A JWT TOKEN ----
def create_jwt_token(expiry_minutes=30):
    """Generate a JWT token for testing"""
    payload = {
        "sub": "test_user",
        "exp": datetime.utcnow() + timedelta(minutes=expiry_minutes)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm = JWT_ALGORITHM )


# ---- 1. LOGIN API TESTS ----
valid_credentials = { 
    "username" : "dst" , 
    "password" : "super_secret_password" 
    }

def test_successful_login():
    """Should return a JWT token for valid credentials"""
    response = requests.post( LOGIN_URL , json = valid_credentials )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert isinstance(data["access_token"], str)

wrong_credentials = {
    "username": "wrong_user",
    "password": "wrongpass"
}

def test_failed_login():
    """Should return 401 for invalid credentials"""
    response = requests.post(LOGIN_URL, json=wrong_credentials )
    
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


# ---- 2. PREDICTION API TESTS ----
# Example input data for prediction - "chance_of_admit": 0.93
valid_input = {
    "gre_score": 334,
    "toefl_score": 116,
    "university_ranking": 4,
    "sop": 4,
    "lor": 3.5,
    "cgpa": 9.54,
    "research": 1,
}

def test_missing_jwt_prediction():
    """Should return 401 when JWT is missing in prediction request"""
    response = requests.post(PREDICT_URL, json=valid_input)
    
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"



def test_valid_prediction():
    """Should return a valid prediction for correct input"""
    valid_token = create_jwt_token()
    headers = {"Authorization": f"Bearer {valid_token}"}

    response = requests.post(PREDICT_URL, json={"data": [valid_input]}, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float, list))  # Ensure prediction is numerical


def test_invalid_prediction():
    """Should return 422 when input data is invalid"""
    valid_token = create_jwt_token()
    headers = {"Authorization": f"Bearer {valid_token}"}

    response = requests.post(PREDICT_URL, json={"invalid_field": "test"}, headers=headers)
    
    assert response.status_code == 422  # Unprocessable Entity


def test_expired_jwt():
    """Should return 401 when JWT is expired"""
    expired_token = create_jwt_token(expiry_minutes=-1)
    headers = {"Authorization": f"Bearer {expired_token}"}

    response = requests.post(PREDICT_URL, json={"data": [1, 2, 3, 4]}, headers=headers)
    
    assert response.status_code == 401
    assert response.json()["detail"] == "Token has expired"
