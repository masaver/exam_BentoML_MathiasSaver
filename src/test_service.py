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
    assert "token" in data
    assert isinstance(data["token"], str)

wrong_credentials = {"username": "wrong_user", "password": "wrong_pass"}

def test_failed_login():
    """Should return 401 for invalid credentials"""
    response = requests.post( LOGIN_URL , json = wrong_credentials )
    
    data = response.json()
    assert "token" not in data


# ---- 2. PREDICTION API TESTS ----
valid_input = {
    "gre_score": 320,
    "toefl_score": 110,
    "university_ranking": 3,
    "sop": 4.5,
    "lor": 4.0,
    "cgpa": 9.54,
    "research": 1
}

def test_missing_jwt_prediction():
    """Should return 401 when JWT is missing in prediction request"""
    response = requests.post(PREDICT_URL, json=valid_input)
    
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing authentication token"

def test_valid_prediction():
    """Should return a valid prediction for correct input"""
    valid_token = create_jwt_token()
    headers = {"Authorization": f"Bearer {valid_token}"}

    response = requests.post(PREDICT_URL, json=valid_input, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float, list))  # Ensure prediction is numerical


# Example of an invalid input (missing required fields, wrong data types)
invalid_input = {
    "gre_score": "high",  # Invalid type: should be an int
    "toefl_score": 110,
    "university_ranking":  "the best",  # Invalid type: should be an int
    "sop": "strong",  # Invalid type: should be a float
    "lor": 4.0,
    "cgpa": 9.54,
    "research": 1
}

def test_invalid_prediction():
    """Should return 400 when input data is invalid"""
    valid_token = create_jwt_token()
    headers = {"Authorization": f"Bearer {valid_token}"}
    response = requests.post(PREDICT_URL, json=invalid_input, headers=headers)
    
    assert response.status_code == 400  # Unprocessable Entity

# ---- 3. JWT AUTHENTICATION TESTS ----
def test_expired_jwt():
    """Should return 401 when JWT is expired"""
    expired_token = create_jwt_token(expiry_minutes=-1)
    headers = {"Authorization": f"Bearer {expired_token}"}

    response = requests.post(PREDICT_URL, json=valid_input, headers=headers)
    
    assert response.status_code == 401
    assert response.json()["detail"] == "Token has expired"

def test_auth_fails_missing_or_invalid():
    headers = {"Authorization": "Bearer invalid_token"}
    response = requests.post(PREDICT_URL, json=valid_input, headers=headers)
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"

    response = requests.post(PREDICT_URL, json=valid_input)  # No token
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"

def test_auth_succeeds_valid_token():
    valid_token = create_jwt_token()
    headers = {"Authorization": f"Bearer {valid_token}"}
    
    response = requests.post(PREDICT_URL, json=valid_input, headers=headers)
    assert response.status_code == 200
    assert "prediction" in response.json()