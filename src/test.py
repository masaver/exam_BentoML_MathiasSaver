import requests

# The URL of the login and prediction endpoints
login_url = "http://127.0.0.1:3000/login"
predict_url = "http://127.0.0.1:3000/v1/models/xgb_regressor/predict"

# Données de connexion
credentials = {
    "username": "dst",
    "password": "super_secret_password"
}

# Send a POST request to the login endpoint
login_response = requests.post(
    login_url,
    headers={"Content-Type": "application/json"},
    json=credentials
)

# Check if the login was successful
if login_response.status_code == 200:
    token = login_response.json().get("token")
    print("Token JWT obtenu:", token)

    # Data to be sent to the prediction endpoint
    data = {
        'gre_score' : 321 , 
        'toefl_score' : 111 , 
        'university_ranking' : 3 , 
        'sop' : 3.5 , 
        'lor' : 4.0 , 
        'cgpa' : 8.83 , 
        'research' : 1
    }

    # Send a POST request to the prediction
    response = requests.post(
        predict_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        },
        json=data
    )

    print(f"API response :\n{response.text}")
else:
    print("Connection Error", login_response.text)