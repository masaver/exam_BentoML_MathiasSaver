import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import jwt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import bentoml
from bentoml.io import JSON
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta

### Train the model ###

# Custom functions
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Split features and target
x_train = pd.read_csv('data/processed/X_train.csv')
x_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# # Add a standar scaler to transforme the features
# scaler = StandardScaler()
# scaler.fit( x_train )

# # Transform the features
# x_train = pd.DataFrame( scaler.transform(x_train) , columns = x_train.columns )
# x_test = pd.DataFrame( scaler.transform(x_test) , columns = x_test.columns )

# Grid Search to find the best hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(
    estimator = xgb_regressor , 
    param_grid = param_grid , 
    cv = 5 , 
    scoring = 'neg_mean_squared_error' , 
    verbose = 1 
    )
grid_search.fit(x_train, y_train)

print(f'Best parameters found: {grid_search.best_params_}')
print(f'Best score: {-1*grid_search.best_score_}')

# Save the best model to a .pkl file
reg = grid_search.best_estimator_
with open('models/xgb_regressor.pkl', 'wb') as f:
    pickle.dump( reg , f )

# Model evaluation

# Predict on training and test data
y_pred_train = reg.predict(x_train)
y_pred_test = reg.predict(x_test)

# Calculate performance metrics for training data
rmse_train = root_mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# Calculate performance metrics for test data
rmse_test = root_mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Save the performance metric to a json file
performance_metrics = {
    'train': {
        'RMSE': rmse_train,
        'Mean Absolute Error': mae_train,
        'R^2 Score': r2_train
    },
    'test': {
        'RMSE': rmse_test,
        'Mean Absolute Error': mae_test,
        'R^2 Score': r2_test
    }
}

with open('models/performance_metrics.json', 'w') as f:
    json.dump(performance_metrics, f, indent=4)

os.system('cat models/performance_metrics.json')

# Save the model in BentoML's Model Store
model_ref = bentoml.sklearn.save_model("xbg_regressor", reg )
print(f"Model saved as: {model_ref}")


### Create & Save the Bentoml Service ###

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "super_secret_key_obtained_from_somewhere!"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "dst": "super_secret_password"
}

# Function to create a JWT token
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

# Autheticatin Middleware class
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/models/xgb_regressor/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})

            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})

            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response
    
# Pydantic model to validate input data
class InputModel(BaseModel):
    gre_score: int
    toefl_score: int
    university_ranking: int
    sop: float
    lor: float
    cgpa: float
    research: int

# Get the model from the Model Store
xgb_reg_runner = bentoml.sklearn.get("xbg_regressor:latest").to_runner()

# Create a service API
xgb_service = bentoml.Service("xgb_reg_service", runners = [ xgb_reg_runner ] )

# Create BentoService
class XgbRegressortService(bentoml.Service):

    #Login endpoint
    @xgb_service.api(input=JSON(), output=JSON())
    def login(credentials: dict) -> dict:
        username = credentials.get("username")
        password = credentials.get("password")
        users = USERS

        if username in users and users[username] == password:
            token = create_jwt_token(username)
            return {"token": token}
        else:
            return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})
        
    # Predict EndPoint
    @xgb_service.api( 
        input = JSON( pydantic_model = InputModel ) , 
        output = JSON() , 
        route='v1/models/xgb_regressor/predict'
        )
    
    async def predict(input_data: InputModel) -> dict:
        # Convert the input data to a numpy array
        input_series = np.array([
            input_data.gre_score, input_data.toefl_score, input_data.university_ranking,
            input_data.sop, input_data.lor, input_data.cgpa, input_data.research
            ])

        result = await xgb_reg_runner.predict.async_run( input_series.reshape(1, -1) )

        return {"prediction": result.tolist()}


# Create service
service = XgbRegressortService( 'xgb_reg_service' )

# Pack model into service
service.pack("model", xgb_reg_runner)

# Save the Bento in the './bentos' folder (this will generate model.yaml)
service.save('/home/ubuntu/bentos/xgb_reg_service')