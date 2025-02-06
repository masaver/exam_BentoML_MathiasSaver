import os
import pickle
import bentoml
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import json
# rom sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

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
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
grid_search = GridSearchCV(
    estimator =  RandomForestRegressor( random_state=17 ) , 
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
with open('models/rf_regressor.pkl', 'wb') as f:
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
model_ref = bentoml.sklearn.save_model("rf_regressor", reg )
print(f"Model saved as: {model_ref}")
