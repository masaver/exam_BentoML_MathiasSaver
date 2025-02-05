import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('./data/raw/admission.csv')

# Define the feature columns and the target variable
X = data.drop(columns=['serial_no', 'chance_of_admit'])
y = data['chance_of_admit']

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split datasets to CSV files
X_train.to_csv('./data/processed/X_train.csv', index=False)
X_test.to_csv('./data/processed/X_test.csv', index=False)
y_train.to_csv('./data/processed/y_train.csv', index=False)
y_test.to_csv('./data/processed/y_test.csv', index=False)