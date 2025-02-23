"""Regression Algorithm Comparison with Cross-Validation.

Author: Nevil Patel, 000892482
This script tests multiple regression algorithms on the Algerian Forest Fires Dataset
using k-fold cross-validation to evaluate performance via Mean Squared Error (MSE).
The goal of this assignment is to identify the best-performing model for fire risk prediction.

References:
- Algerian Forest Fires Dataset: UCI Machine Learning Repository
- scikit-learn documentation for models and cross-validation

Date: 18th Feb, 2025
Author: Nevil Patel, 000892482
"""

"""
I, Nevil Patel, student number 000892482, certify that this material is my original work. No other person's work has been used without due acknowledgment and I have not made my work available to anyone else.
"""

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


## DATA LOADING AND PREPROCESSING

# Load raw data from CSV file (skip header row)
# Format: Features in columns 0-12, target in last column ('fire'/'not fire')
raw = np.genfromtxt('dataset.csv', delimiter=',', skip_header=1, dtype=str)

# Split into features (data) and target variable (fire status)
data = raw[:, :-1].astype(float)  # Convert all columns except last to floats
targets = raw[:, -1]  # Last column contains fire labels

# Encode target labels: 'fire' -> 1.0, others -> 0.0
targets = np.where(targets == 'fire', 1, 0).astype(float)

# Shuffle data to eliminate ordering bias (critical for time-series-like datasets)
# random_state=42 ensures reproducibility of the shuffle
data, targets = shuffle(data, targets, random_state=42)

# Display dataset statistics
print("Total samples: ", data.shape[0])
print("Number of features: ", data.shape[1])

## MODEL DEFINITIONS

# Note: Hyperparameters below were determined via prior grid search hyperparameter tunning

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

# Model configurations with tuned parameters
deg = 2 
k = 3   
d = 3   
estimators = 100

models = {
    # Baseline linear model
    "Linear Regression": LinearRegression(),
    
    # Polynomial regression pipeline: expand features to degree 2 + linear regression
    f"Polynomial (deg={deg})": make_pipeline(PolynomialFeatures(deg), LinearRegression()),
    
    # K-NN with distance-weighted voting (closer neighbors have more influence)
    f"K-NN (k={k})": KNeighborsRegressor(n_neighbors=k,p=1,weights='distance'),
    
    # Decision Tree with complexity control
    f"Decision Tree (d=3)": DecisionTreeRegressor(criterion='squared_error',  max_depth=d, min_samples_split=10),
    
    # Random Forest ensemble with optimized settings
    f"Random Forest (100)": RandomForestRegressor(max_depth=8, max_features='log2', min_samples_split=5, n_estimators=estimators),
    
    # Support Vector Regression with standardized features
    "SVR (linear)": make_pipeline(
        StandardScaler(),  
        SVR(kernel='rbf', C=1, gamma='scale', epsilon=0.1))
}

## CROSS-VALIDATION SETUP

from sklearn.model_selection import KFold

# Initialize 5-fold cross-validation with shuffling
# min(5) ensures we don't create more folds than samples
kf = KFold(n_splits=min(5, data.shape[0]), shuffle=True, random_state=42)

# Split data into training/testing (80/20) for final evaluation
data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.2, random_state=42)

print(f"\nTraining set size: {data_train.shape[0]}")
print(f"Testing set size: {data_test.shape[0]}")


## MODEL EVALUATION

from sklearn.model_selection import cross_validate

final_mse_metrices = {}  # Store MSE metrics for each model

for name, model in models.items():

    # Perform cross-validation with 5 folds
    mse_scores = cross_validate(model, data, targets, cv=kf, scoring='neg_mean_squared_error', return_train_score=False)
    
    # Convert to positive MSE values
    mse_scores = -mse_scores['test_score']
    
    # Store aggregated metrics
    final_mse_metrices[name] = {
        'Average MSE': np.mean(mse_scores),
        'Max MSE': np.max(mse_scores),
        'Min MSE': np.min(mse_scores)
    }


## RESULTS REPORTING
print("\nRegression Performance Comparison:")
for model, metrics in final_mse_metrices.items():
    print("="*40)
    print(f"{model}:")
    print(f"  Average MSE: {metrics['Average MSE']:.4f}")
    print(f"  Minimum MSE: {metrics['Min MSE']:.4f}") 
    print(f"  Maximum MSE: {metrics['Max MSE']:.4f}\n")