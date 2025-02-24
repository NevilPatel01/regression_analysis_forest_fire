"""Hyperparameter Tuning for Regression Models.

This script performs grid search optimization for multiple regression algorithms
on the Algerian Forest Fires Dataset to find optimal model configurations.

Key Steps:
1. Load and preprocess data (shuffle, train/test split)
2. Tune hyperparameters for 5 regression models using GridSearchCV
3. Report best parameters and cross-validated MSE for each model

Note: Each model is tuned independently using the same training set (80% of data).
The test set (20%) is reserved for final evaluation (not used here).

Date: 18th Feb, 2025
Author: Nevil Patel, 000892482
"""

"""
I, Nevil Patel, student number 000892482, certify that this material is my original work. No other person's work has been used without due acknowledgment and I have not made my work available to anyone else.
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

## DATA PREPARATION

# Load dataset from CSV by skipping header row
raw = np.genfromtxt('dataset.csv', delimiter=',', skip_header=1, dtype=str)

# Extract features and convert to floats except the prediction 
data = raw[:, :-1].astype(float)

# Extract target variable and encode as binary
# 'fire' -> 1.0, other values -> 0.0
targets = np.where(raw[:, -1] == 'fire', 1, 0).astype(float)

# Shuffle data to eliminate order effects and ensure random distribution
data, targets = shuffle(data, targets, random_state=42)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

## POLYNOMIAL REGRESSION TUNING

print("="*50)
print("Tuning Polynomial Regression...")

# Create pipeline: polynomial features + linear regression
poly_pipe = make_pipeline(PolynomialFeatures(), LinearRegression())

# Test polynomial degrees
poly_params = {'polynomialfeatures__degree': [2, 3, 4, 5]}

# Grid search with 5-fold CV
poly_grid = GridSearchCV( poly_pipe, poly_params, cv=5, scoring='neg_mean_squared_error' )
poly_grid.fit(X_train, y_train)

print(f"Best Parameters: {poly_grid.best_params_}")
print(f"Best MSE: {-poly_grid.best_score_:.4f}\n")  # Convert MSE back to positive MSE


## K-NEAREST NEIGHBORS TUNING

print("="*50)
print("Tuning K-NN Regressor...")

# Parameter grid with different neighborhood configurations
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'], 
    'p': [1, 2]  
}

knn_grid = GridSearchCV( KNeighborsRegressor(), knn_params, cv=5, scoring='neg_mean_squared_error')
knn_grid.fit(X_train, y_train)

print(f"Best Parameters: {knn_grid.best_params_}")
print(f"Best MSE: {-knn_grid.best_score_:.4f}\n")


## DECISION TREE TUNING

print("="*50)
print("Tuning Decision Tree Regressor...")

# Parameters control tree complexity to prevent overfitting
dt_params = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10], 
    'criterion': ['squared_error', 'friedman_mse'] 
}

dt_grid = GridSearchCV( DecisionTreeRegressor(), dt_params, cv=5, scoring='neg_mean_squared_error')
dt_grid.fit(X_train, y_train)

print(f"Best Parameters: {dt_grid.best_params_}")
print(f"Best MSE: {-dt_grid.best_score_:.4f}\n")


## RANDOM FOREST TUNING

print("="*50)
print("Tuning Random Forest Regressor...")

# Parameters balance ensemble diversity and individual tree complexity
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 4, 6, 8],  
    'max_features': ['sqrt', 'log2'], 
    'min_samples_split': [2, 5, 10] 
}

rf_grid = GridSearchCV( RandomForestRegressor(), rf_params, cv=5, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)

print(f"Best Parameters: {rf_grid.best_params_}")
print(f"Best MSE: {-rf_grid.best_score_:.4f}\n")


## SUPPORT VECTOR REGRESSOR TUNING

print("="*50)
print("Tuning Support Vector Regressor...")

# SVR requires feature scaling for optimal performance
svr_pipe = make_pipeline( StandardScaler(), SVR())

# Kernel and regularization parameters
svr_params = {
    'svr__kernel': ['linear', 'rbf' , 'poly'],  # Relationship between features
    'svr__C': [0.1, 1, 10, 100],  # Regularization strength (inverse of lambda)
    'svr__gamma': ['scale', 'auto', 0.1, 1],  # Kernel coefficient
    'svr__epsilon': [0.1, 0.2, 0.5]  # Error margin tolerance
}

# n_jobs=-1 enables parallel processing for faster tuning
svr_grid = GridSearchCV( svr_pipe, svr_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
svr_grid.fit(X_train, y_train)

print(f"Best Parameters: {svr_grid.best_params_}")
print(f"Best MSE: {-svr_grid.best_score_:.4f}\n")
