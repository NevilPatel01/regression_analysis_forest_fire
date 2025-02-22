"""Regression Algorithm Comparison with Cross-Validation.

This script tests multiple regression algorithms on the Algerian Forest Fires Dataset
using k-fold cross-validation to evaluate performance via Mean Squared Error (MSE).
The goal is to identify the best-performing model for fire risk prediction.

Key Steps:
1. Load and preprocess data (shuffle, split features/targets, encode labels).
2. Define regression models with tuned hyperparameters.
3. Perform 5-fold cross-validation.
4. Report average, minimum, and maximum MSE for each model.

Note: All models are evaluated on the same shuffled data for fair comparison.

References:
- Algerian Forest Fires Dataset: UCI Machine Learning Repository
- scikit-learn documentation for models and cross-validation

Modified from original code by [Your Name], [Date]
"""

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# =================================================================
# DATA LOADING AND PREPROCESSING
# =================================================================

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

# =================================================================
# MODEL DEFINITIONS
# =================================================================
# Note: Hyperparameters below were determined via prior grid search optimization

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

# Model configurations with tuned parameters
deg = 2  # Optimal polynomial degree from prior testing
k = 3    # Best k-value for K-NN
d = 3    # Optimal tree depth limit to prevent overfitting
estimators = 100  # Number of trees in Random Forest

models = {
    # Baseline linear model
    "Linear Regression": LinearRegression(),
    
    # Polynomial regression pipeline: expand features to degree 2 + linear regression
    f"Polynomial (deg={deg})": make_pipeline(
        PolynomialFeatures(deg), 
        LinearRegression()
    ),
    
    # K-NN with distance-weighted voting (closer neighbors have more influence)
    f"K-NN (k={k})": KNeighborsRegressor(
        n_neighbors=k,
        p=1,  # Manhattan distance (more robust to outliers)
        weights='distance'
    ),
    
    # Decision Tree with complexity control
    f"Decision Tree (d=3)": DecisionTreeRegressor(
        criterion='squared_error',  # Standard MSE splitting
        max_depth=d,  # Limit tree depth
        min_samples_split=10  # Require 10 samples to split a node
    ),
    
    # Random Forest ensemble with optimized settings
    f"Random Forest (100)": RandomForestRegressor(
        max_depth=8,  # Deeper than single tree but with feature constraints
        max_features='log2',  # Features per split: log2(n_features)
        min_samples_split=5,  # More splits allowed than in single tree
        n_estimators=estimators  # Number of trees in ensemble
    ),
    
    # Support Vector Regression with standardized features
    "SVR (linear)": make_pipeline(
        StandardScaler(),  # Critical for SVR performance
        SVR(kernel='linear',  # Linear kernel performed best
            C=1,  # Regularization parameter
            gamma='scale',  # Kernel coefficient
            epsilon=0.2)  # Margin of tolerance
    )
}

# =================================================================
# CROSS-VALIDATION SETUP
# =================================================================

from sklearn.model_selection import KFold

# Initialize 5-fold cross-validation with shuffling
# min(5, ...) ensures we don't create more folds than samples
kf = KFold(n_splits=min(5, data.shape[0]), 
          shuffle=True, 
          random_state=42)  # Reproducible splits

# Split data into training/testing (80/20) for final evaluation
# Note: Cross-validation uses full dataset, this split is just for demonstration
data_train, data_test, targets_train, targets_test = train_test_split(
    data, targets, 
    test_size=0.2, 
    random_state=42
)

print(f"\nTraining set size: {data_train.shape[0]}")
print(f"Testing set size: {data_test.shape[0]}")

# =================================================================
# MODEL EVALUATION
# =================================================================

from sklearn.model_selection import cross_validate

results = {}  # Store MSE metrics for each model

for name, model in models.items():
    # Perform cross-validation with 5 folds
    # Note: Uses full dataset (data/targets), not train/test split
    mse_scores = cross_validate(
        model, 
        data, 
        targets, 
        cv=kf, 
        scoring='neg_mean_squared_error',  # sklearn convention: higher=better
        return_train_score=False
    )
    
    # Convert to positive MSE values (lower is better)
    mse_scores = -mse_scores['test_score']
    
    # Store aggregated metrics
    results[name] = {
        'Average MSE': np.mean(mse_scores),
        'Max MSE': np.max(mse_scores),
        'Min MSE': np.min(mse_scores)
    }

# =================================================================
# RESULTS REPORTING
# =================================================================

print("\nRegression Performance Comparison:")
for model, metrics in results.items():
    print("="*40)
    print(f"{model}:")
    print(f"  Average MSE: {metrics['Average MSE']:.4f}")
    print(f"  Minimum MSE: {metrics['Min MSE']:.4f}") 
    print(f"  Maximum MSE: {metrics['Max MSE']:.4f}\n")