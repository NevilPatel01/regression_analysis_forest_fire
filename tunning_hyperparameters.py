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

# Load and prepare data
raw = np.genfromtxt('dataset.csv', delimiter=',', skip_header=1, dtype=str)
data = raw[:, :-1].astype(float)
targets = raw[:, -1]
targets = np.where(targets == 'fire', 1, 0).astype(float)
data, targets = shuffle(data, targets, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data, targets, test_size=0.2, random_state=42
)

# Polynomial Regression Tuning
print("="*50)
print("Tuning Polynomial Regression...")
poly_pipe = make_pipeline(PolynomialFeatures(), LinearRegression())
poly_params = {'polynomialfeatures__degree': [2, 3, 4, 5]}
poly_grid = GridSearchCV(poly_pipe, poly_params, cv=5, scoring='neg_mean_squared_error')
poly_grid.fit(X_train, y_train)
print(f"Best Parameters: {poly_grid.best_params_}")
print(f"Best MSE: {-poly_grid.best_score_:.4f}\n")

# K-NN Regressor Tuning
print("="*50)
print("Tuning K-NN Regressor...")

knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

knn_grid = GridSearchCV(KNeighborsRegressor(), knn_params, cv=5, scoring='neg_mean_squared_error')
knn_grid.fit(X_train, y_train)
print(f"Best Parameters: {knn_grid.best_params_}")
print(f"Best MSE: {-knn_grid.best_score_:.4f}\n")

# Decision Tree Regressor Tuning
print("="*50)
print("Tuning Decision Tree Regressor...")

dt_params = {
    'max_depth': [ 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'criterion': ['squared_error', 'friedman_mse']
}

dt_grid = GridSearchCV(DecisionTreeRegressor(), dt_params, cv=5, scoring='neg_mean_squared_error')
dt_grid.fit(X_train, y_train)
print(f"Best Parameters: {dt_grid.best_params_}")
print(f"Best MSE: {-dt_grid.best_score_:.4f}\n")

# Random Forest Regressor Tuning
print("="*50)
print("Tuning Random Forest Regressor...")

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 4, 6, 8],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=5, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)
print(f"Best Parameters: {rf_grid.best_params_}")
print(f"Best MSE: {-rf_grid.best_score_:.4f}\n")

# Support Vector Regressor Tuning
print("="*50)
print("Tuning Support Vector Regressor...")
svr_pipe = make_pipeline(StandardScaler(), SVR())

svr_params = {
    'svr__kernel': ['linear', 'poly'],
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': ['scale', 'auto', 0.1, 1],
    'svr__epsilon': [0.1, 0.2, 0.5]
}

svr_grid = GridSearchCV(svr_pipe, svr_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
svr_grid.fit(X_train, y_train)
print(f"Best Parameters: {svr_grid.best_params_}")
print(f"Best MSE: {-svr_grid.best_score_:.4f}\n")