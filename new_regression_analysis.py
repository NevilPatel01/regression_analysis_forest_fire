import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


raw = np.genfromtxt('dataset.csv', delimiter=',', skip_header=1, dtype=str)

data=raw[:, :-1].astype(float)
targets=raw[:, -1]

targets = np.where(targets == 'fire', 1, 0).astype(float)

data, targets = shuffle(data, targets, random_state=42)

print("Total samples: ", data.shape[0])
print("Number of features: ", data.shape[1])


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

deg=2
k=3
d=4
estimators=50

models = {
    "Linear Regression": LinearRegression(),
    f"Polynomial (deg={deg})": make_pipeline(PolynomialFeatures(deg), LinearRegression()),
    f"K-NN (k={k})": KNeighborsRegressor(n_neighbors=k, p=1,weights='distance'),
    f"Decision Tree (d=3)": DecisionTreeRegressor(criterion='squared_error', max_depth=3, min_samples_split=10),
    f"Random Forest (100)": RandomForestRegressor(max_depth=8, max_features='log2', min_samples_split= 5, n_estimators=100),
    "SVR (linear)": make_pipeline(StandardScaler(), SVR(C=1, gamma='scale', epsilon=0.2, kernel='linear'))
}


from sklearn.model_selection import KFold

# Set up k-fold cross-validation
kf = KFold(n_splits=min(5, data.shape[0]), shuffle=True, random_state=42)
# print("kf", kf)

data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.2, random_state=42)

# Print the sizes of the training and testing sets
print(f"Training set size: ", data_train.shape[0])
print(f"Testing set size: ", data_test.shape[0])
results = {}

from sklearn.model_selection import cross_validate

for name, model in models.items():
    mse_scores = cross_validate(model, data, targets, cv=kf, scoring='neg_mean_squared_error', return_train_score=False)
    mse_scores = -mse_scores['test_score']  # Convert back to positive MSE
    results[name] = {
        'Average MSE': np.mean(mse_scores),
        'Max MSE': np.max(mse_scores),
        'Min MSE': np.min(mse_scores)
    }

# Print results
print()
print("Regression Performance Comparison:")
for model, metrics in results.items():
    print("="*22)
    print()
    print(model+ ": ")
    print(f"  Average MSE: {metrics['Average MSE']:.4f}")
    print(f"  Minimum MSE: {metrics['Min MSE']:.4f}")
    print(f"  Maximum MSE: {metrics['Max MSE']:.4f}")
    print()
