# Regression Model Comparison Project 🔥📊

A comprehensive comparison of various regression models using k-fold cross-validation to predict binary outcomes (fire detection). Developed for educational purposes and machine learning benchmarking.

![Regression Analysis](https://img.shields.io/badge/Regression-Analysis-blue)
![Scikit-Learn](https://img.shields.io/badge/Powered%20By-Scikit--Learn-orange)
![Python 3](https://img.shields.io/badge/Python-3-blue.svg)

## Features ✨

- **6 Regression Models** compared using MSE metrics
- **k-fold Cross-Validation** (k=5) for robust evaluation
- Automated dataset loading and preprocessing
- Clear performance reporting with key statistics
- Reproducible results through random state control

## Models Implemented 🤖
1. Linear Regression
2. Polynomial Regression (Degree 2)
3. K-Nearest Neighbors (k=5)
4. Decision Tree (Max Depth=4)
5. Random Forest (50 Estimators)
6. Support Vector Regression (Linear Kernel)

## Installation 🛠️

```bash
# Clone repository
git clone https://github.com/NevilPatel01/regression_analysis_forest_fire.git

# Install requirements
pip install numpy scikit-learn

## Usage 🚀
1. Place your dataset.csv in the project root.
2. Run the analysis script:

```bash
python new_regression_analysis.py
```
## Key Metrics 📈
- Mean Squared Error (MSE)
- Maximum MSE across folds
- Minimum MSE across folds

## Technical Details ⚙️
```bash
# Core configuration
test_size = 0.2        # 20% testing data
k_folds = 5            # Cross-validation folds
random_state = 42      # Reproducibility seed

# Model hyperparameters
polynomial_degree = 2
knn_neighbors = 5
tree_depth = 4
forest_estimators = 50
```

