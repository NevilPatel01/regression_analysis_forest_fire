import matplotlib.pyplot as plt
import numpy as np

models = ["Linear", "Polynomial", "K-NN", "Decision Tree", "Random Forest", "SVR"]
avg_mse = [0.0826, 0.5900, 0.0538, 0.0149, 0.0167, 0.0602]
min_mse = [0.0644, 0.0972, 0.0395, 0.0016, 0.0033, 0.0497]
max_mse = [0.1052, 1.7168, 0.0838, 0.0408, 0.0279, 0.0702]

plt.figure(figsize=(12, 6))
bars = plt.bar(models, avg_mse, yerr=[np.subtract(avg_mse, min_mse), np.subtract(max_mse, avg_mse)], capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
plt.title("Regression Algorithm Performance Comparison", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yscale('log')  # Log scale to visualize large MSE ranges
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()