import matplotlib.pyplot as plt

# Data
algorithms = ['Linear', 'Polynomial', 'K-NN', 'Decision Tree', 'Random Forest', 'SVR']
average_mse = [0.0826, 0.5900, 0.538, 0.0149, 0.0167, 0.0836]

# Plot
plt.figure(figsize=(10, 10))
bars = plt.bar(algorithms, average_mse, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'], alpha=0.8)

# Adding values on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# Labels and title
plt.xlabel('Regression Algorithm', fontsize=14)
plt.ylabel('Average MSE', fontsize=14)
plt.title('Comparison of Regression Algorithms (Lower is Better)', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Grid and layout adjustments
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, max(average_mse) * 1.2)  # Adjusting y-axis to fit labels

# Show plot
plt.show()
