import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
X_data = pd.read_csv(r'/Users/muhammadasad/Desktop/ai lab/4/logisticX.csv', header=None)
Y_data = pd.read_csv(r'/Users/muhammadasad/Desktop/ai lab/4/logisticY.csv', header=None)

# Standardize the feature set
X_standardized = (X_data - X_data.mean()) / X_data.std()

# Add bias term
X_standardized.insert(0, 'Bias', 1)

# Convert data to numpy arrays
X = X_standardized.values
Y = Y_data.values

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute the logistic regression cost
def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = sigmoid(X @ theta)
    cost = (-1 / m) * (Y.T @ np.log(predictions) + (1 - Y).T @ np.log(1 - predictions))
    return cost[0][0]

# Gradient descent algorithm
def gradient_descent(X, Y, theta, alpha, iterations):
    m = len(Y)
    cost_history = []
    
    for _ in range(iterations):
        predictions = sigmoid(X @ theta)
        theta -= (alpha / m) * (X.T @ (predictions - Y))
        cost_history.append(compute_cost(X, Y, theta))
    
    return theta, cost_history

# Set parameters for training
alpha = 0.1
iterations = 1000
theta_initial = np.zeros((X.shape[1], 1))

# Train the model
optimal_theta, cost_values = gradient_descent(X, Y, theta_initial, alpha, iterations)
final_cost = compute_cost(X, Y, optimal_theta)

# Plot cost function over iterations
plt.figure(figsize=(8, 6))
plt.plot(range(1, iterations + 1), cost_values, label=f"Alpha: {alpha}")
plt.title("Cost Function Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost Value")
plt.legend()
plt.grid()
plt.show()

# Visualize decision boundary
plt.figure(figsize=(8, 6))
class_0 = Y.flatten() == 0
class_1 = Y.flatten() == 1
plt.scatter(X[class_0, 1], X[class_0, 2], c='blue', label='Class 0', marker='o')
plt.scatter(X[class_1, 1], X[class_1, 2], c='red', label='Class 1', marker='x')

# Compute and plot decision boundary
x_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
y_values = -(optimal_theta[0] + optimal_theta[1] * x_values) / optimal_theta[2]
plt.plot(x_values, y_values, 'g-', label='Decision Boundary')

plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()

# Train model with different learning rates
learning_rates = [0.1, 5]
plt.figure(figsize=(8, 6))

for lr in learning_rates:
    theta_initial = np.zeros((X.shape[1], 1))
    _, cost_values_lr = gradient_descent(X, Y, theta_initial, lr, 100)
    plt.plot(range(1, 101), cost_values_lr, label=f"Learning Rate: {lr}")

plt.title("Cost Function vs Iterations for Different Learning Rates")
plt.xlabel("Iterations")
plt.ylabel("Cost Function Value")
plt.legend()
plt.grid()
plt.show()

# Evaluate model performance
def evaluate_model(X, Y, theta):
    predictions = sigmoid(X @ theta) >= 0.5
    tp = np.sum((predictions == 1) & (Y == 1))
    tn = np.sum((predictions == 0) & (Y == 0))
    fp = np.sum((predictions == 1) & (Y == 0))
    fn = np.sum((predictions == 0) & (Y == 1))
    
    accuracy = (tp + tn) / len(Y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "Confusion Matrix": [[tn, fp], [fn, tp]],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    }

# Get model metrics
model_metrics = evaluate_model(X, Y, optimal_theta)

# Display evaluation results
model_metrics
