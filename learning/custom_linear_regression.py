import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CustomLinearRegression:
    """
    Linear Regression implemented from scratch using Gradient Descent.
    Formula: y = w1*x1 + w2*x2 + ... + wn*xn + b
    """
    def __init__(self, learning_rate=0.01, iterations=1000, l2_reg=0.01):
        self.lr = learning_rate
        self.iterations = iterations
        self.l2_reg = l2_reg # lambda for L2 (Ridge) Regularization
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # L2 Loss = Mean Squared Error + (lambda * sum of weights squared)
            cost = (1 / (2 * n_samples)) * (np.sum((y_predicted - y)**2) + self.l2_reg * np.sum(self.weights**2))
            self.cost_history.append(cost)

            # L2 Gradient: dw = (1/n) * X.T * error + (lambda/n) * weights
            dw = (1 / n_samples) * (np.dot(X.T, (y_predicted - y)) + self.l2_reg * self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # iv. Update Weights and Bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # v. Early Stopping (Convergence Detection)
            if i > 0 and abs(self.cost_history[-2] - cost) < 1e-7:
                print(f"Converged at iteration {i}: Cost improvement is negligible.")
                break

            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost:.4f}")

    def predict(self, X):
        """Returns the predicted results."""
        return np.dot(X, self.weights) + self.bias

# --- Example Usage with Sample Data ---
if __name__ == "__main__":
    print("--- Custom Linear Regression Demonstration ---")
    
    # 1. Create a simple synthetic dataset (Sales based on Size)
    # y = 2*x + 5 (Target slope = 2, bias = 5)
    X = np.array([[10], [20], [30], [40], [50]])
    y = np.array([25, 45, 65, 85, 105]) # y = 2*X + 5

    # Feature Scaling (Crucial for Gradient Descent)
    # We normalize X to keep gradients within a manageable range
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_scaled = (X - X_mean) / X_std

    # 2. Initialize and Train
    model = CustomLinearRegression(learning_rate=0.1, iterations=500)
    model.fit(X_scaled, y)

    # 3. Predict for a specific store size
    test_size = np.array([[60]])
    test_scaled = (test_size - X_mean) / X_std
    prediction = model.predict(test_scaled)

    print(f"\nFinal Weights: {model.weights}")
    print(f"Final Bias: {model.bias}")
    print(f"Predicted sales for size 60: ${prediction[0]:.2f} (Expected ~$125)")

    # 4. Optional: Visualize Cost Reduction
    plt.plot(range(len(model.cost_history)), model.cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost (MSE)")
    plt.title("Convergence of Gradient Descent")
    plt.show()
