import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CustomLinearRegression:
    """
    Linear Regression implemented from scratch using Gradient Descent.
    Formula: y = w1*x1 + w2*x2 + ... + wn*xn + b
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Trains the model using Gradient Descent.
        X: numpy array of features (n_samples, n_features)
        y: numpy array of target values (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Gradient Descent Loop
        for i in range(self.iterations):
            # i. Predict current values (y_hat)
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # ii. Calculate Cost (Mean Squared Error)
            cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y)**2)
            self.cost_history.append(cost)

            # iii. Compute Gradients
            # dw = (1/n) * X.T * (y_predicted - y)
            # db = (1/n) * sum(y_predicted - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # iv. Update Weights and Bias (Move in opposite direction of gradient)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

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
