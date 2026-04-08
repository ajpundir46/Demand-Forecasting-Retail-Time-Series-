import numpy as np
import matplotlib.pyplot as plt

class CustomPolynomialRegression:
    """
    Polynomial Regression implemented from scratch.
    It extends Linear Regression by transforming features into X, X^2, ..., X^n.
    """
    def __init__(self, degree=2, learning_rate=0.01, iterations=1000):
        self.degree = degree
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _transform_features(self, X):
        """
        Manually transforms X into [X, X^2, ..., X^degree].
        Example if degree=2: [2] -> [2, 4]
        """
        n_samples = X.shape[0]
        # Initialize an empty matrix for transformed features
        X_poly = np.zeros((n_samples, self.degree))
        
        for d in range(1, self.degree + 1):
            X_poly[:, d-1] = np.power(X, d).flatten()
            
        return X_poly

    def fit(self, X, y):
        # 1. Transform features first
        X_poly = self._transform_features(X)
        
        # 2. Scale features (CRITICAL for polynomial regression as X^n grows very fast)
        self.mean = np.mean(X_poly, axis=0)
        self.std = np.std(X_poly, axis=0)
        X_scaled = (X_poly - self.mean) / self.std
        
        n_samples, n_features = X_scaled.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 3. Gradient Descent
        for i in range(self.iterations):
            y_pred = np.dot(X_scaled, self.weights) + self.bias
            
            cost = (1 / (2 * n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)

            dw = (1 / n_samples) * np.dot(X_scaled.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 500 == 0:
                print(f"Iteration {i}: Cost {cost:.4f}")

    def predict(self, X):
        X_poly = self._transform_features(X)
        X_scaled = (X_poly - self.mean) / self.std
        return np.dot(X_scaled, self.weights) + self.bias

# --- Demo with Noisy Curve Data ---
if __name__ == "__main__":
    print("--- Custom Polynomial Regression (Degree 2) Demo ---")
    
    # 1. Create Curved Data (y = x^2 + random_noise)
    X = np.linspace(-5, 5, 20).reshape(-1, 1)
    y = X.flatten()**2 + np.random.normal(0, 2, 20)
    
    # 2. Train Degree 2 Model
    model = CustomPolynomialRegression(degree=2, learning_rate=0.1, iterations=2000)
    model.fit(X, y)
    
    # 3. Generate Curve for Visualization
    X_plot = np.linspace(-6, 6, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    # 4. Plot Results
    plt.scatter(X, y, color='red', label='Actual Data')
    plt.plot(X_plot, y_plot, color='blue', label='Polynomial Fit (Deg 2)')
    plt.title("Custom Polynomial Regression from Scratch")
    plt.legend()
    plt.show()

    print(f"\nFinal Weights: {model.weights}")
    print(f"Final Bias: {model.bias}")
