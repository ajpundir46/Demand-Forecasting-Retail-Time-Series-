import numpy as np

class SimplifiedTree:
    """A small tree to fit residuals (the errors) of the previous trees."""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < 2 or np.var(y) == 0:
            return np.mean(y)

        best_mse = float("inf")
        best_idx, best_thresh = None, None
        for i in range(n_features):
            thresholds = np.unique(X[:, i])
            for t in thresholds:
                l_mask, r_mask = X[:, i] <= t, X[:, i] > t
                if np.sum(l_mask) == 0 or np.sum(r_mask) == 0: continue
                mse = (np.var(y[l_mask]) * len(y[l_mask]) + np.var(y[r_mask]) * len(y[r_mask])) / n_samples
                if mse < best_mse:
                    best_mse, best_idx, best_thresh = mse, i, t

        if best_idx is None: return np.mean(y)

        return {
            "idx": best_idx, "thresh": best_thresh,
            "left": self._grow_tree(X[X[:, best_idx] <= best_thresh], y[X[:, best_idx] <= best_thresh], depth + 1),
            "right": self._grow_tree(X[X[:, best_idx] > best_thresh], y[X[:, best_idx] > best_thresh], depth + 1)
        }

    def predict_one(self, x, tree):
        if not isinstance(tree, dict): return tree
        if x[tree["idx"]] <= tree["thresh"]:
            return self.predict_one(x, tree["left"])
        return self.predict_one(x, tree["right"])

class CustomGradientBoosting:
    """
    Gradient Boosting (XGBoost core) from scratch.
    It builds an additive model where each tree fits the 'residuals' of former trees.
    """
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        # 1. Start with an initial prediction (Mean of target)
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)
        
        print(f"Initial Prediction (Mean): {self.initial_prediction:.2f}")

        for i in range(self.n_estimators):
            # 2. Calculate Residuals (The errors we need to correct)
            # Residual = Actual - Previous Predictions
            residuals = y - y_pred
            
            # 3. Train a tree to predict these residuals
            tree = SimplifiedTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # 4. Update the combined predictions
            # New Pred = Old Pred + (Learning Rate * Tree Prediction)
            tree_preds = np.array([tree.predict_one(x, tree.tree) for x in X])
            y_pred += self.lr * tree_preds
            
            self.trees.append(tree)
            mse = np.mean(residuals**2)
            print(f"Tree {i+1}/{self.n_estimators} trained. Resid MSE: {mse:.4f}")

    def predict(self, X):
        # Final Prediction = Init + sum(Learning Rate * individual tree predictions)
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.lr * np.array([tree.predict_one(x, tree.tree) for x in X])
        return y_pred

# --- Demo ---
if __name__ == "__main__":
    print("--- Custom Gradient Boosting (XGBoost Soul) Demo ---")
    X = np.array([[10, 60], [20, 65], [30, 70], [40, 75], [50, 80]])
    y = np.array([100, 110, 210, 220, 350])

    model = CustomGradientBoosting(n_estimators=20, learning_rate=0.1)
    model.fit(X, y)
    
    test_X = np.array([[35, 72]])
    prediction = model.predict(test_X)
    
    print(f"\nTest Input (Size 35, Temp 72): {test_X.flatten()}")
    print(f"Gradient Boosting Prediction: ${prediction[0]:.2f}")
    print("\nExplanation: Instead of independent trees (Random Forest), Boosting builds trees SEQUENTIALLY. Each tree focuses specifically on fixing the mistakes made by all previous trees combined.")
