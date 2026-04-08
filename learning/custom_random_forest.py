import numpy as np

# We reuse the Decision Tree logic from before (Simplified for internal use)
class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < 2 or np.var(y) == 0:
            return np.mean(y)

        # Simple split logic for demo (selects best MSE split)
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

class CustomRandomForest:
    """
    Random Forest Regressor from scratch using Bagging (Bootstrap Aggregating).
    """
    def __init__(self, n_trees=5, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        # Randomly pick indices with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            print(f"Training Tree {i+1}/{self.n_trees}...")
            # 1. Create a bootstrap sample of the data
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # 2. Train a fresh decision tree on this sample
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # 3. Aggregate all tree predictions (Average for regression)
        tree_preds = np.array([tree.predict_one(x, tree.root) for x in X for tree in self.trees])
        tree_preds = tree_preds.reshape(len(X), self.n_trees)
        return np.mean(tree_preds, axis=1)

# --- Demo ---
if __name__ == "__main__":
    print("--- Custom Random Forest Demo ---")
    # Simple dataset: Sales increases with Size and Temperature
    X = np.array([[10, 60], [20, 65], [30, 70], [40, 75], [50, 80]])
    y = np.array([100, 110, 210, 220, 350])

    model = CustomRandomForest(n_trees=10, max_depth=3)
    model.fit(X, y)
    
    test_X = np.array([[35, 72]])
    prediction = model.predict(test_X)
    
    print(f"\nTest Input (Size 35, Temp 72): {test_X.flatten()}")
    print(f"Random Forest Prediction: ${prediction[0]:.2f}")
    print("\nExplanation: By averaging 10 different trees trained on random variations of the data, the model is much more stable than a single tree.")
