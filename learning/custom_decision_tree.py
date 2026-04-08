import numpy as np

class Node:
    """A helper class to represent a split node or a leaf in the tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # If it is a leaf, it stores the predicted value

class CustomDecisionTree:
    """
    Simple Decision Tree Regressor from scratch using MSE as the splitting criterion.
    """
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _calculate_mse(self, y):
        if len(y) == 0: return 0
        return np.mean((y - np.mean(y))**2)

    def _best_split(self, X, y):
        best_mse = float("inf")
        split_idx, split_thresh = None, None
        
        n_samples, n_features = X.shape
        
        # Iterate over every feature and every unique value in that feature
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for thresh in thresholds:
                # Try splitting at this threshold
                left_mask = X_column <= thresh
                right_mask = X_column > thresh
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate Weighted MSE for this split
                y_l, y_r = y[left_mask], y[right_mask]
                mse_l, mse_r = self._calculate_mse(y_l), self._calculate_mse(y_r)
                weighted_mse = (len(y_l)/n_samples)*mse_l + (len(y_r)/n_samples)*mse_r
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    split_idx = feat_idx
                    split_thresh = thresh
                    
        return split_idx, split_thresh

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Stopping criteria: max depth reached, min samples, or pure node (mse=0)
        if (depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) == 0):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        # Find the best split
        idx, thresh = self._best_split(X, y)
        
        if idx is None: # No split possible
            return Node(value=np.mean(y))

        # Split data and grow children recursively
        left_mask = X[:, idx] <= thresh
        right_mask = X[:, idx] > thresh
        
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature_idx=idx, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

# --- Demo with Step Data ---
if __name__ == "__main__":
    print("--- Custom Decision Tree Regressor Demo ---")
    X = np.array([[10], [20], [30], [40], [50], [60]])
    y = np.array([100, 100, 200, 200, 300, 300]) # Step function

    model = CustomDecisionTree(max_depth=3)
    model.fit(X, y)
    
    test_X = np.array([[15], [35], [55]])
    predictions = model.predict(test_X)
    
    print(f"Test Inputs: {test_X.flatten()}")
    print(f"Predictions: {predictions}")
    print("\nExplanation: Decision trees create step-like predictions by partitioning data space into regions.")
