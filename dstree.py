import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _entropy(self, y):
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _information_gain(self, X, y, feature_idx, threshold):
        # Calculate parent entropy
        parent_entropy = self._entropy(y)

        # Split data based on the threshold
        left_idxs = X[:, feature_idx] < threshold
        right_idxs = ~left_idxs

        # Calculate the number of samples in each split
        n_left = np.sum(left_idxs)
        n_right = np.sum(right_idxs)

        # Calculate the entropy of the left and right splits
        left_entropy = self._entropy(y[left_idxs])
        right_entropy = self._entropy(y[right_idxs])

        # Calculate information gain
        information_gain = parent_entropy - (n_left / len(y) * left_entropy) - (n_right / len(y) * right_entropy)
        return information_gain

    def _find_best_split(self, X, y):
        best_gain = 0
        best_feature_idx = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _build_tree(self, X, y, depth=0, max_depth=3):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # If only one class in y or max_depth reached, return the class label
        if len(unique_classes) == 1 or depth == max_depth:
            return unique_classes[0]

        # If no features left or stopping condition reached, return the most frequent class
        if n_features == 0:
            return np.argmax(np.bincount(y))

        # Find the best split
        best_feature_idx, best_threshold = self._find_best_split(X, y)

        # If no split found, return the most frequent class
        if best_feature_idx is None:
            return np.argmax(np.bincount(y))

        # Split the data based on the best split
        left_idxs = X[:, best_feature_idx] < best_threshold
        right_idxs = ~left_idxs

        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_idxs], y[left_idxs], depth + 1, max_depth)
        right_subtree = self._build_tree(X[right_idxs], y[right_idxs], depth + 1, max_depth)

        # Create node dictionary
        node = {
            'feature_index': best_feature_idx,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

        return node

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if isinstance(node, dict):
            if x[node['feature_index']] < node['threshold']:
                return self._predict_one(x, node['left'])
            else:
                return self._predict_one(x, node['right'])
        else:
            return node


# Load data from Excel file
data = pd.read_excel('decesion_tree_data.xlsx')

# Separate features and target variable
X = data.drop('play tennis', axis=1).values  # Features
y = data['play tennis'].values  # Target variable

# Initialize and train decision tree
dt = DecisionTree()
dt.fit(X, y)

# Make predictions
# Assume X_test is your test data
X_test = np.array([[1, 5],
                   [6, 3],
                   [7, 2]])

predictions = dt.predict(X_test)
print("Predictions:", predictions)
