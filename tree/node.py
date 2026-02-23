class Node:
    def __init__(self, feature_index=None,
        threshold=None,
        left=None,
        right=None,
        weight=None,
        n_sample=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.weight = weight
        self.right = right
        self.left = left
        self.n_sample = n_sample

    def is_leaf(self):
        return self.left is None and self.right is None and self.weight is not None

    def __str__(self):
        if self.is_leaf():
            return f"(weight: {self.weight} | n_sample: {self.n_sample})"

        return f"(feature_index: {self.feature_index} | threshold: {self.threshold} | n_sample: {self.n_sample})"