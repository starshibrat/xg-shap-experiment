from typing import Tuple
import numpy as np
from tree.param import Param
from tree.node import Node
from mini_shap.dumb_tree_shap import DumbassTreeShap3
from mini_shap.stupid_tree_shap import StupidTreeExplainer
from collections import deque

class VectorizedSingleHistTreeBoost:
    def __init__(self, 
                max_depth: int,
                grad: np.ndarray,
                hess: np.ndarray,
                max_bin: int = 256,
                LAMBDA: np.float32 = 1e-9,
                GAMMA: np.float32 = 1e-9,
                min_child_weight: float = 1.0,
        ):
        self.max_depth = max_depth
        self.grad = grad
        self.hess = hess
        self.tree = None
        self.leaves = None
        self.max_bin = max_bin
        self.LAMBDA = LAMBDA
        self.GAMMA = GAMMA
        self.min_child_weight = min_child_weight
        self.leaves = None

    def _build_whole_histogram(self, binned_X, g, h, num_feature) -> Tuple[np.ndarray, np.ndarray]:
        # feature 1 bin: 0 ~ (max_bin - 1)
        # feature 2 bin: max_bin ~ (max_bin*2 - 1)

        flat_indices = (binned_X + np.arange(num_feature) * self.max_bin).ravel()
        flat_g = np.repeat(g, num_feature)
        flat_h = np.repeat(h, num_feature)

        hist_g = np.bincount(flat_indices,weights=flat_g, minlength=num_feature*self.max_bin).reshape((num_feature, self.max_bin))
        hist_h = np.bincount(flat_indices, weights=flat_h, minlength=num_feature*self.max_bin).reshape((num_feature, self.max_bin))

        return hist_g, hist_h

    def _find_best_split_feature_from_whole_histogram(self, hist_g, hist_h) -> Tuple[np.float32, np.int32]:
        sum_grad = np.cumsum(hist_g, axis=1)
        sum_hess = np.cumsum(hist_h, axis=1)
        g_total = sum_grad[:, -1].reshape(-1, 1)
        h_total = sum_hess[:, -1].reshape(-1, 1)

        sum_g_right = g_total - sum_grad
        sum_h_right = h_total - sum_hess

        score_l = (sum_grad ** 2) / (sum_hess + self.LAMBDA)
        score_r = (sum_g_right ** 2) / (sum_h_right + self.LAMBDA)
        score_total = (g_total ** 2)/(h_total + self.LAMBDA)
        gain = 0.5 * (score_l + score_r - score_total)[:, :-1] - self.GAMMA

        valid_split_mask = (sum_hess >= self.min_child_weight) & (sum_h_right >= self.min_child_weight)
        gain = np.where(valid_split_mask[:, :-1], gain, -np.inf)

        best_gain_feature, best_gain_bin_index = np.unravel_index(np.argmax(gain), shape=gain.shape)

        return gain[best_gain_feature, best_gain_bin_index], best_gain_feature, best_gain_bin_index


    def fit(self, X_binned: np.ndarray, bin_edges: np.ndarray):
        g, h = self.grad, self.hess
        self.tree = Node()
        n_features = X_binned.shape[1]
        hist_g_root, hist_h_root = self._build_whole_histogram(
            X_binned,
            g, h,
            n_features,
        )
        leaves = []

        queue = deque([(self.tree, np.arange(len(X_binned)), 0, hist_g_root, hist_h_root)])

        while queue:
            level_size = len(queue)

            if queue[0][2] > self.max_depth:
                break

            for _ in range(level_size):
                node, sample_indices, depth, hist_g, hist_h = queue.popleft()
                node.n_sample = len(sample_indices)

                if depth == self.max_depth:
                    node.weight = -np.sum(hist_g) / (np.sum(hist_h) + self.LAMBDA)
                    node.sample_indices = sample_indices
                    leaves.append(node)
                    continue

                best_gain, best_gain_feature, best_gain_bin_index = self._find_best_split_feature_from_whole_histogram(hist_g, hist_h)

                if best_gain > 0:
                    node.feature_index = best_gain_feature
                    node.threshold = bin_edges[best_gain_bin_index + 1, best_gain_feature]
                    
                    left_mask = X_binned[sample_indices, best_gain_feature] <= best_gain_bin_index
                    left_indices = sample_indices[left_mask]
                    right_indices = sample_indices[~left_mask]

                    node.left, node.right = Node(), Node()

                    if np.any(left_mask) and np.any(~left_mask):

                        if len(left_indices) < len(right_indices):
                            hist_g_left, hist_h_left = self._build_whole_histogram(
                                X_binned[left_indices],
                                g[left_indices],
                                h[left_indices],
                                n_features,
                            )
                            hist_g_right, hist_h_right = hist_g - hist_g_left, hist_h - hist_h_left
                        else:
                            hist_g_right, hist_h_right = self._build_whole_histogram(
                                X_binned[right_indices],
                                g[right_indices],
                                h[right_indices],
                                n_features,
                            )

                            hist_g_left, hist_h_left = hist_g - hist_g_right, hist_h - hist_h_right

                        queue.append((node.left, left_indices, depth + 1, hist_g_left, hist_h_left))
                        queue.append((node.right, right_indices, depth + 1, hist_g_right, hist_h_right))
                else:
                    node.weight = -np.sum(hist_g) / (np.sum(hist_h) + self.LAMBDA)
                    node.sample_indices = sample_indices
                    leaves.append(node)
        
        self.leaves = leaves

    def tree_to_array(self):
        children_left = []
        children_right = []
        features = []
        thresholds = []
        values = []
        n_node_samples = []
        weighted_n_node_samples = []
        is_leaf = []

        node_count = 0

        def walk(node: Node):
            nonlocal node_count

            node_id = node_count
            node_count += 1

            children_left.append(-1)
            children_right.append(-1)
            features.append(node.feature_index if node.feature_index is not None else -2)
            thresholds.append(node.threshold if node.threshold is not None else -2.0)
            n_node_samples.append(node.n_sample if node.n_sample is not None else 0)
            weighted_n_node_samples.append(float(node.n_sample))
            is_leaf.append(1)

            values.append(node.weight if node.weight is not None else 0.0)

            if not node.is_leaf():
                is_leaf[node_id] = 0
                if node.left:
                    children_left[node_id] = node_count
                    walk(node.left)

                if node.right:
                    children_right[node_id] = node_count
                    walk(node.right)

        walk(self.tree)

        return {
            "children_left": np.array(children_left, dtype=np.intp),
            "children_right": np.array(children_right, dtype=np.intp),
            "features": np.array(features, dtype=np.intp),
            "thresholds": np.array(thresholds, dtype=np.float64),
            "values": np.array(values, dtype=np.float64),
            "n_node_samples": np.array(n_node_samples, dtype=np.intp),
            "weighted_n_node_samples": np.array(weighted_n_node_samples, dtype=np.float64),
            "is_leaf": np.array(is_leaf, dtype=np.bool)
        }

    def _predict(
        self,
        node: Node,
        X: np.ndarray,
    ):
        if node.is_leaf():
            return node.weight

        if X[node.feature_index] <= node.threshold:
            return self._predict(node.left, X)
        else:
            return self._predict(node.right, X)
    
    def predict_train(
        self,
        n_samples: int
    ) -> np.ndarray:
        y_pred = np.zeros(n_samples)

        for leaf in self.leaves:
            y_pred[leaf.sample_indices] = leaf.weight
        
        return y_pred


    def predict(self, X: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=(X.shape[0],), dtype=np.float32)
        for i in range(len(res)):
            res[i] = self._predict(self.tree, X[i])

        return res

    def print_tree(self):
        self._print_tree(self.tree, 1, with_parent=True)

    def _print_tree(self, node, depth, with_parent=False):
        if node is None:
            return

        if with_parent:
            print(str(node))

        tab = "\t" * depth

        if node.left:
            nl = tab + f"-> L: {node.left}"
            print(nl)
            self._print_tree(node.left, depth + 1)

        if node.right:
            nr = tab + f"-> R: {node.right}"
            print(nr)
            self._print_tree(node.right, depth + 1)
    
    def calculate_shap(self, X):
        explainer = DumbassTreeShap3(model=self)
        
        return explainer.shap_values(X), explainer.expected_value
    
    def calculate_shap_vectorized(self, X):
        explainer = StupidTreeExplainer(model=self)

        return explainer.vectorized_shap_values(X), explainer.expected_value

