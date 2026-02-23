from typing import Tuple
import numpy as np
from tree.param import Param
from tree.node import Node
from multiprocessing import Pool, cpu_count

class HistDecisionTreeRegressorParallel:
    def __init__(self, max_depth: int, max_bin: int = 256):
        self.max_depth = max_depth
        self.tree = None
        self.max_bin = max_bin

    def _quantize_features(self, X):
        max_val = np.max(X, axis=0)
        min_val = np.min(X, axis=0)

        width = (max_val - min_val) / self.max_bin

        bin_indices = (X - min_val) / width
        bin_indices = np.clip(
            bin_indices.astype(np.int32),
            a_min=0,
            a_max=self.max_bin - 1,
        )

        # the X become forms of bins
        # subtracted by minus then "rounded" to closest group of bin

        return bin_indices, min_val, width


    def _build_histogram(self, binned_feature, g, h, f_idx) -> Tuple[np.ndarray, int]:
        hist = np.zeros(shape=(self.max_bin, 2))
        np.add.at(hist[:, 0], binned_feature, g)
        np.add.at(hist[:, 1], binned_feature, h)

        return hist, f_idx

    def _find_best_split_feature(self, hist, f_idx) -> Tuple[np.float32, np.int32, int]:
        sum_grad = np.cumsum(hist[:, 0])
        sum_hess = np.cumsum(hist[:, 1])

        g_total = sum_grad[-1]
        h_total = sum_hess[-1]
        ld = 1e-9

        sum_g_right = g_total - sum_grad
        sum_h_right = h_total - sum_hess

        score_l = (sum_grad**2)/(sum_hess + ld)
        score_r = (sum_g_right**2)/(sum_h_right + ld)
        score_total = (g_total**2)/(h_total + ld)

        gain = (0.5 * (score_l + score_r - score_total))[:-1]
        g_i = np.argmax(gain)

        return gain[g_i], g_i, f_idx
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_binned, min_val, width = self._quantize_features(X)
        g = y
        h = np.ones_like(y)

        self.tree = Node()
        queue = [(self.tree, np.arange(len(y)), 0)]
        n_features = X.shape[1]

        with Pool(processes=cpu_count() - 1) as pool:
            while queue:
                level_size = len(queue)

                if queue[0][2] > self.max_depth:
                    break

                for _ in range(level_size):
                    node, sample_indices, depth = queue.pop(0)
                    node.n_sample = len(sample_indices)

                    if len(np.unique(y[sample_indices])) == 1 or depth == self.max_depth:
                        node.weight = sum(g[sample_indices]) / (sum(h[sample_indices]) + 1e-9)
                        continue

                    hists = [
                        self._build_histogram(
                        X_binned[sample_indices, f_idx],
                        g[sample_indices],
                        h[sample_indices],
                        f_idx) for f_idx in range(n_features)]
                    

                    gains = pool.starmap(
                        self._find_best_split_feature,
                        hists,
                    )
                    
                    gains.sort(key=lambda g: g[0], reverse=True)
                    best_gain = gains[0]
                    

                    if best_gain[0] > 0:
                        node.feature_index = best_gain[2]
                        node.threshold = (best_gain[1] + 1) * width[best_gain[2]] + min_val[best_gain[2]]

                        left_mask = X_binned[sample_indices, best_gain[2]] <= best_gain[1]
                        left_indices = sample_indices[left_mask]
                        right_indices = sample_indices[~left_mask]

                        node.left, node.right = Node(), Node()

                        if np.any(left_mask) and np.any(~left_mask):
                            queue.append((node.left, left_indices, depth + 1))
                            queue.append((node.right, right_indices, depth + 1))

                    else:
                        node.weight = np.sum(g[sample_indices]) / (np.sum(h[sample_indices] + 1e-9))

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
