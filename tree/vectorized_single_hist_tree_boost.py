from typing import Tuple
import numpy as np
from tree.node import Node
from mini_shap.dogwater_tree_shap import DogwaterTreeExplainer
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
                categorical_features: list[int] = [],
                verbose: bool = False,
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
        self.verbose = verbose

        self.children_left = None
        self.children_right = None
        self.features = None
        self.thresholds = None
        self.values = None
        self.n_node_samples = None
        self.weighted_n_node_samples = None
        self.is_leaf = None
        self.categorical_features = categorical_features
        self.is_categorical = None

    def _sort_categorical_features(self, hist_g: np.ndarray, hist_h: np.ndarray, num_feature: int):
        if len(self.categorical_features) == 0:
            return None

        bin_categorical_features = np.bincount(self.categorical_features, minlength=num_feature).astype(bool)
        ratios = hist_g[bin_categorical_features]/(hist_h[bin_categorical_features] + 1e-7)
        sort_idx = np.argsort(ratios, axis=1)

        categorical_sort_idx = np.zeros_like(hist_g, dtype=np.uint8)

        rows = np.arange(len(self.categorical_features))[:, None]
        hist_g[bin_categorical_features] = hist_g[bin_categorical_features][rows, sort_idx]
        hist_h[bin_categorical_features] = hist_h[bin_categorical_features][rows, sort_idx]

        categorical_sort_idx[bin_categorical_features] = sort_idx

        if self.verbose:
            print("\n===SORT CATEGORICAL FEATURES===")
            print("hist_g")
            print(hist_g)
            print("hist_h")
            print(hist_h)
            print("categorical_sort_idx")
            print(categorical_sort_idx)


        return categorical_sort_idx


    def _build_whole_histogram(self, binned_X, g, h, num_feature) -> Tuple[np.ndarray, np.ndarray]:
        flat_indices = (binned_X + np.arange(num_feature) * self.max_bin).ravel()
        flat_g = np.repeat(g, num_feature)
        flat_h = np.repeat(h, num_feature)

        hist_g = np.bincount(flat_indices,weights=flat_g, minlength=num_feature*self.max_bin).reshape((num_feature, self.max_bin))
        hist_h = np.bincount(flat_indices, weights=flat_h, minlength=num_feature*self.max_bin).reshape((num_feature, self.max_bin))

        categorical_sort_idx = self._sort_categorical_features(hist_g, hist_h, num_feature)

        return hist_g, hist_h, categorical_sort_idx

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

        if self.verbose:
            print(f"====GAIN ARRAY====")
            print(gain)

        best_gain_feature, best_gain_bin_index = np.unravel_index(np.argmax(gain), shape=gain.shape)

        return gain[best_gain_feature, best_gain_bin_index], best_gain_feature, best_gain_bin_index
    

    def _calculate_left_and_right_histogram(self, X_binned: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray, hist_g: np.ndarray, hist_h: np.ndarray, g: np.ndarray, h: np.ndarray, n_features: np.ndarray):
        if len(left_indices) < len(right_indices):
            if self.verbose:
                print(f"g={g}\nleft_indices={left_indices}")

            hist_g_left, hist_h_left, categorical_sort_idx_left = self._build_whole_histogram(
                X_binned[left_indices],
                g[left_indices],
                h[left_indices],
                n_features,
            )
            hist_g_right, hist_h_right = hist_g - hist_g_left, hist_h - hist_h_left
            categorical_sort_idx_right = self._sort_categorical_features(hist_g_right, hist_h_right, n_features)
        else:
            hist_g_right, hist_h_right, categorical_sort_idx_right = self._build_whole_histogram(
                X_binned[right_indices],
                g[right_indices],
                h[right_indices],
                n_features,
            )

            hist_g_left, hist_h_left = hist_g - hist_g_right, hist_h - hist_h_right
            categorical_sort_idx_left = self._sort_categorical_features(hist_g_left, hist_h_left, n_features)
        
        return hist_g_left, hist_h_left, categorical_sort_idx_left, hist_g_right, hist_h_right, categorical_sort_idx_right

    def fit(self, X_binned: np.ndarray, bin_edges: np.ndarray):
        g, h = self.grad, self.hess
        self.tree = Node()
        n_features = X_binned.shape[1]
        hist_g_root, hist_h_root, categorical_sort_idx_root = self._build_whole_histogram(
            X_binned,
            g, h,
            n_features,
        )

        if self.verbose:
            print("====FITTING START====")
            print(f"categorical features={self.categorical_features}")
            print(f"hist_g_root={hist_g_root}")
            print(f"hist_h_root={hist_h_root}")

        leaves = []
        children_left = []
        children_right = []
        features = []
        thresholds = []
        values = []
        n_node_samples = []
        weighted_n_node_samples = []
        is_leaf = []
        is_categorical = []

        stack = [(
            self.tree,
            np.arange(len(X_binned)),
            0, 
            hist_g_root,
            hist_h_root,
            categorical_sort_idx_root,
            -1,
            False,
        )]
        
        node_count = 0

        while stack:
            node, sample_indices, depth, hist_g, hist_h, categorical_sort_idx, parent_id, is_left = stack.pop()
            node_id = node_count
            node_count += 1

            if self.verbose:
                print(f"\nprocessing node_id={node_id}\nwith sample_indices: {sample_indices[:3]}...")

            if parent_id != -1:
                if is_left:
                    children_left[parent_id] = node_id
                else:
                    children_right[parent_id] = node_id

            n_sample = len(sample_indices)
            node.n_sample = n_sample

            children_left.append(-1)
            children_right.append(-1)
            is_leaf.append(0)
            is_categorical.append(0)
            values.append(0)
            weighted_n_node_samples.append(n_sample)
            n_node_samples.append(n_sample)
            features.append(-2)
            thresholds.append(-2.0)

            if depth == self.max_depth:
                weight = -np.sum(hist_g) / (np.sum(hist_h) + self.LAMBDA)
                node.weight = weight
                node.sample_indices = sample_indices
                leaves.append(node)
                is_leaf[node_id] = 1
                values[node_id] = weight
                continue

            best_gain, best_gain_feature, best_gain_bin_index = self._find_best_split_feature_from_whole_histogram(hist_g, hist_h)

            if self.verbose:
                print("====GAIN====")
                print(f"found best_gain={best_gain}\nbest_gain_feature={best_gain_feature}\nbest_gain_bin_index={best_gain_bin_index}")

            if best_gain > 0:
                if best_gain_feature in self.categorical_features:
                    node.is_categorical = True
                    is_categorical[node_id] = 1
                    mask = hist_h[best_gain_feature, :best_gain_bin_index+1] > 0
                    threshold = categorical_sort_idx[best_gain_feature, :best_gain_bin_index+1][mask]
                    left_mask = np.isin(X_binned[sample_indices, best_gain_feature], threshold)
                else:
                    threshold = bin_edges[best_gain_bin_index + 1, best_gain_feature]
                    left_mask = X_binned[sample_indices, best_gain_feature] <= best_gain_bin_index
                    
                node.threshold = threshold
                node.feature_index = best_gain_feature
                features[node_id] = best_gain_feature
                thresholds[node_id] = threshold

                
                left_indices = sample_indices[left_mask]
                right_indices = sample_indices[~left_mask]
                
                if self.verbose:
                    print(f"Splitted on threshold={threshold}\nleft_indices={left_indices[:3]}...\nright_indices={right_indices[:3]}...")

                node.left, node.right = Node(), Node()

                if np.any(left_mask) and np.any(~left_mask):
                    hist_g_left, hist_h_left, categorical_sort_idx_left, hist_g_right, hist_h_right, categorical_sort_idx_right = self._calculate_left_and_right_histogram(
                        X_binned,
                        left_indices,
                        right_indices,
                        hist_g,
                        hist_h,
                        g,
                        h,
                        n_features,
                    )
                    
                    stack.append((node.right, right_indices, depth + 1, hist_g_right, hist_h_right, categorical_sort_idx_right, node_id, False))

                    stack.append((node.left, left_indices, depth + 1, hist_g_left, hist_h_left, categorical_sort_idx_left, node_id, True))
                else:
                    weight = -np.sum(hist_g) / (np.sum(hist_h) + self.LAMBDA)
                    node.weight = weight
                    node.sample_indices = sample_indices
                    leaves.append(node)
                    is_leaf[node_id] = 1
                    values[node_id] = weight
            else:
                weight = -np.sum(hist_g) / (np.sum(hist_h) + self.LAMBDA)
                node.weight = weight
                node.sample_indices = sample_indices
                leaves.append(node)
                is_leaf[node_id] = 1
                values[node_id] = weight
        
        self.leaves = leaves
        self.children_left = np.array(children_left, dtype=np.intp)
        self.children_right = np.array(children_right, dtype=np.intp)
        self.features = np.array(features, dtype=np.intp)
        self.thresholds = thresholds
        self.values = np.array(values, dtype=np.float32)
        self.n_node_samples = np.array(n_node_samples, dtype=np.uint32)
        self.weighted_n_node_samples = np.array(weighted_n_node_samples, dtype=np.intp)
        self.is_leaf = np.array(is_leaf, dtype=np.bool)
        self.is_categorical = np.array(is_categorical, dtype=np.bool)
    
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

        queue = deque([(0, np.arange(0, X.shape[0]))])

        while queue:
            node_index, sample_indices = queue.popleft()
            if self.is_leaf[node_index]:
                res[sample_indices] = self.values[node_index]
                continue

            if self.is_categorical[node_index]:
                left_mask = np.isin(X[sample_indices, self.features[node_index]], self.thresholds[node_index])
            else:
                left_mask = X[sample_indices, self.features[node_index]] <= self.thresholds[node_index]

            left_indices = sample_indices[left_mask]
            right_indices = sample_indices[~left_mask]

            queue.append((self.children_left[node_index], left_indices))
            queue.append((self.children_right[node_index], right_indices))
        
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
        explainer = DogwaterTreeExplainer(model=self)
        
        return explainer.vectorized_shap_values(X), explainer.expected_value

