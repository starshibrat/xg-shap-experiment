from typing import Tuple
import numpy as np
from tree.node import Node
from mini_shap.dogwater_tree_shap import DogwaterTreeExplainer
from collections import deque
import graphviz

class SingleTree:
    def __init__(self, 
                max_depth: int,
                grad: np.ndarray,
                hess: np.ndarray,
                LAMBDA: np.float32 = 1e-9,
                GAMMA: np.float32 = 1e-9,
                min_child_weight: float = 1.0,
                categorical_features: list[int] = [],
                verbose: bool = False,
                column_max_val: np.ndarray = None,
        ):
        self.max_depth = max_depth
        self.grad = grad
        self.hess = hess
        self.leaves = None
        self.leaf_sample_indices = None
        self.LAMBDA = LAMBDA
        self.GAMMA = GAMMA
        self.min_child_weight = min_child_weight
        self.verbose = verbose
        self.column_max_val = column_max_val

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
        self.missing_go_left = None

    def _find_best_categorical_split(
        self,
        category_data: np.array,
        g: np.array,
        h: np.array,
        missing_val_code: np.uint8,
    ):
        G_all = np.bincount(category_data, weights=g, minlength=missing_val_code + 1)
        H_all = np.bincount(category_data, weights=h, minlength=missing_val_code + 1)

        G_missing = G_all[missing_val_code]
        H_missing = H_all[missing_val_code]

        G_c = G_all[:missing_val_code]
        H_c = H_all[:missing_val_code]

        valid_mask = H_c > 0
        valid_cats = np.where(valid_mask)[0]

        if len(valid_cats) == 0 and H_missing == 0:
            return 0.0, [], True
        
        G_valid = G_c[valid_mask]
        H_valid = H_c[valid_mask]

        stat = G_valid / (H_valid + self.LAMBDA)
        sort_idx = np.argsort(stat)
        sorted_cats = valid_cats[sort_idx]
        G_sorted = G_valid[sort_idx]
        H_sorted = H_valid[sort_idx]

        G_L_base = np.concatenate(([0.0], np.cumsum(G_sorted)))
        H_L_base = np.concatenate(([0.0], np.cumsum(H_sorted)))

        G_total_obs = G_L_base[-1]
        H_total_obs = H_L_base[-1]

        G_R_base = G_total_obs - G_L_base
        H_R_base = H_total_obs - H_L_base

        G_L_DL = G_L_base + G_missing
        H_L_DL = H_L_base + H_missing
        G_R_DL = G_R_base
        H_R_DL = H_R_base

        G_L_DR = G_L_base
        H_L_DR = H_L_base
        G_R_DR = G_R_base + G_missing
        H_R_DR = H_R_base + H_missing
        
        G_tot_all = G_total_obs + G_missing
        H_tot_all = H_total_obs + H_missing

        def calc_gain(G_L, H_L, G_R, H_R):
            gain_L = (G_L**2) / (H_L + self.LAMBDA)
            gain_R = (G_R**2) / (H_R + self.LAMBDA)
            gain_root = (G_tot_all**2) / (H_tot_all + self.LAMBDA)
            gain = 0.5 * (gain_L + gain_R - gain_root)
            
            # Mask out splits that result in an empty child node
            invalid = (H_L == 0) | (H_R == 0)
            gain[invalid] = -np.inf
            return gain

        gain_DL = calc_gain(G_L_DL, H_L_DL, G_R_DL, H_R_DL)
        gain_DR = calc_gain(G_L_DR, H_L_DR, G_R_DR, H_R_DR)

        best_idx_DL = np.argmax(gain_DL)
        best_gain_DL = gain_DL[best_idx_DL]

        best_idx_DR = np.argmax(gain_DR)
        best_gain_DR = gain_DR[best_idx_DR]

        if best_gain_DL == -np.inf and best_gain_DR == -np.inf:
            return 0.0, [], True
        
        if best_gain_DL > best_gain_DR:
            best_gain = best_gain_DL
            default_left = True
            best_idx = best_idx_DL
        else:
            best_gain = best_gain_DR
            default_left = False
            best_idx = best_idx_DR

        left_categories = sorted_cats[:best_idx].tolist() 

        return best_gain, left_categories, default_left

    
    def clean_up_tree(self):
        self.grad = None
        self.hess = None
        self.leaves = None
        self.leaf_sample_indices = None


    def fit(self, X_binned: np.ndarray):
        g, h = self.grad, self.hess

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
        missing_go_left = []
        leaf_sample_indices = []

        stack = [(
            np.arange(len(X_binned)),
            0, 
            -1,
            False,
        )]
        
        node_count = 0

        while stack:
            sample_indices, depth, parent_id, is_left = stack.pop()
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

            children_left.append(-1)
            children_right.append(-1)
            is_leaf.append(0)
            is_categorical.append(0)
            values.append(0)
            
            weighted_n_node_samples.append(n_sample)
            n_node_samples.append(n_sample)
            features.append(-2)
            thresholds.append(-2.0)
            missing_go_left.append(0)

            if depth == self.max_depth:
                weight = -np.sum(g[sample_indices]) / (np.sum(h[sample_indices]) + self.LAMBDA)
                leaves.append(node_id)
                leaf_sample_indices.append(sample_indices)
                is_leaf[node_id] = 1
                values[node_id] = weight
                continue

            best_gain = 0
            best_feature = 0
            best_threshold = []
            best_default_left = False

            for feat in range(len(self.categorical_features)):
                current_feature = self.categorical_features[feat]

                current_best_gain, left_categories, default_left = self._find_best_categorical_split(
                    X_binned[sample_indices, current_feature],
                    g[sample_indices],
                    h[sample_indices],
                    self.column_max_val[current_feature],
                )

                if current_best_gain > best_gain:
                    best_gain = current_best_gain
                    best_feature = current_feature
                    best_threshold = left_categories
                    best_default_left = default_left
                

            if self.verbose:
                print("====GAIN====")
                print(f"found best_gain={best_gain}\nbest_gain_feature={best_feature}\nbest_threshold={best_threshold}")

            if best_gain > 0:
                missing_go_left[node_id] = best_default_left

                current_feature_data = X_binned[sample_indices, best_feature]

                if best_feature in self.categorical_features:
                    is_categorical[node_id] = 1
                    left_mask = np.isin(current_feature_data, best_threshold)
                
                nan_mask = current_feature_data == self.column_max_val[best_feature]
                valid_mask = ~nan_mask
                    
                base_left_mask = left_mask & valid_mask 
                base_right_mask = (~left_mask) & valid_mask

                if best_default_left:
                    final_left_mask = base_left_mask | nan_mask
                    final_right_mask = base_right_mask
                else:
                    final_left_mask = base_left_mask
                    final_right_mask = base_right_mask | nan_mask
                
                if self.verbose:
                    print(f"Splitted on threshold={best_threshold}\nleft_indices={final_left_mask[:3]}...\nright_indices={final_right_mask[:3]}...")

                left_indices = sample_indices[final_left_mask]
                right_indices = sample_indices[final_right_mask]

                if len(left_indices) > 0 and len(right_indices) > 0:
                    features[node_id] = best_feature
                    thresholds[node_id] = best_threshold
                    
                    stack.append((right_indices, depth + 1, node_id, False))

                    stack.append((left_indices, depth + 1, node_id, True))
                else:
                    weight = -np.sum(g[sample_indices]) / (np.sum(h[sample_indices]) + self.LAMBDA)
                    leaves.append(node_id)
                    leaf_sample_indices.append(sample_indices)
                    is_leaf[node_id] = 1
                    values[node_id] = weight
            else:
                weight = -np.sum(g[sample_indices]) / (np.sum(h[sample_indices]) + self.LAMBDA)
                leaves.append(node_id)
                leaf_sample_indices.append(sample_indices)
                is_leaf[node_id] = 1
                values[node_id] = weight
        
        
        self.children_left = np.array(children_left, dtype=np.intp)
        self.children_right = np.array(children_right, dtype=np.intp)
        self.features = np.array(features, dtype=np.intp)
        self.thresholds = thresholds
        self.values = np.array(values, dtype=np.float32)
        self.n_node_samples = np.array(n_node_samples, dtype=np.uint32)
        self.weighted_n_node_samples = np.array(weighted_n_node_samples, dtype=np.intp)
        self.is_leaf = np.array(is_leaf, dtype=np.bool)
        self.is_categorical = np.array(is_categorical, dtype=np.bool)
        self.missing_go_left = np.array(missing_go_left, dtype=np.bool)
        self.leaves = leaves
        self.leaf_sample_indices = leaf_sample_indices

    def predict_train(
        self,
        n_samples: int
    ) -> np.ndarray:
        y_pred = np.zeros(n_samples)

        for leaf in range(len(self.leaves)):
            y_pred[self.leaf_sample_indices[leaf]] = self.values[self.leaves[leaf]]
        
        return y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=(X.shape[0],), dtype=np.float32)

        queue = deque([(0, np.arange(0, X.shape[0]))])

        while queue:
            node_index, sample_indices = queue.popleft()
            if self.is_leaf[node_index]:
                res[sample_indices] = self.values[node_index]
                continue
            
            threshold = self.thresholds[node_index]
            feature = self.features[node_index]
            nan_mask = np.isnan(X[sample_indices, feature])

            if self.is_categorical[node_index]:
                left_mask = np.isin(X[sample_indices, feature], threshold)
            else:
                left_mask = X[sample_indices, feature] <= threshold
            
            missing_indices = sample_indices[nan_mask]

            left_indices = sample_indices[left_mask & ~nan_mask]
            right_indices = sample_indices[~left_mask & ~nan_mask]

            if self.missing_go_left[node_index]:
                left_indices = np.concatenate((left_indices, missing_indices), axis=0)
            else:
                right_indices = np.concatenate((right_indices, missing_indices), axis=0)

            queue.append((self.children_left[node_index], left_indices))
            queue.append((self.children_right[node_index], right_indices))
        
        return res

    
    def calculate_shap(self, X):
        explainer = DogwaterTreeExplainer(model=self)
        
        return explainer.vectorized_shap_values(X), explainer.expected_value
    
    def plot_custom_tree(self, filename="tree_plot", feature_names=None):
        dot = graphviz.Digraph(name=filename, format='png')
        dot.attr(rankdir='TB')  # Top to Bottom layout
        
        n_nodes = len(self.children_left)
        
        for i in range(n_nodes):
            if self.is_leaf[i]:
                # Format leaf nodes (Boxes, Green)
                val_str = f"\nValue: {self.values[i]:.4f}" if self.values is not None else ""
                label = f"Leaf {i}{val_str}"
                dot.node(str(i), label, shape='box', style='filled', fillcolor='lightgreen')
            else:
                # Format decision nodes (Ellipses, Blue)
                if feature_names is not None and self.features[i] < len(feature_names):
                    feat_name = feature_names[self.features[i]]
                else:
                    feat_name = f"Feature {self.features[i]}"
                
                if self.features[i] in self.categorical_features:
                    label = f"{feat_name} ∈ {self.thresholds[i]}"
                else:
                    label = f"{feat_name} <= {self.thresholds[i]}"

                dot.node(str(i), label, shape='ellipse', style='filled', fillcolor='lightblue')
                
                # Connect to children
                if self.children_left[i] != -1:
                    dot.edge(str(i), str(self.children_left[i]), label=" True")
                if self.children_right[i] != -1:
                    dot.edge(str(i), str(self.children_right[i]), label=" False")
                    
        return dot
    