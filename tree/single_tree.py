import numpy as np
from mini_shap.dogwater_tree_shap import DogwaterTreeExplainer
from collections import deque
import graphviz

class SingleTree:
    def __init__(self, 
                max_depth: int,
                grad: np.ndarray,
                hess: np.ndarray,
                reg_lambda: np.float32 = 1e-9,
                min_split_loss: np.float32 = 1e-9,
                min_child_weight: float = 1.0,
                categorical_features: list[int] = [],
                verbose: bool = False,
                column_max_val: np.ndarray = None,
                learning_rate: np.float32 = 0.5,
        ):
        self.max_depth = max_depth
        self.grad = grad
        self.hess = hess
        self.leaves = None
        self.leaf_sample_indices = None
        self.reg_lambda = reg_lambda
        self.min_split_loss = min_split_loss
        self.min_child_weight = min_child_weight
        self.verbose = verbose
        self.column_max_val = column_max_val
        self.learning_rate = learning_rate

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

    # def _find_best_categorical_split(
    #     self,
    #     category_data: np.array,
    #     g: np.array,
    #     h: np.array,
    #     missing_val_code: np.uint8,
    # ):
    #     G_all = np.bincount(category_data, weights=g, minlength=missing_val_code + 1)
    #     H_all = np.bincount(category_data, weights=h, minlength=missing_val_code + 1)

    #     G_missing = G_all[missing_val_code]
    #     H_missing = H_all[missing_val_code]

    #     G_c = G_all[:missing_val_code]
    #     H_c = H_all[:missing_val_code]

    #     valid_mask = H_c > 0
    #     valid_cats = np.where(valid_mask)[0]

    #     if len(valid_cats) == 0 and H_missing == 0:
    #         return 0.0, [], True
        
    #     G_valid = G_c[valid_mask]
    #     H_valid = H_c[valid_mask]

    #     stat = G_valid / (H_valid + self.reg_lambda)
    #     sort_idx = np.argsort(stat)
    #     sorted_cats = valid_cats[sort_idx]
    #     G_sorted = G_valid[sort_idx]
    #     H_sorted = H_valid[sort_idx]

    #     G_L_base = np.concatenate(([0.0], np.cumsum(G_sorted)))
    #     H_L_base = np.concatenate(([0.0], np.cumsum(H_sorted)))

    #     G_total_obs = G_L_base[-1]
    #     H_total_obs = H_L_base[-1]

    #     G_R_base = G_total_obs - G_L_base
    #     H_R_base = H_total_obs - H_L_base

    #     G_L_DL = G_L_base + G_missing
    #     H_L_DL = H_L_base + H_missing
    #     G_R_DL = G_R_base
    #     H_R_DL = H_R_base

    #     G_L_DR = G_L_base
    #     H_L_DR = H_L_base
    #     G_R_DR = G_R_base + G_missing
    #     H_R_DR = H_R_base + H_missing
        
    #     G_tot_all = G_total_obs + G_missing
    #     H_tot_all = H_total_obs + H_missing

    #     def calc_gain(G_L, H_L, G_R, H_R):
    #         gain_L = (G_L**2) / (H_L + self.reg_lambda)
    #         gain_R = (G_R**2) / (H_R + self.reg_lambda)
    #         gain_root = (G_tot_all**2) / (H_tot_all + self.reg_lambda)
    #         gain = 0.5 * (gain_L + gain_R - gain_root)
            
    #         # Mask out splits that result in an empty child node
    #         invalid = (H_L == 0) | (H_R == 0)
    #         gain[invalid] = -np.inf
    #         return gain

    #     gain_DL = calc_gain(G_L_DL, H_L_DL, G_R_DL, H_R_DL)
    #     gain_DR = calc_gain(G_L_DR, H_L_DR, G_R_DR, H_R_DR)

    #     best_idx_DL = np.argmax(gain_DL)
    #     best_gain_DL = gain_DL[best_idx_DL]

    #     best_idx_DR = np.argmax(gain_DR)
    #     best_gain_DR = gain_DR[best_idx_DR]

    #     if best_gain_DL == -np.inf and best_gain_DR == -np.inf:
    #         return 0.0, [], True
        
    #     if best_gain_DL > best_gain_DR:
    #         best_gain = best_gain_DL
    #         default_left = True
    #         best_idx = best_idx_DL
    #     else:
    #         best_gain = best_gain_DR
    #         default_left = False
    #         best_idx = best_idx_DR

    #     left_categories = sorted_cats[:best_idx].tolist() 

    #     return best_gain, left_categories, default_left
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

        valid_mask = H_c > 0.0
        valid_cats = np.where(valid_mask)[0]

        if len(valid_cats) == 0 and H_missing == 0:
            return 0.0, [], True
        
        G_valid = G_c[valid_mask]
        H_valid = H_c[valid_mask]

        stat = G_valid / (H_valid + self.reg_lambda)
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
            gain_L = (G_L**2) / (H_L + self.reg_lambda)
            gain_R = (G_R**2) / (H_R + self.reg_lambda)
            gain_root = (G_tot_all**2) / (H_tot_all + self.reg_lambda)
            gain = 0.5 * (gain_L + gain_R - gain_root)
            
            # Mask out splits that result in an empty child node
            invalid = (H_L  < self.min_child_weight) | (H_R == self.min_child_weight) | (gain <= self.min_split_loss)
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
    
    def _find_best_split(self,
        category_data: np.array,
        g: np.array,
        h: np.array,
        missing_val_code: np.uint8,
    ):
        G_bins = np.bincount(category_data, weights=g, minlength=missing_val_code + 1)
        H_bins = np.bincount(category_data, weights=h, minlength=missing_val_code + 1)

        if self.verbose:
            print("\n===FIND BEST SPLIT INIT===\n")
            print(f"g={g}\n")
            print(f"h={h}\n")
            print(f"categories = {np.arange(missing_val_code + 1)}\n")
            bin_count = np.bincount(category_data, minlength=missing_val_code + 1)
            print(f"bin_count={bin_count}\n")
            print(f"G_bins={G_bins}\n")
            print(f"H_bins={H_bins}\n")

        G_missing = G_bins[missing_val_code]
        H_missing = H_bins[missing_val_code]

        G_bins = G_bins[:missing_val_code]
        H_bins = H_bins[:missing_val_code]

        G_total = np.sum(G_bins)
        H_total = np.sum(H_bins)

        category_score = G_bins / (H_bins + self.reg_lambda)

        sorted_idx = np.argsort(category_score)
        sorted_category = np.arange(missing_val_code)[sorted_idx]

        sorted_G_bins = G_bins[sorted_idx]
        sorted_H_bins = H_bins[sorted_idx]

        if self.verbose:
            print("====STATISTICS CALCULATION====")
            print(f"\nmissing_val_code={missing_val_code}")
            print(f"\nG_bins(non-missing)={G_bins}")
            print(f"\nH_bins(non-missing)={H_bins}")
            print(f"\nG_bins(missing)={G_missing}")
            print(f"\nH_bins(missing)={H_missing}")
            print(f"\nCategory score={category_score}")
            print(f"\nsorted category\n{sorted_category}\n")
            print("\n")

        best_gain = -1
        best_left_categories = None
        best_default_left = False

        def calc_gain(g_l, h_l, g_r, h_r, g_total, h_total):

            if (
                h_l < self.min_child_weight or
                h_r < self.min_child_weight
            ):
                return -1

            score_left = (g_l ** 2) / (h_l + self.reg_lambda)
            score_right = (g_r ** 2) / (h_r + self.reg_lambda)
            score_parent = (g_total ** 2) / (h_total + self.reg_lambda)

            score = 0.5 * (score_left + score_right - score_parent) - self.min_split_loss

            if self.verbose:
                print(f"\n===CALCULATING SCORE OR GAIN===\n")
                print(f"\ng_l={g_l}\nh_l={h_l}\ng_r={g_r}\nh_r={h_r}\ng_total={g_total}\nh_total={h_total}\n")
                print(f"score_left={score_left}")
                print(f"score_right={score_right}")
                print(f"score_parent={score_parent}")

            return score
        
        G_L, H_L = 0, 0

        for j in range(len(sorted_category) - 1):
            if self.verbose:
                print(f"current categories={sorted_category[:j+1]}\n")

            G_L += sorted_G_bins[j]
            H_L += sorted_H_bins[j]

            G_R = G_total - G_L
            H_R = H_total - H_L

            gain_default_left = calc_gain(
                G_L + G_missing,
                H_L + H_missing,
                G_R,
                H_R,
                G_total + G_missing,
                H_total + H_missing,
            )

            if gain_default_left > best_gain:
                best_gain = gain_default_left
                best_left_categories = sorted_category[:j + 1]
                best_default_left = True

            if self.verbose:
                print(f"\ndefault left (G_L + G_missing)\nscore={gain_default_left}\n")
            
            gain_default_right = calc_gain(
                G_L,
                H_L,
                G_R + G_missing,
                H_R + H_missing,
                G_total + G_missing,
                H_total + H_missing
            )

            if gain_default_right > best_gain:
                best_gain = gain_default_right
                best_left_categories = sorted_category[:j + 1]
                best_default_left = False
            
            if self.verbose:
                print(f"\ndefault right (G_R + G_missing)\nscore={gain_default_right}\n")

        if self.verbose:
            print(f"\n===best value from current feature====\n")
            print(f"best_gain={best_gain}")
            print(f"best_left_categories={best_left_categories}")
            print(f"best_default_left={best_default_left}")
            print("\n")

        return (best_gain, best_left_categories, best_default_left)
                
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
                print(f"\nprocessing node_id={node_id}\nwith sample_indices: {sample_indices[:5]}...")

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
            
            weighted_n_node_samples.append(0)
            n_node_samples.append(n_sample)
            features.append(-2)
            thresholds.append(-2.0)
            missing_go_left.append(0)

            if depth == self.max_depth:
                weight = -np.sum(g[sample_indices]) / (np.sum(h[sample_indices]) + self.reg_lambda) * self.learning_rate
                leaves.append(node_id)
                leaf_sample_indices.append(sample_indices)
                is_leaf[node_id] = 1
                values[node_id] = weight
                weighted_n_node_samples[node_id] = np.sum(h[sample_indices])
                continue

            best_gain = 0
            best_feature = None
            best_threshold = []
            best_default_left = False

            for feat in range(len(self.categorical_features)):
                current_feature = self.categorical_features[feat]

                # current_best_gain, left_categories, default_left = self._find_best_categorical_split(
                #     X_binned[sample_indices, current_feature],
                #     g[sample_indices],
                #     h[sample_indices],
                #     self.column_max_val[current_feature],
                # )

                current_best_gain, left_categories, default_left = self._find_best_split(
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

                left_indices = sample_indices[final_left_mask]
                right_indices = sample_indices[final_right_mask]

                if self.verbose:
                    print(f"Splitted on threshold={best_threshold}\nleft_indices={left_indices[:5]}...\nright_indices={right_indices[:5]}...")
                    print(f"left_sample={left_indices.shape[0]}, right_sample={right_indices.shape[0]}")

                if len(left_indices) > 0 and len(right_indices) > 0:
                    features[node_id] = best_feature
                    thresholds[node_id] = best_threshold
                    
                    stack.append((right_indices, depth + 1, node_id, False))

                    stack.append((left_indices, depth + 1, node_id, True))
                else:
                    weight = -np.sum(g[sample_indices]) / (np.sum(h[sample_indices]) + self.reg_lambda) * self.learning_rate
                    leaves.append(node_id)
                    leaf_sample_indices.append(sample_indices)
                    is_leaf[node_id] = 1
                    values[node_id] = weight
                    weighted_n_node_samples[node_id] = np.sum(h[sample_indices])
            else:
                weight = -np.sum(g[sample_indices]) / (np.sum(h[sample_indices]) + self.reg_lambda) * self.learning_rate
                leaves.append(node_id)
                leaf_sample_indices.append(sample_indices)
                is_leaf[node_id] = 1
                values[node_id] = weight
                weighted_n_node_samples[node_id] = np.sum(h[sample_indices])
        
        
        self.children_left = np.array(children_left, dtype=np.intp)
        self.children_right = np.array(children_right, dtype=np.intp)
        self.features = np.array(features, dtype=np.intp)
        self.thresholds = thresholds
        self.values = np.array(values, dtype=np.float32)
        self.n_node_samples = np.array(n_node_samples, dtype=np.uint32)
        self.weighted_n_node_samples = np.array(weighted_n_node_samples, dtype=np.float64)
        self.is_leaf = np.array(is_leaf, dtype=np.bool)
        self.is_categorical = np.array(is_categorical, dtype=np.bool)
        self.missing_go_left = np.array(missing_go_left, dtype=np.bool)
        self.leaves = leaves
        self.leaf_sample_indices = leaf_sample_indices

        self._backfill_weights_and_values(0)

    def _backfill_weights_and_values(self, node_id: int):
        if self.is_leaf[node_id]:
            return self.weighted_n_node_samples[node_id], self.values[node_id]

        left_child = self.children_left[node_id]
        right_child = self.children_right[node_id]

        left_cover, left_val = self._backfill_weights_and_values(left_child)
        right_cover, right_val = self._backfill_weights_and_values(right_child)

        parent_cover = left_cover + right_cover
        self.weighted_n_node_samples[node_id] = parent_cover

        if parent_cover > 0:
            parent_val = (left_val * left_cover + right_val * right_cover) / parent_cover
        else:
            parent_val = 0.0

        self.values[node_id] = parent_val
        
        return parent_cover, parent_val

    def predict_train(
        self,
        n_samples: int
    ) -> np.ndarray:
        y_pred = np.zeros(n_samples)

        if self.verbose:
            print("\n===PREDICT TRAIN===\n")

        for leaf in range(len(self.leaves)):
            y_pred[self.leaf_sample_indices[leaf]] = self.values[self.leaves[leaf]]

            if self.verbose:
                print(f"sample_indices={self.leaf_sample_indices[leaf]}")
                print(f"values = {self.values[self.leaves[leaf]]}\n")

        
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
                
                label += f"\nCover = {self.weighted_n_node_samples[i]:.4f}"

                dot.node(str(i), label, shape='ellipse', style='filled', fillcolor='lightblue')

                default_left = self.missing_go_left[i]
                
                # Connect to children
                if self.children_left[i] != -1:
                    label = " True"
                    if default_left:
                        label += ", missing"

                    dot.edge(str(i), str(self.children_left[i]), label=label)
                if self.children_right[i] != -1:
                    label = " False"
                    if not default_left:
                        label += ", missing"

                    dot.edge(str(i), str(self.children_right[i]), label=label)
                    
        return dot
    