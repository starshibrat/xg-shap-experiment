import numpy as np
from math import factorial
from typing import Tuple
from common.common import sigmoid
from tree.xg_classifier import XGClassifier
from tree.single_tree import SingleTree

class IndependentTreeExplainer:
    def __init__(self, model, refset=None, verbose=False, model_output: str='raw'):

        if refset is None or len(refset) == 0:
            assert("refset must not be empty!")

        self.verbose = verbose
        self.refset = refset
        self.model_output = model_output
        self.model = model

        if model_output == 'raw':
            self.expected_value = model.predict_raw(self.refset).mean()
        elif model_output == 'probability':
            self.expected_value = sigmoid(model.predict_raw(self.refset)).mean()

    def _calc_weight(self, d: int, k: int):

        res = factorial(k) * factorial(d-k-1)/ factorial(d)

        if self.verbose:
            print(f"\n===CALC_WEIGHT===\n")
            print(f"d={d}, k={k}")
            print(f"res={res}\n")

        return res
    
    def _full_vectorized_recurse(self, x_indices: np.ndarray, c_indices: np.ndarray, node_index: int, U: int, V:int, xlist: list, clist: list, X: np.ndarray, C: np.ndarray, phi: np.ndarray, cur_tree: SingleTree) -> Tuple[np.float32, np.float32]:

        x_samples = len(x_indices)
        c_samples = len(c_indices)

        if x_samples == 0 or c_samples == 0:
            return 0, 0
        
        if cur_tree.is_leaf[node_index]:
            pos, neg = 0.0, 0.0
            if U != 0:
                pos = self._calc_weight(V, U - 1) * cur_tree.values[node_index]
            if U != V:
                neg = -self._calc_weight(V, U) * cur_tree.values[node_index]

            return pos * c_samples , neg * c_samples
        
        split_idx = cur_tree.features[node_index]
        threshold = cur_tree.thresholds[node_index]
        missing_default_left = cur_tree.missing_go_left[node_index]

        x_vals = X[x_indices, split_idx]
        c_vals = C[c_indices, split_idx]

        cond_C_left = ~np.isin(c_vals, threshold)
        cond_C_missing = np.isnan(c_vals)
        
        cond_X_left = ~np.isin(x_vals, threshold)
        cond_X_missing = np.isnan(x_vals)
        

        if missing_default_left:
            x_left_mask = cond_X_missing | cond_X_left
            c_left_mask = cond_C_missing | cond_C_left
        else:
            x_left_mask = cond_X_left & ~cond_X_missing
            c_left_mask = cond_C_left & ~cond_C_missing

        x_right_mask = ~x_left_mask
        c_right_mask = ~c_left_mask

        x_left_indices = x_indices[x_left_mask]
        x_right_indices = x_indices[x_right_mask]
        c_left_indices = c_indices[c_left_mask]
        c_right_indices = c_indices[c_right_mask]
        
        a = cur_tree.children_right[node_index]
        b = cur_tree.children_left[node_index]

        pos_out = np.zeros(len(x_indices), dtype=np.float32)
        neg_out = np.zeros(len(x_indices), dtype=np.float32)

        if clist[split_idx] > 0:

            if np.any(c_left_mask):
                p, n = self._full_vectorized_recurse(x_indices, c_left_indices, b, U, V, xlist, clist, X, C, phi, cur_tree)

                pos_out += p
                neg_out += n
            
            if np.any(c_right_mask):
                p, n = self._full_vectorized_recurse(x_indices, c_right_indices, a, U, V, xlist, clist, X, C, phi, cur_tree)

                pos_out += p
                neg_out += n
            
            return pos_out, neg_out
        
        if xlist[split_idx] > 0:

            if np.any(x_left_mask):
                p_l, n_l = self._full_vectorized_recurse(x_left_indices, c_left_indices, b, U, V, xlist, clist, X, C, phi, cur_tree)
                p_r, n_r = self._full_vectorized_recurse(x_left_indices, c_right_indices, b, U, V, xlist, clist, X, C, phi, cur_tree)

                pos_out[x_left_mask] += p_l + p_r
                neg_out[x_left_mask] += n_l + n_r

            
            if np.any(x_right_mask):
                p_l, n_l = self._full_vectorized_recurse(x_right_indices, c_left_indices, a, U, V, xlist, clist, X, C, phi, cur_tree)
                p_r, n_r = self._full_vectorized_recurse(x_right_indices, c_right_indices, a, U, V, xlist, clist, X, C, phi, cur_tree)

                pos_out[x_right_mask] += p_r + p_l
                neg_out[x_right_mask] += n_r + n_l


            return pos_out, neg_out
        
        if len(c_left_indices) > 0:
            if len(x_left_indices) > 0:
                p_l, n_l = self._full_vectorized_recurse(x_left_indices, c_left_indices, b, U, V, xlist, clist, X, C, phi, cur_tree)

                pos_out[x_left_mask] += p_l
                neg_out[x_left_mask] += n_l
            
            if len(x_right_indices) > 0:
                xlist[split_idx] += 1
                posx, negx = self._full_vectorized_recurse(x_right_indices, c_left_indices, a, U + 1, V + 1, xlist, clist, X, C, phi, cur_tree)
                xlist[split_idx] -= 1

                clist[split_idx] += 1
                posc, negc = self._full_vectorized_recurse(x_right_indices, c_left_indices, b, U, V + 1, xlist, clist, X, C, phi, cur_tree)
                clist[split_idx] -= 1
                
                phi[x_right_indices, split_idx] += posx + negc
                pos_out[x_right_mask] += posx + posc
                neg_out[x_right_mask] += negx + negc
        
        if len(c_right_indices) > 0:
            if len(x_right_indices) > 0:
                p_r, n_r = self._full_vectorized_recurse(x_right_indices, c_right_indices, a, U, V, xlist, clist, X, C, phi, cur_tree)

                pos_out[x_right_mask] += p_r
                neg_out[x_right_mask] += n_r
            
            if len(x_left_indices) > 0:
                xlist[split_idx] += 1
                posx, negx = self._full_vectorized_recurse(x_left_indices, c_right_indices, b, U + 1, V + 1, xlist, clist, X, C, phi, cur_tree)
                xlist[split_idx] -= 1

                clist[split_idx] += 1
                posc, negc = self._full_vectorized_recurse(x_left_indices, c_right_indices, a, U, V + 1, xlist, clist, X, C, phi, cur_tree)
                clist[split_idx] -= 1

                phi[x_left_indices, split_idx] += posx + negc

                pos_out[x_left_mask] += posx + posc
                neg_out[x_left_mask] += negx + negc
        
        return pos_out, neg_out
    
    def full_vectorized_explain(self, X: np.ndarray) -> np.ndarray:
        shap_values = np.zeros_like(X)

        if isinstance(self.model, XGClassifier):
            for t in self.model.trees:
                xlist = [0 for _ in range(self.refset.shape[1])]
                clist = [0 for _ in range(self.refset.shape[1])]
                phi = np.zeros_like(X, dtype=np.float32)

                self._full_vectorized_recurse(np.arange(X.shape[0]), np.arange(self.refset.shape[0]), 0, 0, 0, xlist, clist, X, self.refset, phi, t)
                shap_values += phi / self.refset.shape[0]
        elif isinstance(self.model, SingleTree):
            xlist = [0 for _ in range(self.refset.shape[1])]
            clist = [0 for _ in range(self.refset.shape[1])]
            phi = np.zeros_like(X, dtype=np.float32)

            self._full_vectorized_recurse(np.arange(X.shape[0]), np.arange(self.refset.shape[0]), 0, 0, 0, xlist, clist, X, self.refset, phi, self.model)

            shap_values = phi / self.refset.shape[0]
        
        return shap_values

    def export_shap_values(self, filename: str=None, shap_values: np.ndarray=None):
        """
        Export the shap_values to .npy file.
        Last row is self.expected_value.
        """

        if shap_values is None:
            assert("shap_values parameter must not empty")

        if filename is None:
            filename = "shap_values.npy"

        arr = np.vstack((shap_values, np.full_like(shap_values[0], self.expected_value)))

        with open(filename, 'wb') as f:
            np.save(f, arr)

    
    