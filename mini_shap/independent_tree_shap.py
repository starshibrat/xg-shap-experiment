import numpy as np
from math import factorial
from typing import Tuple

class IndependentTreeExplainer:
    def __init__(self, model, refset=None, verbose=False,):

        if refset is None or len(refset) == 0:
            assert("refset must not be empty!")

        self.features = model.features
        self.thresholds = model.thresholds
        self.values = model.values
        self.n_node_samples = model.n_node_samples
        self.weighted_n_node_samples = model.weighted_n_node_samples
        self.children_left = model.children_left
        self.children_right = model.children_right
        self.is_leaf = model.is_leaf
        self.categorical_features = model.categorical_features
        self.is_categorical = model.is_categorical
        self.missing_go_left = model.missing_go_left
        self.verbose = verbose
        self.refset = refset

    def _calc_weight(self, d: int, k: int):

        res = factorial(k) * factorial(d-k-1)/ factorial(d)

        if self.verbose:
            print(f"\n===CALC_WEIGHT===\n")
            print(f"d={d}, k={k}")
            print(f"res={res}\n")

        return res
    
    def _recurse(self, node_index: int, U: int, V:int, xlist: list, clist: list, x: np.ndarray, c: np.ndarray, phi: np.ndarray) -> Tuple[np.float32, np.float32]:
        if self.verbose:
            print("\n====RECURSE====\n")
            print(f"node_index: {node_index}\nU: {U}, V: {V}\nxlist: {xlist}\nclist: {clist}\n\n")

        if self.is_leaf[node_index]:
            pos, neg = 0, 0
            if U != 0:
                if self.verbose:
                    print("U != 0, calc pos")
                pos = self._calc_weight(V, U - 1) * self.values[node_index]
            if U != V:
                if self.verbose:
                    print("U != V, calc neg")

                neg = -self._calc_weight(V, U) * self.values[node_index]
            
            if self.verbose:
                print(f"(pos, neg)={pos, neg}")
            
            return (pos, neg)
        
        
        split_idx = self.features[node_index]
        threshold = self.thresholds[node_index]

        k = None

        missing_default_left = self.missing_go_left[node_index]

        cond_x_missing = np.isnan(x[split_idx])
        cond_c_missing = np.isnan(c[split_idx])
        cond_x_left = x[split_idx] in threshold
        cond_c_left = c[split_idx] in threshold
        cond_x_right = not(cond_x_left) and not(cond_x_missing)
        cond_c_right = not(cond_c_left) and not(cond_c_missing)

        if self.verbose:
            print(f"node {node_index} = internal")
            print(f"cond_x_missing = {cond_x_missing}")
            print(f"cond_c_missing = {cond_c_missing}")
            print(f"missing -> go {"left" if missing_default_left else "right"}")
            print("### Note: in -> go left | not in and not missing -> go right")
            print(f"cond_x_left = {cond_x_left} | x[{split_idx}] is {"in" if cond_x_left else "not in"} threshold")
            print(f"cond_c_left = {cond_c_left} | c[{split_idx}] is {"in" if cond_c_left else "not in"} threshold")

        a = self.children_right[node_index]
        b = self.children_left[node_index]

        if cond_x_left and cond_c_left:
            k = b
        if cond_x_right and cond_c_right:
            k = a
        
        if cond_x_missing and cond_c_left:
            if missing_default_left:
                k = b
        
        if cond_x_missing and cond_c_right:
            if not missing_default_left:
                k = a
        
        if cond_x_left and cond_c_missing:
            if missing_default_left:
                k = b
        
        if cond_x_right and cond_c_missing:
            if not missing_default_left:
                k = a
        
        if cond_x_missing and cond_c_missing:
            k = b if missing_default_left else a
        
        if xlist[split_idx] > 0:
            if cond_x_missing:
                k = b if missing_default_left else a
            else:
                k = b if cond_x_left else a
        
        if clist[split_idx] > 0:
            if cond_c_missing:
                k = b if missing_default_left else a
            else:
                k = b if cond_c_left else a

        if k is not None:
            if self.verbose:
                print(f"k is not None. k={k}")
                print(f"both use the same k, recurse")

            return self._recurse(k, U, V, xlist, clist, x, c, phi)
        
        if cond_x_left and cond_c_right:
            if self.verbose:
                expr1 = f"x[split_idx]={x[split_idx]} in threshold={threshold}"
                expr2 = f"c[split_idx]={c[split_idx]} in threshold={threshold}"
                print(f"{expr1} && (not ({expr2}) && not mising)")

            xlist[split_idx] += 1
            posx, negx = self._recurse(b, U + 1, V + 1, xlist, clist, x, c, phi)
            xlist[split_idx] -= 1

            clist[split_idx] += 1
            posc, negc = self._recurse(a, U, V + 1, xlist, clist, x, c, phi)
            clist[split_idx] -= 1
        if cond_x_right and cond_c_left:
            if self.verbose:
                expr1 = f"x[split_idx]={x[split_idx]} in threshold={threshold}"
                expr2 = f"c[split_idx]={c[split_idx]} in threshold={threshold}"
                print(f"(not({expr1}) && not missing) && {expr2})")
            xlist[split_idx] += 1
            posx, negx = self._recurse(a, U + 1, V + 1, xlist, clist, x, c, phi)
            xlist[split_idx] -= 1

            clist[split_idx] += 1
            posc, negc = self._recurse(b, U, V + 1, xlist, clist, x, c, phi)
            clist[split_idx] -= 1
        
        if cond_x_missing and cond_c_left:
            if not missing_default_left:
                if self.verbose:
                    expr2 = f"c[split_idx]={c[split_idx]} in threshold={threshold}"
                    print(f"missing (go right) && {expr2}")

                xlist[split_idx] += 1
                posx, negx = self._recurse(a, U + 1, V + 1, xlist, clist, x, c, phi)
                xlist[split_idx] -= 1

                clist[split_idx] += 1
                posc, negc = self._recurse(b, U, V + 1, xlist, clist, x, c, phi)
                clist[split_idx] -= 1
        
        if cond_x_left and cond_c_missing:
            if not missing_default_left:
                if self.verbose:
                    expr1 = f"x[split_idx]={x[split_idx]} in threshold={threshold}"
                    print(f"{expr1} && missing (go right)")

                xlist[split_idx] += 1
                posx, negx = self._recurse(b, U + 1, V + 1, xlist, clist, x, c, phi)
                xlist[split_idx] -= 1

                clist[split_idx] += 1
                posc, negc = self._recurse(a, U, V + 1, xlist, clist, x, c, phi)
                clist[split_idx] -= 1

        if cond_x_missing and cond_c_right:
            if missing_default_left:
                if self.verbose:
                    expr2 = f"c[split_idx]={c[split_idx]} in threshold={threshold}"
                    print(f"missing (go left) && not(({expr2}) && not missing)")

                xlist[split_idx] += 1
                posx, negx = self._recurse(b, U + 1, V + 1, xlist, clist, x, c, phi)
                xlist[split_idx] -= 1

                clist[split_idx] += 1
                posc, negc = self._recurse(a, U, V + 1, xlist, clist, x, c, phi)
                clist[split_idx] -= 1
        
        if cond_x_right and cond_c_missing:
            if missing_default_left:
                if self.verbose:
                    expr2 = f"c[split_idx]={c[split_idx]} in threshold={threshold}"
                    print(f"{expr1} && missing (go left)")

                xlist[split_idx] += 1
                posx, negx = self._recurse(a, U + 1, V + 1, xlist, clist, x, c, phi)
                xlist[split_idx] -= 1

                clist[split_idx] += 1
                posc, negc = self._recurse(b, U, V + 1, xlist, clist, x, c, phi)
                clist[split_idx] -= 1

        if self.verbose:
            print(f"phi [{split_idx}] = {phi[split_idx]} got added {posx + negc}")

        phi[split_idx] += posx + negc
        return (posx + posc, negx + negc)
    
    def explain(self, x: np.ndarray) -> np.ndarray:
        phi = np.zeros(self.refset.shape[1], dtype=np.float32)
        xlist = [0 for _ in range(self.refset.shape[1])]
        clist = [0 for _ in range(self.refset.shape[1])]
        for c in self.refset:
            if self.verbose:
                print(f"\nExplaining foreground sample = {x} using background_sample = {c}...\n")

            self._recurse(0, 0, 0, xlist, clist, x, c, phi)

            if self.verbose:
                print(f"\n state of phi={phi}")

        
        return phi / self.refset.shape[0]
    
    def _half_vectorized_recurse(self, sample_indices: np.ndarray, node_index: int, U: int, V:int, xlist: list, clist: list, X: np.ndarray, c: np.ndarray, phi: np.ndarray) -> Tuple[np.float32, np.float32]:

        n_samples = len(sample_indices)
        if n_samples == 0:
            return 0, 0
        
        if self.is_leaf[node_index]:
            pos, neg = 0.0, 0.0
            if U != 0:
                pos = self._calc_weight(V, U - 1) * self.values[node_index]
            if U != V:
                neg = -self._calc_weight(V, U) * self.values[node_index]
            
            return pos, neg
        
        split_idx = self.features[node_index]
        threshold = self.thresholds[node_index]
        missing_default_left = self.missing_go_left[node_index]

        cond_c_left = c[split_idx] in threshold
        cond_c_missing = np.isnan(c[split_idx])
        c_goes_left = cond_c_left or (cond_c_missing and missing_default_left)

        x_vals = X[sample_indices, split_idx]

        cond_X_missing = np.isnan(x_vals)
        cond_X_left = np.isin(x_vals, threshold)

        if missing_default_left:
            left_mask = cond_X_missing | cond_X_left
        else:
            left_mask = cond_X_left & ~cond_X_missing
        
        right_mask = ~left_mask
        left_indices = sample_indices[left_mask]
        right_indices = sample_indices[right_mask]

        a = self.children_right[node_index]
        b = self.children_left[node_index]

        pos_out = np.zeros(n_samples, dtype=np.float32)
        neg_out = np.zeros(n_samples, dtype=np.float32)

        if clist[split_idx] > 0:
            k = b if c_goes_left else a
            return self._half_vectorized_recurse(sample_indices, k, U, V, xlist, clist, X, c, phi)
        
        if xlist[split_idx] > 0:
            if np.any(left_mask):
                p_left, n_left = self._half_vectorized_recurse(left_indices, b, U, V, xlist, clist, X, c, phi)

                pos_out[left_mask] = p_left
                neg_out[left_mask] = n_left

            if np.any(right_mask):
                p_right, n_right =  self._half_vectorized_recurse(right_indices, a, U, V, xlist, clist, X, c, phi)

                pos_out[right_mask] = p_right
                neg_out[right_mask] = n_right

            return pos_out, neg_out
        
        if c_goes_left:
            if np.any(left_mask):
                pos_l, neg_l = self._half_vectorized_recurse(left_indices, b, U, V, xlist, clist, X, c, phi)
                pos_out[left_mask] = pos_l
                neg_out[left_mask] = neg_l

            if np.any(right_mask):
                xlist[split_idx] += 1
                posx, negx = self._half_vectorized_recurse(right_indices, a, U + 1, V + 1, xlist, clist, X, c, phi)
                xlist[split_idx] -= 1

                clist[split_idx] += 1
                posc, negc = self._half_vectorized_recurse(right_indices, b, U, V + 1, xlist, clist, X, c, phi)
                clist[split_idx] -= 1

                phi[right_indices, split_idx] += posx + negc

                pos_out[right_mask] = posx + posc
                neg_out[right_mask] = negx + negc
        else:
            if np.any(right_mask):
                pos_r, neg_r = self._half_vectorized_recurse(right_indices, a, U, V, xlist, clist, X, c, phi)
                pos_out[right_mask] = pos_r
                neg_out[right_mask] = neg_r
            
            if np.any(left_mask):
                xlist[split_idx] += 1
                posx, negx = self._half_vectorized_recurse(left_indices, b, U + 1, V + 1, xlist, clist, X, c, phi)
                xlist[split_idx] -= 1

                clist[split_idx] += 1
                posc, negc = self._half_vectorized_recurse(left_indices, a, U, V + 1, xlist, clist, X, c, phi)
                clist[split_idx] -= 1

                phi[left_indices, split_idx] += posx + negc

                pos_out[right_mask] = posx + posc
                neg_out[right_mask] = negx + negc

        return pos_out, neg_out
    
    def half_vectorized_explain(self, X: np.ndarray) -> np.ndarray:
        phi = np.zeros_like(X, dtype=np.float32)
        xlist = [0 for _ in range(self.refset.shape[1])]
        clist = [0 for _ in range(self.refset.shape[1])]

        for c in self.refset:
            self._half_vectorized_recurse(np.arange(X.shape[0]), 0, 0, 0, xlist, clist, X, c, phi)
        
        return phi / self.refset.shape[0]