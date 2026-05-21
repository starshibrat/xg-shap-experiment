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

        missing_cond_x = np.isnan(x[split_idx])
        missing_cond_c = np.isnan(c[split_idx])
        cond_1 = x[split_idx] in threshold
        cond_2 = c[split_idx] in threshold

        if self.verbose:
            print(f"node {node_index} = internal")
            print(f"missing_cond_x = {missing_cond_x}")
            print(f"missing_cond_c = {missing_cond_c}")
            print(f"missing -> go {"left" if missing_default_left else "right"}")
            print("### Note: in -> go left | not in and not missing -> go right")
            print(f"cond_1 = {cond_1} | x[{split_idx}] is {"in" if cond_1 else "not in"} threshold")
            print(f"cond_2 = {cond_2} | c[{split_idx}] is {"in" if cond_2 else "not in"} threshold")

        a = self.children_right[node_index]
        b = self.children_left[node_index]

        if cond_1 and cond_2:
            k = b
        if not(cond_1) and not(cond_2):
            k = a
        
        if missing_cond_x and missing_cond_c:
            k = b if missing_default_left else a
        
        if xlist[split_idx] > 0:
            if missing_cond_x:
                k = b if missing_default_left else a
            else:
                k = b if cond_1 else a
        
        if clist[split_idx] > 0:
            if missing_cond_c:
                k = b if missing_default_left else a
            else:
                k = b if cond_2 else a

        if k is not None:
            if self.verbose:
                print(f"k is not None. k={k}")
                print(f"both use the same k, recurse")

            return self._recurse(k, U, V, xlist, clist, x, c, phi)
        
        if cond_1 and (not(cond_2) and not(missing_cond_c)):
            
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
        if (not(cond_1) and not(missing_cond_x)) and cond_2:
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
        
        if missing_cond_x and cond_2:
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
        
        if cond_1 and missing_cond_c:
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

        if missing_cond_x and (not(cond_2) and not(missing_cond_c)):
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
        
        if (not(cond_1) and not(missing_cond_x)) and missing_cond_c:
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
        