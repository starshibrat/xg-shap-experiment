import numpy as np
import concurrent.futures
import os
from common import common

class DogwaterTreeExplainer:
    def __init__(self, model, verbose=False):
        self.features = model.features
        self.thresholds = model.thresholds
        self.values = model.values
        self.n_node_samples = model.n_node_samples
        self.weighted_n_node_samples = model.weighted_n_node_samples
        self.children_left = model.children_left
        self.children_right = model.children_right
        self.is_leaf = model.is_leaf
        self.expected_value = self._calc_expected_values()
        self.verbose = verbose

    def _calc_expected_values(self):
        def G(j):
            if self.children_right[j] < 0:
                return self.values[j]

            a = self.children_left[j]
            b = self.children_right[j]

            return (G(a) * self.weighted_n_node_samples[a] + G(b) * self.weighted_n_node_samples[b]) / self.weighted_n_node_samples[j]

        return G(0)


    def _path_len(self, unique_path: np.ndarray) -> np.int64:
        target_index = np.argmin(unique_path, axis=0)[0]
        # print("target_index:", target_index)
        # print("repr_path:", representative_path)

        if unique_path[target_index][0] != -2:
            return len(unique_path)
        
        return target_index

    def _path_append(self, unique_path: np.ndarray, new_path: np.ndarray):
        target_index = np.argmin(unique_path, axis=0)[0]
        # assert that feature_index still not added yet (-2)
        assert unique_path[target_index][0] == -2
        unique_path[target_index] = new_path

    def _path_pop(self, unique_path):
        target_index = np.argmin(unique_path, axis=0)[0] - 1

        if target_index < 0:
            unique_path[-1] = -2
            return

        assert unique_path[target_index][0] != -2
        unique_path[target_index] = -2

    def _vectorized_extend(self, 
        unique_path: np.float32, zero_fraction: np.float32, one_fraction: np.float32, feature_index: int):

        l = self._path_len(unique_path)
        new_pweight = 1.0 if l == 0 else 0.0

        new_path = np.array([feature_index, zero_fraction, one_fraction, new_pweight])

        if self.verbose:
            print(f"\nEXTEND unique_path: {unique_path[:2]}\nWITH zero_frac={zero_fraction}, one_frac={one_fraction}, feature_index={feature_index}, l={l}")
            print(f"\nnew_path={new_path}")
        
        self._path_append(unique_path, new_path)

        for i in range(l - 1, -1, -1):
            pweight_i_plus_one = unique_path[i + 1, 3] + one_fraction * unique_path[i, 3] * (i + 1) / (l + 1)
            pweight_i = unique_path[i, 3] * zero_fraction * (l - i) / (l + 1)
            unique_path[i + 1, 3] = pweight_i_plus_one
            unique_path[i, 3] = pweight_i


        if self.verbose:
            print(f"EXTEND DONE. extended path={unique_path}")
    
    def _vectorized_unwind(self, unique_path: np.ndarray, path_index: int):
        if self.verbose:
            print(f"\nUNWIND unique_path: {unique_path}\nWITH path_index={path_index}")

        l = self._path_len(unique_path) - 1
        n = unique_path[l, 3]

        for j in range(l - 1, -1, -1):
            if unique_path[0, path_index, 2] != 0:
                tmp = unique_path[j, 3]
                unique_path[j, 3] = n * (l + 1) / ((j + 1) * unique_path[path_index, 2])
                n = tmp - unique_path[j, 3] * unique_path[path_index, 1] * (l - j) / (l + 1)
            else:
                unique_path[j, 3] = unique_path[j, 3] * (l + 1) / (unique_path[path_index, 1] * (l - j))
        
        for j in range(path_index, l):
            unique_path[j, :3] = unique_path[j + 1, :3]

        if self.verbose:
            print(f"AFTER UPDATE. path={unique_path}")

        self._path_pop(unique_path)

        if self.verbose:
            print(f"UNWIND DONE. unwound path={unique_path}")


    def _vectorized_unwound_sum(self, unique_path: np.ndarray, index: int) -> np.float32:
        representative_path = unique_path
        depth = self._path_len(unique_path) - 1
        total = 0

        if representative_path[index, 2] != 0:
            n = representative_path[depth, 3]
            for j in range(depth - 1, -1, -1):
                tmp = n / ((j + 1) * representative_path[index, 2])
                total += tmp
                n = representative_path[j, 3] - tmp * representative_path[index, 1] * (depth - j)
        else:
            for j in range(depth - 1, -1, -1):
                total += representative_path[j, 3] / ((depth - j) * representative_path[index, 1])
        
        return total * (depth + 1)

    def _save_path_state(self, unique_path: np.ndarray) -> np.ndarray:
        representative_path = unique_path.copy()

        if self.verbose:
            print(f"saved state={representative_path}")

        return representative_path

    def _restore_path_state(self, unique_path: np.ndarray, saved_path: np.ndarray):
        
        if self.verbose:
            print("\n===RESTORING===")
            print(f"restore state={saved_path}")

        unique_path[:] = saved_path

        if self.verbose:
            print(f"restored state={unique_path[:2]}")


    def _vectorized_recurse(self, active_indices: np.ndarray, X: np.ndarray, node_index: int, unique_path: np.ndarray, zero_fraction: np.float32, one_fraction: np.float32, feature_index: int, phi: np.ndarray):

        if self.verbose:
            print(f"START RECURSE. len(active_indices)= {len(active_indices)}. zero_fraction={zero_fraction}\none_fraction={one_fraction}\nunique_path={unique_path[:2]}\nfeature_index={feature_index}")

        if len(active_indices) == 0:
            return
        
        if zero_fraction == 0:
            return


        self._vectorized_extend(unique_path, zero_fraction, one_fraction, feature_index)

        if self.is_leaf[node_index]:
            if self.verbose:
                print(f"\nCURRENT unique_path: {unique_path}")
                print("UPDATING PHI...")
            for i in range(self._path_len(unique_path)):
                if unique_path[i, 0] == -1:
                    continue

                val = self._vectorized_unwound_sum(unique_path, i) * (unique_path[i, 2] - unique_path[i, 1]) * self.values[node_index]
                if self.verbose:
                    print(f"phi[{active_indices[0]}, {unique_path[i, 0]}] += {val}")

                phi[active_indices, unique_path[i, 0].astype(int)] += val
        else:
            a = self.children_left[node_index]
            b = self.children_right[node_index]

            split_index = self.features[node_index]
            threshold = self.thresholds[node_index]

            go_left_mask = X[active_indices, split_index] <= threshold
            left_indices = active_indices[go_left_mask]
            right_indices = active_indices[~go_left_mask]

            iz, io = 1.0, 1.0
            

            for i in range(self._path_len(unique_path)):
                if self.verbose:
                    print(f"finding unique_path feature_index={split_index}, current={unique_path[i, 0]}")
                if unique_path[i, 0] == split_index:
                    if self.verbose:
                        print(f"unique_path with feature_index={split_index} found. {unique_path[0, i]}")
                    iz = unique_path[i, 1]
                    io = unique_path[i, 2]
                    self._vectorized_unwind(unique_path, i)
                    break

            hot_frac_left = self.weighted_n_node_samples[a] / self.weighted_n_node_samples[node_index]

            cold_frac_left = self.weighted_n_node_samples[b] / self.weighted_n_node_samples[node_index]

            hot_frac_right = self.weighted_n_node_samples[b] / self.weighted_n_node_samples[node_index]

            cold_frac_right = self.weighted_n_node_samples[a] / self.weighted_n_node_samples[node_index]

            if self.verbose:
                print("hot_frac_left:", hot_frac_left)
                print("cold_frac_left:", cold_frac_left)
                print("hot_frac_right:", hot_frac_right)
                print("cold_frac_right:", cold_frac_right)

            checkpoint_path = self._save_path_state(unique_path)

            if self.verbose:
                print("TRAVERSING LEFT NODE...")

            self._vectorized_recurse(left_indices, X, a, unique_path, iz * hot_frac_left, io, split_index, phi)

            self._restore_path_state(unique_path, checkpoint_path) 

            self._vectorized_recurse(left_indices, X, b, unique_path, iz * cold_frac_left, 0.0, split_index, phi)

            if self.verbose:
                print("TRAVERSING RIGHT NODE...")

            self._restore_path_state(unique_path, checkpoint_path) 
            
            self._vectorized_recurse(right_indices, X, b, unique_path, iz * hot_frac_right, io, split_index, phi)

            self._restore_path_state(unique_path, checkpoint_path) 

            self._vectorized_recurse(right_indices, X, a, unique_path, iz * cold_frac_right, 0.0, split_index, phi)
    

    def vectorized_shap_values(self, X: np.ndarray) -> np.ndarray:
        feature_used = len(np.unique(self.features[self.features >= 0]))
        phi = np.zeros_like(X)
        # each unique_path consists of
        # [feature_index, zero_fraction, one_fraction, pweight]
        unique_path = np.full(
            shape=(feature_used + 1, 4),
            fill_value=-2,
            dtype=np.float32
        )

        self._vectorized_recurse(np.arange(0, X.shape[0]), X, 0, unique_path, 1.0, 1.0, -1, phi)

        return phi