import copy
from mini_shap.path import Path, PathElement
from tree.regressor import HistDecisionTreeRegressor
import numpy as np

class DumbassTreeShap3:
  def __init__(self, model: HistDecisionTreeRegressor, verbose=False):
    self.model = model
    tree_array = model.tree_to_array()
    self.features = tree_array['features']
    self.thresholds = tree_array['thresholds']
    self.values = tree_array['values']
    self.n_node_samples = tree_array['n_node_samples']
    self.weighted_n_node_samples = tree_array['weighted_n_node_samples']
    self.children_left = tree_array['children_left']
    self.children_right = tree_array['children_right']
    self.is_leaf = tree_array['is_leaf']
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

  def _extend(self, unique_path: Path,
              zero_fraction: float,
              one_fraction: float,
              feature_index: int):
    if self.verbose:
      print(f"\nEXTEND unique_path: {unique_path}\nWITH zero_frac={zero_fraction}, one_frac={one_fraction}, feature_index={feature_index}")

    l = len(unique_path)
    unique_path.append(PathElement(
        feature_index=feature_index,
        zero_fraction=zero_fraction,
        one_fraction=one_fraction,
        pweight=(1.0 if l == 0 else 0.0)
    ))
    for i in range(l - 1, -1, -1):
      unique_path[i + 1].pweight += one_fraction * unique_path[i].pweight * (i + 1) / (l + 1)
      unique_path[i].pweight = zero_fraction*unique_path[i].pweight * (l - i) / (l + 1)

    if self.verbose:
      print(f"EXTEND DONE. extended path={unique_path}")


  def _unwind(self, unique_path: Path, path_index: int):
    if self.verbose:
      print(f"\nUNWIND unique_path: {unique_path}\nWITH path_index={path_index}")

    l = len(unique_path) - 1
    n = unique_path[l].pweight

    for j in range(l - 1, -1, -1):
      if unique_path[path_index].one_fraction != 0:
        tmp = unique_path[j].pweight
        unique_path[j].pweight = n * (l + 1) / ((j + 1) * unique_path[path_index].one_fraction)
        n = tmp - unique_path[j].pweight * unique_path[path_index].zero_fraction * (l - j) / (l + 1)
      else:
        unique_path[j].pweight = unique_path[j].pweight * (l + 1) / (unique_path[path_index].zero_fraction * (l - j))

    for j in range(path_index, l):
      unique_path[j].feature_index = unique_path[j+1].feature_index
      unique_path[j].zero_fraction = unique_path[j+1].zero_fraction
      unique_path[j].one_fraction = unique_path[j+1].one_fraction

    unique_path.pop()

    if self.verbose:
      print(f"UNWIND DONE. unwound path={unique_path}")


  def _unwound_sum(self, unique_path: Path, index: int):
    depth = len(unique_path) - 1
    total = 0

    if unique_path[index].one_fraction != 0:
      n = unique_path[depth].pweight
      for j in range(depth - 1, -1, -1):
        tmp = n / ((j + 1) * unique_path[index].one_fraction)
        total += tmp
        n = unique_path[j].pweight - tmp * unique_path[index].zero_fraction * (depth - j)
    else:
      for j in range(depth - 1, -1, -1):
        total += unique_path[j].pweight / ((depth - j) * unique_path[index].zero_fraction)

    return total * (depth + 1)


  def _recurse(self, x: np.ndarray, node_index: int, unique_path: Path,
               zero_fraction: float, one_fraction: float,
               feature_index: int, phi: np.ndarray):

    if zero_fraction == 0:
      return

    self._extend(unique_path, zero_fraction, one_fraction, feature_index)

    if self.is_leaf[node_index]:
      assert len(set(p.feature_index for p in unique_path)) == len(unique_path)
      if self.verbose:
        print(f"\nCURRENT unique_path: {unique_path}")
        print("UPDATING PHI...")
      for i in range(len(unique_path)):
        if unique_path[i].feature_index == -1:
          continue

        val = self._unwound_sum(unique_path, i) * (unique_path[i].one_fraction - unique_path[i].zero_fraction) * self.values[node_index]
        if self.verbose:
          print(f"phi[{unique_path[i].feature_index}] += {val}")
        phi[unique_path[i].feature_index] += val
    else:
      a = self.children_left[node_index]
      b = self.children_right[node_index]

      split_index = self.features[node_index]

      if x[split_index] <= self.thresholds[node_index]:
        h, c = a, b
      else:
        h, c = b, a

      iz, io = 1.0, 1.0

      for i in range(len(unique_path)):
        if unique_path[i].feature_index == split_index:
          iz, io = unique_path[i].zero_fraction, unique_path[i].one_fraction
          self._unwind(unique_path, i)
          break

      hot_frac = self.weighted_n_node_samples[h] / self.weighted_n_node_samples[node_index]
      cold_frac = self.weighted_n_node_samples[c] / self.weighted_n_node_samples[node_index]

      m_copy = copy.deepcopy(unique_path)

      self._recurse(x, h, unique_path, iz * hot_frac, io, split_index, phi)
      self._recurse(x, c, m_copy, iz * cold_frac, 0, split_index, phi)

  def shap_values(self, x: np.ndarray) -> np.ndarray:
    phi = np.zeros_like(x)
    for i in range(len(x)):
      self._recurse(x[i], 0, [], 1.0, 1.0, -1, phi[i])
    return phi