import numpy as np

class PathElement:
  def __init__(self,
               feature_index: np.int32=-1,
               zero_fraction: np.float32=-1,
               one_fraction: np.float32=-1,
               pweight: np.float32=-1
  ):
    self.feature_index = feature_index
    self.zero_fraction = zero_fraction
    self.one_fraction = one_fraction
    self.pweight = pweight

  def __repr__(self):
    return f"""PathElement(
feature_index(i) = {self.feature_index},
zero_fraction(z) = {self.zero_fraction},
one_fraction(o) = {self.one_fraction},
pweight(w) = {self.pweight})"""

type Path = list[PathElement]