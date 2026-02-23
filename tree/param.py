import numpy as np

class Param:
    def __init__(self,
        feature_index: int,
        gain: np.float32,
        idx=None,
        feature_value=None,
    ):
        self.feature_index = feature_index
        self.feature_value = feature_value
        self.gain = gain
        self.idx = idx

    def __str__(self):
        return f"feature_index: {self.feature_index} | feature_value: {self.feature_value} | gain: {self.gain}"