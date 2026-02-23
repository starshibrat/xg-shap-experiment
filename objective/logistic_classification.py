from common import common
import numpy as np

class LogisticClassification:
    def __init__(self, label, y_pred):
        self.label = label
        self.p = self.transform(y_pred)
    
    def transform(self, x):
        return common.sigmoid(x)

    def first_order_gradient(self):
        return self.p - self.label

    def second_order_gradient(self):
        eps = 1e-16
        return np.maximum(self.p * (1 - self.p), eps)
