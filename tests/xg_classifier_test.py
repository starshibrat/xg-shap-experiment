import unittest
from tree.xg_classifier import XGClassifier
import numpy as np
from common.helper import *

class TestXGClassifier(unittest.TestCase):
    def test_conversion_model_raw_prediction(self):
        X_train, X_test, y_train, y_test, scale_pos_weight = load_data_and_scale_pos_weight(10000, cast_string=True)
        X_train_np, X_test_np, y_train_np, y_test_np = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), y_train.to_numpy().astype(np.float32), y_test.to_numpy().astype(np.float32)

        m1 = load_dummy_lib_classifier(scale_pos_weight, md=7, ntree=100)
        m1.fit(X_train, y_train)

        m2 = XGClassifier.load_from_xgbclassifier(m1, columns=list(X_train.columns))

        pred_raw1 = m1.predict(X_test, output_margin=True)
        pred_raw2 = m2.predict_raw(X_test_np)

        index = 0

        for t in np.isclose(pred_raw1, pred_raw2, rtol=1e-2):
            if not t:
                print(f"FAIL TEST: {pred_raw1[index]} != {pred_raw2[index]}")
            
            index += 1
            self.assertEqual(t, np.True_)
    


if __name__ == '__main__':
    unittest.main()