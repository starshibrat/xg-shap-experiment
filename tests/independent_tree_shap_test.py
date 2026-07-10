from mini_shap.independent_tree_shap import IndependentTreeExplainer
import pandas as pd
import unittest
from tree.xg_classifier import XGClassifier
import numpy as np
from datetime import datetime
from common.helper import *
from mini_shap.independent_tree_shap import IndependentTreeExplainer

class TestTreeSHAP(unittest.TestCase):
    # @unittest.skip
    def test_local_accuracy_full_vectorized_1(self):
        X_train, X_test, y_train, y_test, scale_pos_weight = load_data_and_scale_pos_weight(10000, cast_string=True)
        X_train_np, X_test_np, y_train_np, y_test_np = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), y_train.to_numpy().astype(np.float32), y_test.to_numpy().astype(np.float32)

        dt2 = load_dummy_scratch_classifier(scale_pos_weight=scale_pos_weight)

        dt2.fit(X_train_np, y_train_np)

        foreground_sample = X_test_np
        background_sample = X_train_np

        start_time = datetime.now()

        explanation = IndependentTreeExplainer(dt2, refset=background_sample, model_output='raw')
        shap_values = explanation.full_vectorized_explain(foreground_sample)

        time_diff = datetime.now() - start_time

        shap_sum = np.full(shape=len(foreground_sample), dtype=np.float32, fill_value=explanation.expected_value) + shap_values.sum(1)

        print(f"Time taken = {time_diff}")

        raw_score = dt2.predict_raw(X_test_np)

        print(shap_sum[:40])
        print("====")
        print(raw_score[:40])

        index = 0

        for t in np.isclose(shap_sum, raw_score, rtol=1e-2):
          if not t:
            print(f"FAIL TEST: {shap_sum[index]} != {raw_score[index]}")

          index += 1

          self.assertEqual(t, np.True_)

    # @unittest.skip
    def test_local_accuracy_full_vectorized_2(self):

        X_test_np = np.array([[ 1., np.nan,  1.,  1.,  1.,  0.,  1.,  1.,  1., np.nan,  5.,  4.,  0.,
         2.,  0.,  2.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,
        np.nan,  0.,  3.,  1.,  1.,  1.],
         [ 1.,  4.,  1.,  1.,  1.,  1.,  1.,  1.,  0., np.nan,  5.,  0.,  1.,
         3.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  0.,  1.,  1.],
         [ 0., np.nan,  0.,  1.,  0.,  0.,  1.,  1., np.nan, np.nan,  5.,  0.,  1.,
         1.,  0., np.nan,  0.,  1.,  1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,  1.,
         1.,  0., np.nan,  1.,  0.,  1.],
         [ 0.,  4.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
         2.,  1.,  2.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  2.,  0.,  1.,  0.]], dtype=np.float32)

        X_train_np = np.array([[ 1.,  4.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  3.,  0.,  1.,
         3.,  1.,  2.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  0.,  1.,  0.],
         [ 0.,  4.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  3.,  0.,
         2.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  0.,  1.,  0.],
         [ 1.,  4.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  4.,  0.,  1.,
         3.,  0.,  3.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  2.,  0.,  1.,  0.],
         [ 0.,  4.,  0.,  1.,  1.,  1.,  0.,  1.,  1., np.nan,  5.,  0.,  1.,
         2.,  0.,  3.,  1.,  2.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,
         1.,  0.,  2.,  0.,  1.,  0.],
         [ 1.,  3.,  0.,  1.,  1.,  1.,  1.,  1.,  0., np.nan,  5.,  0.,  1.,
         3.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  2.,  0.,  1.,  1.],
         [ 0.,  4.,  1.,  1.,  1.,  1.,  1.,  0.,  1., np.nan,  5.,  0.,  1.,
         3.,  1.,  3.,  0.,  2.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  1.,  0.,  0.],
         [ 0.,  1.,  1.,  1.,  0.,  1.,  0., np.nan,  0.,  0.,  0.,  0.,  0.,
         2.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  2.,  0.,  1.,  0.],
         [ 0.,  4.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  2.,  0.,
         3.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  0.,  1.,  0.],
         [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  4.,  0.,  0.,
         1.,  1.,  3.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  0.,  1.,  0.],
         [ 1., np.nan,  0.,  1.,  1.,  1.,  1.,  0.,  1., np.nan,  5.,  1.,  0.,
         1.,  0.,  3.,  1.,  2.,  0.,  1.,  1.,  1., np.nan,  1.,  1.,  1.,
         1.,  0.,  2.,  1.,  1.,  0.],
         [ 0.,  4.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  4.,  0.,  1.,
         3.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  0.,  1.,  1.],
         [ 1., np.nan,  0.,  1.,  1.,  1.,  1.,  1.,  0., np.nan,  5.,  0.,  1.,
         2.,  0.,  2.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  1.,  1.,  1.],
         [ 1.,  4.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  4.,  0.,
         3.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  0.,  1.,  0.],
         [ 0.,  4.,  1.,  1.,  1.,  1.,  0.,  0., np.nan,  0.,  2.,  0.,  0.,
         3.,  1.,  2.,  0.,  0.,  1.,  1., np.nan, np.nan, np.nan, np.nan, np.nan,  1.,
         1.,  0., np.nan,  0.,  1.,  0.],
         [ 0.,  4.,  1.,  1.,  1.,  1.,  0.,  0.,  0., np.nan,  5.,  0.,  0.,
         3.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  0.,  0.,  0.],
         [ 0.,  4.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  2.,  0.,  1.,
         3.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  0.,  3.,  0.,  1.,  0.]], dtype=np.float32)

        y_train_np = np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
        dtype=np.float32)


        dt2 = load_dummy_scratch_classifier(scale_pos_weight=13/3, md=1)

        dt2.fit(X_train_np, y_train_np)


        foreground_sample = X_test_np
        background_sample = X_train_np

        start_time = datetime.now()
        explanation = IndependentTreeExplainer(dt2, refset=background_sample, model_output='raw')
        shap_values = explanation.full_vectorized_explain(foreground_sample)

        time_diff = datetime.now() - start_time

        shap_sum = np.full(shape=len(foreground_sample), dtype=np.float32, fill_value=explanation.expected_value) + shap_values.sum(1)

        print(f"Time taken = {time_diff}")

        raw_score = dt2.predict_raw(X_test_np)

        print(shap_sum[:40])
        print("====")
        print(raw_score[:40])

        index = 0

        for t in np.isclose(shap_sum, raw_score, rtol=1e-2):
            if not t:
                print(f"FAIL TEST: {shap_sum[index]} != {raw_score[index]}")

            index += 1

            self.assertEqual(t, np.True_)

    def test_conversion_model_shap_value(self):
        X_train, X_test, y_train, y_test, scale_pos_weight = load_data_and_scale_pos_weight(100000, cast_string=True)
        X_train_np, X_test_np, y_train_np, y_test_np = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), y_train.to_numpy().astype(np.float32), y_test.to_numpy().astype(np.float32)

        m1 = load_dummy_lib_classifier(scale_pos_weight, md=4, ntree=50)
        m1.fit(X_train, y_train)

        m2 = XGClassifier.load_from_xgbclassifier(m1, columns=list(X_train.columns))

        pred_raw1 = m1.predict(X_test, output_margin=True)

        indices = np.random.choice(len(X_train_np), 1000, replace=False)

        explanation = IndependentTreeExplainer(m2, refset=X_train_np[indices], model_output='raw')
        shap_sum = explanation.full_vectorized_explain(X_test_np).sum(1) + explanation.expected_value

        index = 0

        for t in np.isclose(pred_raw1, shap_sum, rtol=1e-2):
            if not t:
                print(f"FAIL TEST: {pred_raw1[index]} != {shap_sum[index]}")
            
            index += 1
            self.assertEqual(t, np.True_)


    # def test_local_accuracy_proba_full_vectorized(self):
    #     X_train, X_test, y_train, y_test, scale_pos_weight = load_data_and_scale_pos_weight(100000, cast_string=True)
    #     X_train_np, X_test_np, y_train_np, y_test_np = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), y_train.to_numpy().astype(np.float32), y_test.to_numpy().astype(np.float32)

    #     m1 = load_dummy_lib_classifier(scale_pos_weight, md=4, ntree=50)
    #     m1.fit(X_train, y_train)

    #     m2 = XGClassifier.load_from_xgbclassifier(m1, columns=list(X_train.columns))

    #     pred_raw1 = m1.predict(X_test, output_margin=True)

    #     indices = np.random.choice(len(X_train_np), 1000, replace=False)

    #     base_value, shap_value = m2.calculate_interventional_shap(X_test_np, X_train_np[indices])
    #     shap_sum = shap_value.sum(axis=1) + base_value

    #     index = 0

    #     for t in np.isclose(pred_raw1, shap_sum, rtol=1e-2):
    #         if not t:
    #             print(f"FAIL TEST: {pred_raw1[index]} != {shap_sum[index]}")
            
    #         index += 1
    #         self.assertEqual(t, np.True_)

if __name__ == '__main__':
    unittest.main()