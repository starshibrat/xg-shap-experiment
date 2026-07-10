from mini_shap.independent_tree_shap import IndependentTreeExplainer
import unittest
from tree.xg_classifier import XGClassifier
import numpy as np
from datetime import datetime
from common.helper import *
from mini_shap.independent_tree_shap import IndependentTreeExplainer
import shap

class TestTreeSHAP(unittest.TestCase):
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

    def test_single_tree_compare_with_kernel_shap(self):
        """Compare with Kernel SHAP, which makes the same independence assumptions
        as Independent Tree SHAP.  Namely, they both assume independence between the
        set being conditioned on, and the remainder set.
        """

        # FIXME: this test should ideally pass with any random seed. See #2960
        random_seed = 0
        rs = np.random.RandomState(random_seed)

        n_samples = 300

        cat_4_choices = np.array(["0", "1", np.nan], dtype=np.float32)

        df = pd.DataFrame({
            'cat_1': rs.choice(["0", "1", "2"], size=n_samples),
            'cat_2': rs.choice(["0", "1", "2", "3"], size=n_samples),
            'cat_3': rs.choice(["0", "1"], size=n_samples),
            'cat_4': rs.choice(cat_4_choices, size=n_samples, p=[0.4, 0.4, 0.2]),
        })
        
        logits = (df['cat_1'] == 0).astype(float) * 2.0 - \
             (df['cat_2'] == 1).astype(float) * 1.5 + \
             (df['cat_3'] == 1).astype(float) * 0.5 + \
             (df["cat_4"] == 1).astype(float) * 1.2 - \
             (df["cat_4"].isna()).astype(float) * 2.5
        
        for col in df.columns:
            df[col] = df[col].astype(dtype="Int8").astype("str").astype("category")

        probs = 1 / (1 + np.exp(-logits))
        y = rs.binomial(n=1, p=probs)

        model = load_dummy_lib_classifier(1, md=6, ntree=1)
        model.fit(df, y)

        m2 = XGClassifier.load_from_xgbclassifier(model, columns=list(df.columns))

        def f(inp):
            return model.predict(inp, output_margin=True)
        
        background_sample = df.iloc[:50]
        background_sample_np = background_sample.to_numpy(dtype=np.float32)

        expl = IndependentTreeExplainer(m2, refset=background_sample_np, model_output='raw')

        expl_kern = shap.KernelExplainer(f, background_sample)

        ypred = model.predict(df, output_margin=True)
        # Compare for five random samples
        for _ in range(20):
            x_ind = rs.choice(len(df))
            x_sample = df.iloc[x_ind : x_ind + 1]
            x_sample_np = x_sample.to_numpy(dtype=np.float32)

            itshap = expl.full_vectorized_explain(x_sample_np)

            kshap = expl_kern.shap_values(x_sample, nsamples=2000)
            assert np.allclose(itshap, kshap), "Kernel SHAP doesn't match Independent Tree SHAP!"
            assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), "SHAP values don't sum to model output!"

    def test_several_tree_compare_with_kernel_shap(self):
        """Compare with Kernel SHAP, which makes the same independence assumptions
        as Independent Tree SHAP.  Namely, they both assume independence between the
        set being conditioned on, and the remainder set.
        """

        # FIXME: this test should ideally pass with any random seed. See #2960
        random_seed = 0
        rs = np.random.RandomState(random_seed)

        n_samples = 300

        cat_4_choices = np.array(["0", "1", np.nan], dtype=np.float32)

        df = pd.DataFrame({
            'cat_1': rs.choice(["0", "1", "2"], size=n_samples),
            'cat_2': rs.choice(["0", "1", "2", "3"], size=n_samples),
            'cat_3': rs.choice(["0", "1"], size=n_samples),
            'cat_4': rs.choice(cat_4_choices, size=n_samples, p=[0.4, 0.4, 0.2]),
        })
        
        logits = (df['cat_1'] == 0).astype(float) * 2.0 - \
             (df['cat_2'] == 1).astype(float) * 1.5 + \
             (df['cat_3'] == 1).astype(float) * 0.5 + \
             (df["cat_4"] == 1).astype(float) * 1.2 - \
             (df["cat_4"].isna()).astype(float) * 2.5
        
        for col in df.columns:
            df[col] = df[col].astype(dtype="Int8").astype("str").astype("category")

        probs = 1 / (1 + np.exp(-logits))
        y = rs.binomial(n=1, p=probs)

        model = load_dummy_lib_classifier(1, md=6, ntree=5)
        model.fit(df, y)

        m2 = XGClassifier.load_from_xgbclassifier(model, columns=list(df.columns))

        def f(inp):
            return model.predict(inp, output_margin=True)
        
        background_sample = df.iloc[:50]
        background_sample_np = background_sample.to_numpy(dtype=np.float32)

        expl = IndependentTreeExplainer(m2, refset=background_sample_np, model_output='raw')

        expl_kern = shap.KernelExplainer(f, background_sample)

        ypred = model.predict(df, output_margin=True)
        # Compare for five random samples
        for _ in range(20):
            x_ind = rs.choice(len(df))
            x_sample = df.iloc[x_ind : x_ind + 1]
            x_sample_np = x_sample.to_numpy(dtype=np.float32)

            itshap = expl.full_vectorized_explain(x_sample_np)

            kshap = expl_kern.shap_values(x_sample, nsamples=2000)
            assert np.allclose(itshap, kshap), "Kernel SHAP doesn't match Independent Tree SHAP!"
            assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), "SHAP values don't sum to model output!"



if __name__ == '__main__':
    unittest.main()