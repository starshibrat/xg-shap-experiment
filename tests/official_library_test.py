import shap
import numpy as np
import xgboost
import sklearn
import unittest
from tree.xg_classifier import XGClassifier

class TestSHAP(unittest.TestCase):
    @unittest.skip
    def test_provided_background_independent(self):
        X, y = shap.datasets.iris()
        # Select the first 100 rows, so that the y values contain only 0s and 1s
        X = X[:100]
        y = y[:100]
        train_x, test_x, train_y, _ = sklearn.model_selection.train_test_split(X, y, random_state=1)
        feature_names = ["a", "b", "c", "d"]
        dtrain = xgboost.DMatrix(train_x, label=train_y, feature_names=feature_names)
        dtest = xgboost.DMatrix(test_x, feature_names=feature_names)

        params = {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "max_depth": 4,
            "eta": 0.1,
            "nthread": -1,
        }

        bst = xgboost.train(params=params, dtrain=dtrain, num_boost_round=100)


        # explainer = shap.TreeExplainer(bst, test_x, feature_perturbation="interventional")

        m = XGClassifier.load_from_booster(bst, columns=feature_names)

        X_test_np = test_x.to_numpy().astype(np.float32)

        pred_raw = m.predict_raw(X_test_np)

        print("pred_raw", pred_raw)

        base_value, shap_values = m.calculate_interventional_shap(X_test_np, X_test_np)

        a = base_value + shap_values.sum(1)
        b = bst.predict(dtest, output_margin=True)

        print("a\n", a)
        print("b\n", b)

        # diffs = explainer.expected_value + explainer.shap_values(test_x).sum(1) - bst.predict(dtest, output_margin=True)

        diffs = a - b

        assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output!"
        assert np.abs(base_value - bst.predict(dtest, output_margin=True).mean()) < 1e-4, (
            "Bad expected_value!"
        )
    
    @unittest.skip
    def test_provided_background_independent_prob_output(self):
        X, y = shap.datasets.iris()
        # Select the first 100 rows, so that the y values contain only 0s and 1s
        X = X[:100]
        y = y[:100]
        train_x, test_x, train_y, _ = sklearn.model_selection.train_test_split(X, y, random_state=1)
        feature_names = ["a", "b", "c", "d"]
        dtrain = xgboost.DMatrix(train_x, label=train_y, feature_names=feature_names)
        dtest = xgboost.DMatrix(test_x, feature_names=feature_names)

        for objective in ["binary:logistic"]:
            params = {
                "booster": "gbtree",
                "objective": objective,
                "max_depth": 4,
                "eta": 0.1,
                "nthread": -1,
            }

            bst = xgboost.train(params=params, dtrain=dtrain, num_boost_round=100)

            explainer = shap.TreeExplainer(bst, test_x, feature_perturbation="interventional", model_output="probability")
            diffs = explainer.expected_value + explainer.shap_values(test_x).sum(1) - bst.predict(dtest)
            assert np.max(np.abs(diffs)) < 1e-4, "SHAP values don't sum to model output!"
            assert np.abs(explainer.expected_value - bst.predict(dtest).mean()) < 1e-4, "Bad expected_value!"

    @unittest.skip
    def test_single_tree_compare_with_kernel_shap(self):
        """Compare with Kernel SHAP, which makes the same independence assumptions
        as Independent Tree SHAP.  Namely, they both assume independence between the
        set being conditioned on, and the remainder set.
        """

        # FIXME: this test should ideally pass with any random seed. See #2960
        random_seed = 0
        rs = np.random.RandomState(random_seed)

        n = 100
        X = rs.normal(size=(n, 7))
        y = np.matmul(X, [-2, 1, 3, 5, 2, 20, -5])

        # train a model with single tree
        Xd = xgboost.DMatrix(X, label=y)
        model = xgboost.train({"eta": 1, "max_depth": 6, "base_score": 0, "lambda": 0}, Xd, 1)
        ypred = model.predict(Xd)

        # Compare for five random samples
        for _ in range(5):
            x_ind = rs.choice(X.shape[1])
            x = X[x_ind : x_ind + 1, :]

            expl = shap.TreeExplainer(model, X, feature_perturbation="interventional")

            def f(inp):
                return model.predict(xgboost.DMatrix(inp))

            expl_kern = shap.KernelExplainer(f, X)

            itshap = expl.shap_values(x)
            kshap = expl_kern.shap_values(x, nsamples=150)
            assert np.allclose(itshap, kshap), "Kernel SHAP doesn't match Independent Tree SHAP!"
            assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), "SHAP values don't sum to model output!"

    @unittest.skip
    def test_several_trees(self):
        """Make sure Independent Tree SHAP sums up to the correct value for
        larger models (20 trees).
        """
        # FIXME: this test should ideally pass with any random seed. See #2960
        random_seed = 0
        rs = np.random.RandomState(random_seed)

        n = 1000
        X = rs.normal(size=(n, 7))
        b = np.array([-2, 1, 3, 5, 2, 20, -5])
        y = np.matmul(X, b)
        max_depth = 6

        # train a model with single tree
        Xd = xgboost.DMatrix(X, label=y)
        model = xgboost.train({"eta": 1, "max_depth": max_depth, "base_score": 0, "lambda": 0}, Xd, 20)
        ypred = model.predict(Xd)

        # Compare for five random samples
        for _ in range(5):
            x_ind = rs.choice(X.shape[1])
            x = X[x_ind : x_ind + 1, :]
            expl = shap.TreeExplainer(model, X, feature_perturbation="interventional")
            itshap = expl.shap_values(x)
            assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), "SHAP values don't sum to model output!"

    @unittest.skip
    def test_single_tree_nonlinear_transformations(self):
        """Make sure Independent Tree SHAP single trees with non-linear
        transformations.
        """
        # Supported non-linear transforms
        # def sigmoid(x):
        #     return(1/(1+np.exp(-x)))

        # def log_loss(yt,yp):
        #     return(-(yt*np.log(yp) + (1 - yt)*np.log(1 - yp)))

        # def mse(yt,yp):
        #     return(np.square(yt-yp))

        # FIXME: this test should ideally pass with any random seed. See #2960
        random_seed = 0

        rs = np.random.RandomState(random_seed)

        n = 100
        X = rs.normal(size=(n, 7))
        y = np.matmul(X, [-2, 1, 3, 5, 2, 20, -5])
        y = y + abs(min(y))
        y = rs.binomial(n=1, p=y / max(y))

        # train a model with single tree
        Xd = xgboost.DMatrix(X, label=y)
        model = xgboost.train(
            {"eta": 1, "max_depth": 6, "base_score": y.mean(), "lambda": 0, "objective": "binary:logistic", "enable_categorical": True}, Xd, 1
        )
        pred = model.predict(Xd, output_margin=True)  # In margin space (log odds)
        trans_pred = model.predict(Xd)  # In probability space

        expl = shap.TreeExplainer(model, X, feature_perturbation="interventional")

        def f(inp):
            return model.predict(xgboost.DMatrix(inp), output_margin=True)

        expl_kern = shap.KernelExplainer(f, X)

        x_ind = 0
        x = X[x_ind : x_ind + 1, :]
        itshap = expl.shap_values(x)
        kshap = expl_kern.shap_values(x, nsamples=300)
        assert np.allclose(itshap.sum() + expl.expected_value, pred[x_ind]), (
            "SHAP values don't sum to model output on explaining margin!"
        )
        assert np.allclose(itshap, kshap), "Independent Tree SHAP doesn't match Kernel SHAP on explaining margin!"

        model.set_attr(objective="binary:logistic")
        expl = shap.TreeExplainer(model, X, feature_perturbation="interventional", model_output="probability")
        itshap = expl.shap_values(x)
        assert np.allclose(itshap.sum() + expl.expected_value, trans_pred[x_ind]), (
            "SHAP values don't sum to model output on explaining logistic!"
        )

        # expl = shap.TreeExplainer(model, X, feature_perturbation="interventional",
        # model_output="logloss")
        # itshap = expl.shap_values(x,y=y[x_ind])
        # margin_pred = model.predict(xgb.DMatrix(x),output_margin=True)
        # currpred = log_loss(y[x_ind],sigmoid(margin_pred))
        # assert np.allclose(itshap.sum(), currpred - expl.expected_value), \
        # "SHAP values don't sum to model output on explaining logloss!"

    @unittest.skip
    def test_xgboost_mixed_types(self):
        X, y = shap.datasets.california(n_points=500)
        X["HouseAge"] = X["HouseAge"].astype(np.int64)
        X["IsOld"] = X["HouseAge"] > 30
        bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 1000)
        shap_values = shap.TreeExplainer(bst).shap_values(X)
        shap.dependence_plot(0, shap_values, X, show=False)

    @unittest.skip
    def test_xgboost_classifier_independent_margin(self):
        # FIXME: this test should ideally pass with any random seed. See #2960
        random_seed = 0

        # train XGBoost model
        rs = np.random.RandomState(random_seed)
        n = 1000
        X = rs.normal(size=(n, 7))
        y = np.matmul(X, [-2, 1, 3, 5, 2, 20, -5])
        y = y + abs(min(y))
        y = rs.binomial(n=1, p=y / max(y))

        model = self.xgboost.XGBClassifier(n_estimators=10, max_depth=5, random_state=random_seed, tree_method="exact")
        model.fit(X, y)
        predicted = model.predict(X, output_margin=True)

        m = XGClassifier.load_from_xgbclassifier(model)

        base_value, shap_values = m.calculate_interventional_shap(X, X)

        base_values = np.full(shape=len(X), fill_value=base_value)

        assert shap_values == (*X.shape,)
        assert base_values.shape == (len(X),)

        # check that SHAP values sum to model output
        assert np.allclose(
            shap_values.sum(1) + base_values,
            predicted,
            atol=1e-7,
        )

    @unittest.skip
    def test_xgboost_classifier_independent_probability(self, random_seed):
        # train XGBoost model
        rs = np.random.RandomState(random_seed)
        n = 1000
        X = rs.normal(size=(n, 7))
        b = np.array([-2, 1, 3, 5, 2, 20, -5])
        y = np.matmul(X, b)
        y = y + abs(min(y))
        y = rs.binomial(n=1, p=y / max(y))

        model = self.xgboost.XGBClassifier(n_estimators=10, max_depth=5, random_state=random_seed)
        model.fit(X, y)
        predicted = model.predict_proba(X)

        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(
            model,
            X,
            feature_perturbation="interventional",
            model_output="probability",
        )
        explanation = explainer(X)
        # check the properties of Explanation object
        assert explanation.values.shape == (*X.shape,)
        assert explanation.base_values.shape == (len(X),)

        # check that SHAP values sum to model output
        assert np.allclose(
            explanation.values.sum(1) + explanation.base_values,
            predicted[:, 1],
        )