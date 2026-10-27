What's this repository about?
===
SHAP already has it's own dedicated library and works great with XGBoost or any popular models. This reposistory is not by any means an attempt to replace or compete, but to learn on how the algorithm works specifically. Also, the main objective is to do my own research, so you might encounter unrelated stuffs.

Installing the necessary libraries
===
In your virtual environment run:
```
pip install -r requirements.txt
```

Status
===
All necessary tests are passed successfully. You can run the commands below:
```
make test_indshap
make test_xgc
```
Note: official_library_test.py are not considered to be a test case for the implementation, it's just a reference on how the official library test it's implementation.
(https://github.com/shap/shap/blob/master/tests/explainers/test_tree.py)


Modules
===
- **common** : Group of functions that likely to be used more than once in any modules
- **datasets** : Datasets used for experiment
- **mini_shap** : Implementation of independent tree shap
- **model** : Models that has been built
- **notebook** : Experiments itself
- **objective** : Objective for XGBoost
- **tests** : Implementation test cases
- **tree** : Custom implementation of XGBoost and Tree
- **utils** : Group of functions for cleaning dataset, framing and plotting something

About The Experiment
===
The experiment is SHAP values analysis for diabetes prediction. The datasets used are BRFSS 2017, 2019, 2021, and 2023 that got obtained from CDC with the format of .XPT then got converted to parquet, as you can see in init() function (utils/cleaning.py). Then they undergo some cleaning and got unified as one (out_v3.parquet). The unified dataset then used for building XGBoost and applying SHAP value for further analysis. You can view the experiments in /notebook in this order:
- **Dataset cleaning and unification** (brfss.ipynb)
- **Exploratory data analysis** (eda.ipynb)
- **Finding XGBoost's best parameters using grid search and cross validation** (model_building_grid_search.ipynb)
- **Final model building** (model_building_final_model.ipynb)
- **Model metrics evaluation** (model_evaluation.ipynb)
- **SHAP value calculation** (shap.ipynb)
- **SHAP value plotting** (shap_explanation.ipynb)
