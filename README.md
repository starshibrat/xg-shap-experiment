What's this repository about?
===
XGBoost and TreeSHAP already have their own dedicated library and are integrated nicely. This repository is not in any means to replace but to learn on how they work by implementing the algorithm itself. Also, I'm writing this for my undergraduate thesis. Hope everyone who found this repository helpful.



Status
===
Necessary tests are passed successfully. You can run commands below:
```
make test_indshap
make test_xgc
```
As for official_library_test.py are not considered to be a test for the implementation, since it's for reference that obtained from official library (https://github.com/shap/shap/blob/master/tests/explainers/test_tree.py)


Installing the necessary libraries
===
In your venv or any virtual environment run:
```
pip install -r requirements.txt
```

On notebook
===
notebook directory contains necessary steps for thesis stuffs. In other word, the result of implementation and experiment.
