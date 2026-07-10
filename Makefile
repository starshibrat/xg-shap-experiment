run_tests:
	python -m unittest discover -s tests -p "*_test.py"

test_indshap:
	python -m unittest discover -s tests -p "independent_tree_shap_test.py"

test_xgc:
	python -m unittest discover -s tests -p "xg_classifier_test.py"