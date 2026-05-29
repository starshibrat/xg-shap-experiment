import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH1 = os.path.join(
    PROJECT_PATH, "tree"
)

SOURCE_PATH2 = os.path.join(
    PROJECT_PATH, "mini_shap"
)

SOURCE_PATH3 = os.path.join(
    PROJECT_PATH, "datasets/BRFSS"
)

sys.path.append(SOURCE_PATH1)
sys.path.append(SOURCE_PATH2)
sys.path.append(SOURCE_PATH3)