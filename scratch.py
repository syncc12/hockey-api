# import sys
# sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
# sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
# sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

# from pymongo import MongoClient
# from joblib import dump, load
# import warnings
# from util.team_models import PREDICT, PREDICT_H2H, PREDICT_SCORE_H2H, W_MODELS, L_MODELS
# from util.team_helpers import XGBWrapper, XGBWrapperInverse
# # from util.helpers import team_lookup
# from sklearn.metrics import accuracy_score
# from training_input import test_input
# from constants.inputConstants import X_INPUTS_T, Y_OUTPUTS
# import pandas as pd
import numpy as np

# counter = 0
# for w1 in range(1, 51, 1):
#   for w2 in range(1, 51, 1):
#     for w3 in range(1, 51, 1):
#       counter += 1
# print(counter)

# Hypothetical dataset: Each row represents [hyperparameter1, hyperparameter2, accuracy]
data = np.array([
    [0.1, 0.01, 0.9],
    [0.2, 0.02, 0.92],
    # Add more data points
])

X = data[:, :2]  # hyperparameters
y = data[:, 2]   # accuracy

print('X:', X)
print('y:', y)