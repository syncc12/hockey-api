import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Program Files\Graphviz\bin')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import TEST_INPUTS
from constants.constants import VERSION, FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
import json
import optuna


OUTPUT = 'winnerB'

model = load(f'models/nhl_ai_v{XGB_FILE_VERSION}_xgboost_{OUTPUT}_1.joblib')

MAX_FEATURE_COUNT = 150


def test():
  data = pd.DataFrame(TEST_INPUTS, index=[0])
  dtest = xgb.DMatrix(data)

  probability = model.predict(dtest)
  prediction = [1 if i > 0.5 else 0 for i in probability]
  print(prediction,probability)

def plot():
  plt.figure(figsize=(10, 8))
  plot_tree(model, num_trees=1)
  plt.show()

def significance():
  # importance = model.feature_importances_
  # print(importance)
  xgb.plot_importance(model, max_num_features=MAX_FEATURE_COUNT)
  plt.show()

def score():
  feature_importance = model.get_score(importance_type='weight')  # You can also use 'gain' or 'cover'
  print(feature_importance)
  xgb.plot_importance(model, max_num_features=MAX_FEATURE_COUNT)
  plt.show()

def score_weight():
  feature_importance = model.get_score(importance_type='weight')  # You can also use 'gain' or 'cover'
  print(feature_importance)
  xgb.plot_importance(model, max_num_features=MAX_FEATURE_COUNT)
  plt.show()

def score_gain():
  feature_importance = model.get_score(importance_type='gain')  # You can also use 'gain' or 'cover'
  print(feature_importance)
  xgb.plot_importance(model, max_num_features=MAX_FEATURE_COUNT)
  plt.show()

def score_cover():
  feature_importance = model.get_score(importance_type='cover')  # You can also use 'gain' or 'cover'
  print(feature_importance)
  xgb.plot_importance(model, max_num_features=MAX_FEATURE_COUNT)
  plt.show()

# test()
# significance()
# plot()
# score()
score_weight()
score_gain()
score_cover()