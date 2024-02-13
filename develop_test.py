import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Program Files\Graphviz\bin')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import TEST_INPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import plot_tree, plot_importance
import matplotlib.pyplot as plt
import json
import optuna


OUTPUT = 'winnerB'

model = load(f'models/nhl_ai_v{FILE_VERSION}_xgboost_{OUTPUT}.joblib')

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
  plot_importance(model)
  plt.show()

significance()