import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import X_INPUTS, Y_OUTPUTS, AWAY_FORWARD_INPUTS, AWAY_DEFENSE_INPUTS, AWAY_GOALIE_INPUTS, HOME_FORWARD_INPUTS, HOME_DEFENSE_INPUTS, HOME_GOALIE_INPUTS, BASE_INPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
import xgboost as xgb
import json
import optuna


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]
TRAINING_DATA = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
TEST_DATA = load(f'test_data/test_data_v{FILE_VERSION}.joblib')


OUTPUT = 'winnerB'

TRIAL = True

NUM_BOOST_ROUND = 500
N_TRIALS = 100

PARAMS = {
  'max_depth': 23,  # the maximum depth of each tree
  'eta': 0.18,  # the training step for each iteration
  'objective': 'binary:logistic',  # binary classification
  'eval_metric': 'logloss'  # evaluation metric
}
EPOCHS = 10  # the number of training iterations
THRESHOLD = 0.5

# Best So Far: WinnerB: Accuracy: 60.43% | eta: 0.18 | max_depth: 23 | epochs: 10

# records = []

# f = open('records/xgboost_records.txt', 'a')

def train(db):
  data = pd.DataFrame(TRAINING_DATA)
  test_data = pd.DataFrame(TEST_DATA)
  x_train = data [BASE_INPUTS]
  y_train = data [[OUTPUT]].values.ravel()
  x_test = test_data [BASE_INPUTS]
  y_test = test_data [[OUTPUT]].values.ravel()

  groups = [AWAY_FORWARD_INPUTS, AWAY_DEFENSE_INPUTS, AWAY_GOALIE_INPUTS, HOME_FORWARD_INPUTS, HOME_DEFENSE_INPUTS, HOME_GOALIE_INPUTS]
  names = ['af', 'ad', 'ag', 'hf', 'hd', 'hg']
  
  for group, name in zip(groups, names):
    x_train[f'mean_{name}'] = x_train[group].mean(axis=1)
    x_train[f'std_{name}'] = x_train[group].std(axis=1)
    x_train[f'max_{name}'] = x_train[group].max(axis=1)
    x_train[f'min_{name}'] = x_train[group].min(axis=1)
    x_test[f'mean_{name}'] = x_test[group].mean(axis=1)
    x_test[f'std_{name}'] = x_test[group].std(axis=1)
    x_test[f'max_{name}'] = x_test[group].max(axis=1)
    x_test[f'min_{name}'] = x_test[group].min(axis=1)

  
  # x_train = data.drop(columns=[OUTPUT])
  # y_train = data[OUTPUT]
  # x_test = test_data.drop(columns=[OUTPUT])
  # y_test = test_data[OUTPUT]

  dtrain = xgb.DMatrix(x_train, label=y_train)
  dtest = xgb.DMatrix(x_test, label=y_test)

  bst = xgb.train(PARAMS, dtrain, EPOCHS)

  preds = bst.predict(dtest)

  # Convert probabilities to binary output with a threshold of 0.5
  predictions = [1 if i > THRESHOLD else 0 for i in preds]
  # predictions = [round(i) for i in preds]

  accuracy = accuracy_score(y_test, predictions)
  # model_data = f"Accuracy: {'%.2f%%' % (accuracy * 100.0)} | eta: {eta} | max_depth: {max_depth} | epochs: {epochs}"
  model_data = f"Accuracy: {'%.2f%%' % (accuracy * 100.0)}"
  # records.append(model_data)
  # f.write(f'{model_data}\n')
  print(model_data)

  TrainingRecords = db['dev_training_records']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'savedAt': timestamp,
    'lastTrainedId': TRAINING_DATA[len(TRAINING_DATA)-1]['id'],
    'version': VERSION,
    'inputs': X_INPUTS,
    'outputs': Y_OUTPUTS,
    'randomState': RANDOM_STATE,
    'startingSeason': START_SEASON,
    'finalSeason': END_SEASON,
    'model': 'XGBoost Classifier',
    'fileName': 'train_xgb_set.py',
    'threshold': THRESHOLD,
    'params': PARAMS,
    'epochs': EPOCHS,
    'accuracies': {
      OUTPUT: accuracy,
    },
  })

  
  dump(bst, f'models/nhl_ai_v{FILE_VERSION}_xgboost_set_{OUTPUT}.joblib')


train(db)

