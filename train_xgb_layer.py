import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, TEST_DATA_VERSION,TEST_DATA_FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
from util.xgb_helpers import learning_rate_schedule, update_learning_rate, LearningRateScheduler
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import json
import os

# winnerB Accuracy: 53.89% | eta: 0.75 | max_depth: 20 | initial_lr: 0.2 | final_lr: 0.1 || Best So Far: Accuracy: 59.64% | eta: 0.01 | max_depth: 13 | initial_lr: 0.19 | final_lr: 0.019

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
SEASONS = [20162017, 20172018, 20182019, 20202021, 20212022, 20222023]
print(len(SEASONS))
# SEASON_DATA = os.listdir(f'training_data/v{VERSION}')
TRAINING_DATAS = [load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib') for season in SEASONS]
# TRAINING_DATA = load(f'training_data/training_data_v{FILE_VERSION}_{START_SEASON}_{END_SEASON}.joblib')
TEST_DATA = load(f'test_data/test_data_v{TEST_DATA_FILE_VERSION}.joblib')



OUTPUT = 'winnerB'

TRIAL = False
DRY_RUN = True

NUM_BOOST_ROUND = 500
N_TRIALS = 100

PARAMS = [
  {
    'max_depth': 5,
    'eta': 0.01,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cuda',
    'tree_method': 'hist',
  },
  {
    'max_depth': 5,
    'eta': 0.01,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cuda',
    'tree_method': 'hist',
  },
  {
    'max_depth': 5,
    'eta': 0.01,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cuda',
    'tree_method': 'hist',
  },
  {
    'max_depth': 23,
    'eta': 0.18,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cuda',
    'tree_method': 'hist',
  },
  {
    'max_depth': 23,
    'eta': 0.18,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cuda',
    'tree_method': 'hist',
  },
  {
    'max_depth': 23,
    'eta': 0.18,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cuda',
    'tree_method': 'hist',
  },
]
EPOCHS = 10
THRESHOLD = 0.5

def train(db,params,x_trains,y_trains,x_test,y_test,trial=False):
  if not trial and not DRY_RUN:
    print('Inputs:', X_INPUTS)
    print('Output:', OUTPUT)
    print('Params:', params)
  
  count = 0
  dtest = xgb.DMatrix(x_test, label=y_test)
  bst = None
  for x_train,y_train,param in zip(x_trains,y_trains,params):
    count += 1
    dtrain = xgb.DMatrix(x_train, label=y_train)
    if bst:
      bst = xgb.train(param, dtrain, EPOCHS, num_boost_round=NUM_BOOST_ROUND, xgb_model=bst)
    else:
      bst = xgb.train(param, dtrain, EPOCHS, num_boost_round=NUM_BOOST_ROUND)
    preds = bst.predict(dtest)

    predictions = [1 if i > 0.5 else 0 for i in preds]

    accuracy = accuracy_score(y_test, predictions)
    print(f'Layer {count} Accuracy: {accuracy*100:.2f}%')
    
  if not trial:
    model_data = f"Accuracy: {'%.2f%%' % (accuracy * 100.0)}"
    print(model_data)

    TrainingRecords = db['dev_training_records']

    timestamp = time.time()
    TrainingRecords.insert_one({
      'savedAt': timestamp,
      'version': VERSION,
      'XGBVersion': XGB_VERSION,
      'testDataVersion': TEST_DATA_VERSION,
      'inputs': X_INPUTS,
      'outputs': Y_OUTPUTS,
      'randomState': RANDOM_STATE,
      'startingSeason': START_SEASON,
      'finalSeason': END_SEASON,
      'model': 'XGBoost Classifier (Layer)',
      'threshold': THRESHOLD,
      'params': params,
      'epochs': EPOCHS,
      'accuracies': {
        OUTPUT: accuracy,
      },
    })
    if not DRY_RUN:
      dump(bst, f'models/nhl_ai_v{XGB_FILE_VERSION}_xgboost_ls_{OUTPUT}.joblib')
  return accuracy

if __name__ == '__main__':
  datas = [pd.DataFrame(training_data) for training_data in TRAINING_DATAS]
  test_data = pd.DataFrame(TEST_DATA)
  datas = [training_data.sort_values(by='id') for training_data in datas]
  test_data = test_data.sort_values(by='id')
  x_train = [data[X_INPUTS] for data in datas]
  y_train = [data[[OUTPUT]].values.ravel() for data in datas]
  x_test = test_data [X_INPUTS]
  y_test = test_data [[OUTPUT]].values.ravel()
  if TRIAL:
    best = {
      'max_depth': 0,
      'eta': 0,
      'accuracy': 0,
    }
    best_initial_lr = 0
    best_final_lr = 0
    for max_depth in range(10,101):
      for eta in np.arange(0.01, 0.91, 0.01):
            params = {
              'max_depth': max_depth,  # the maximum depth of each tree
              'eta': eta,  # the training step for each iteration
              'objective': 'binary:logistic',  # binary classification
              'eval_metric': 'aucpr',  # evaluation metric
              'device': 'cuda',
              'tree_method': 'hist',
            }
            accuracy = train(db,params,x_train,y_train,x_test,y_test,trial=True)
            if accuracy > best['accuracy']:
              best['max_depth'] = max_depth
              best['eta'] = eta
              best['accuracy'] = accuracy
            p_accuracy = f'{OUTPUT} Accuracy:{(accuracy*100):.2f}%|eta:{eta}|max_depth:{max_depth}'
            p_best = f'Best So Far: Accuracy:{(best["accuracy"]*100):.2f}%|eta:{best["eta"]}|max_depth:{best["max_depth"]}'
            print(f'{p_accuracy}||{p_best}')
    best_params = {
      'max_depth': best['max_depth'],
      'eta': best['eta'],
      'objective': 'binary:logistic',
      'eval_metric': params['eval_metric'],
      'device': params['device'],
      'tree_method': params['tree_method'],
    }

    best_params = {
      'max_depth': best['max_depth'],
      'eta': best['eta'],
      'objective': 'binary:logistic',
      'eval_metric': 'aucpr',
      'device': 'cuda',
      'tree_method': 'hist',
    }
    train(db,best_params,x_train,y_train,x_test,y_test,trial=False)
  else:
    train(db,PARAMS,x_train,y_train,x_test,y_test,trial=False)

