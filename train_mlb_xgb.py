import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.constants import MLB_VERSION, FILE_MLB_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pages.mlb.inputs import X_INPUTS_MLB_S, ENCODE_COLUMNS, mlb_training_input, mlb_test_input


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]

TEST_SEASONS = [
  2023,
]
SEASONS = [
  2023,
  2022,
  2021,
  2020,
  2019,
  2018,
  2017,
  2016,

  # 2015,
  # 2014,
  # 2013,
  # 2012,
  # 2011,
  # 2010,
  # 2009,
  # 2008,
  # 2007,
  # 2006,
  # 2005,
  # 2004,
  # 2003,
  # 2002,
  # 2001,
  # 2000,
]


OUTPUT = 'winner'

USE_TEST_SPLIT = False

TRIAL = False
DRY_RUN = False

NUM_BOOST_ROUND = 500
N_TRIALS = 100
EARLY_STOPPING_ROUNDS = 10

PARAMS = {
  'max_depth': 9,
  'eta': 0.31,
  'objective': 'binary:logistic',
  'eval_metric': 'logloss',
  'device': 'cuda',
  'tree_method': 'hist'
}
EPOCHS = 10
THRESHOLD = 0.5

relevance = []


def train(db,params,dtrain,dtest,y_test,trial=False,collect=False):
  if not trial:
    print('Inputs:', X_INPUTS_MLB_S)
    print('Output:', OUTPUT)
    print('Params:', params)

  bst = xgb.train(params, dtrain, EPOCHS)

  preds = bst.predict(dtest)

  predictions = [1 if i > THRESHOLD else 0 for i in preds]

  accuracy = accuracy_score(y_test, predictions)

  if collect:
    relevance.append({
      'accuracy':accuracy,
      'max_depth': params['max_depth'],
      'eta': params['eta'],
    })

  if not trial:
    model_data = f"Accuracy: {'%.2f%%' % (accuracy * 100.0)}"
    print(model_data)

    if not DRY_RUN:
      save_path = f'models/mlb_ai_v{FILE_MLB_VERSION}_xgboost_{OUTPUT}.joblib'
      dump(bst, save_path)

    TrainingRecords = db['dev_training_records']

    timestamp = time.time()
    TrainingRecords.insert_one({
      'sport': 'mlb',
      'savedAt': timestamp,
      'version': MLB_VERSION,
      'inputs': X_INPUTS_MLB_S,
      'randomState': RANDOM_STATE,
      'startingSeason': START_SEASON,
      'finalSeason': END_SEASON,
      'seasons': SEASONS,
      'model': 'XGBoost Classifier',
      'threshold': THRESHOLD,
      'params': PARAMS,
      'epochs': EPOCHS,
      'accuracies': {
        OUTPUT: accuracy,
      },
    })
  return accuracy

if __name__ == '__main__':
  if not USE_TEST_SPLIT:
    SEASONS = [season for season in SEASONS if season not in TEST_SEASONS]
    print('Test Seasons:', TEST_SEASONS)
  print('Training Seasons:', SEASONS)
  TRAINING_DATA = mlb_training_input(SEASONS)
  data = pd.DataFrame(TRAINING_DATA)
  for column in ENCODE_COLUMNS:
    encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
    data = data[data[column] != -1]
    data[column] = encoder.transform(data[column])

  x_train = data [X_INPUTS_MLB_S]
  y_train = data [[OUTPUT]]
  if USE_TEST_SPLIT:
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
    test_df = pd.concat([pd.DataFrame(x_test), y_test.reset_index(drop=True)], axis=1)
    dump(test_df,f'pages/mlb/data/test_data.joblib')
  
  y_train = y_train.values.ravel()
  
  if not USE_TEST_SPLIT:
    TEST_DATA = mlb_test_input(TEST_SEASONS)
    test_data = pd.DataFrame(TEST_DATA)
    dump(test_data,f'pages/mlb/data/test_data.joblib')
    for column in ENCODE_COLUMNS:
      encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
      test_data = test_data[test_data[column] != -1]
      test_data[column] = encoder.transform(test_data[column])

    x_test = test_data [X_INPUTS_MLB_S]
    y_test = test_data [[OUTPUT]].values.ravel()

  dtrain = xgb.DMatrix(x_train, label=y_train)
  dtest = xgb.DMatrix(x_test, label=y_test)

  if TRIAL:

    best = {
      'max_depth': 0,
      'eta': 0,
      'accuracy': 0,
      'training_data': '',
    }
    for max_depth in range(5,31):
      for eta in np.arange(0.01, 0.4, 0.01):
        params = {
          'max_depth': max_depth,
          'eta': eta,
          'objective': 'binary:logistic',
          'eval_metric': 'logloss',
          'device': 'cuda',
          'tree_method': 'hist',
        }
        accuracy = train(db,params,dtrain,dtest,y_test,trial=True)
        if accuracy > best['accuracy']:
          best['max_depth'] = max_depth
          best['eta'] = eta
          best['accuracy'] = accuracy
        p_eta = f'{round(eta,2)}'.ljust(4)
        p_best_eta = f'{round(best["eta"],2)}'.ljust(4)
        p_accuracy = f'{OUTPUT} Accuracy:{(accuracy*100):.2f}%|eta:{p_eta}|max_depth:{max_depth}'
        p_best = f'Best: Accuracy:{(best["accuracy"]*100):.2f}%|eta:{p_best_eta}|max_depth:{best["max_depth"]}'
        print(f'{p_accuracy}||{p_best}')
    best_params = {
      'max_depth': best['max_depth'],
      'eta': best['eta'],
      'objective': 'binary:logistic',
      'eval_metric': params['eval_metric'],
      'device': params['device'],
      'tree_method': params['tree_method'],
    }
    train(db,best_params,dtrain,dtest,y_test,trial=False)
  else:
    train(db,PARAMS,dtrain,dtest,y_test,trial=False)