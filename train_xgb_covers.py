import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import X_INPUTS_T, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, TEST_DATA_VERSION,TEST_DATA_FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, XGB_TEAM_VERSION, XGB_TEAM_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from xgboost import XGBClassifier
from util.helpers import team_lookup
from training_input import training_input, test_input
from util.xgb_helpers import mcc_eval
from util.team_helpers import away_rename, home_rename

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]
SEASONS = [
  # 20052006,
  # 20062007,
  # 20072008,
  # 20082009,
  # 20092010,
  # 20102011,
  # 20112012,
  # 20122013,
  # 20132014,
  # 20142015,
  # 20152016,
  # 20162017,
  20172018,
  20182019,
  20192020,
  20202021,
  20212022,
  20222023,
]

OUTPUT = 'covers'

TRIAL = True
DRY_RUN = False

NUM_BOOST_ROUND = 500
N_TRIALS = 100
EARLY_STOPPING_ROUNDS = 10
EPOCHS = 10
THRESHOLD = 0.5

GENERAL_PARAMS = {
  'max_depth': 8,  # the maximum depth of each tree
  'eta': 0.08,  # the training step for each iteration
  'objective': 'binary:logistic',  # binary classification
  'eval_metric': 'logloss',  # evaluation metric
  'device': 'cuda',
  'tree_method': 'hist',
}

def train(db,params,dtrain,dtest,inverse=False,trial=False,output=OUTPUT):

  bst = xgb.train(params, dtrain['dm'], EPOCHS)

  preds = bst.predict(dtest['dm'])
  y_test = dtest['y_test']

  predictions = [1 if i < THRESHOLD else 0 for i in preds] if inverse else [1 if i > THRESHOLD else 0 for i in preds]

  accuracy = accuracy_score(y_test, predictions)
  if not trial:
    if accuracy < 0.5:
      inverse = True
      accuracy = 1 - accuracy
    model_data = f"{output} Accuracy: {'%.2f%%' % (accuracy * 100.0)}{' INVERSE' if inverse else ''}"
    print(model_data)

    if not DRY_RUN:
      TrainingRecords = db['dev_training_records']

      timestamp = time.time()
      TrainingRecords.insert_one({
        'savedAt': timestamp,
        'version': VERSION,
        'XGBVersion': XGB_TEAM_VERSION,
        'testDataVersion': TEST_DATA_VERSION,
        'inputs': X_INPUTS_T,
        'outputs': OUTPUT,
        'randomState': RANDOM_STATE,
        'startingSeason': START_SEASON,
        'finalSeason': END_SEASON,
        'seasons': SEASONS,
        'model': 'XGBoost Classifier',
        'file': 'train_xgb_covers.py',
        'threshold': THRESHOLD,
        'params': params,
        'epochs': EPOCHS,
        'accuracies': {
          output: accuracy,
        },
      })

      # path_accuracy = ('%.2f%%' % (accuracy * 100.0)).replace('.','_')
      # save_path = f'models/nhl_ai_v{XGB_FILE_VERSION}_xgboost_{team_name}_{OUTPUT}_{path_accuracy}.joblib'
      save_path = f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_{output}{"_F" if inverse else ""}.joblib'
      dump(bst, save_path)
  return accuracy

if __name__ == '__main__':
  teamLookup = team_lookup(db)
  TRAINING_DATA = training_input(SEASONS)
  data1 = pd.DataFrame(TRAINING_DATA)
  data2 = pd.DataFrame(TRAINING_DATA)

  data1.rename(columns=home_rename, inplace=True)
  data2.rename(columns=away_rename, inplace=True)
  data = pd.concat([data1, data2], axis=0)
  data.reset_index(drop=True, inplace=True)
  all_data = data


  x_train = data [X_INPUTS_T]
  y_train = data [[OUTPUT]].values.ravel()
  dtrain = {'dm':xgb.DMatrix(x_train, label=y_train),'x_train':x_train,'y_train':y_train,'len':len(x_train)}

  TEST_DATA = test_input(X_INPUTS_T,[OUTPUT],no_format=True)
  test_data1 = pd.DataFrame(TEST_DATA)
  test_data2 = pd.DataFrame(TEST_DATA)
  test_data1.rename(columns=home_rename, inplace=True)
  test_data2.rename(columns=away_rename, inplace=True)
  test_data = pd.concat([test_data1, test_data2], axis=0)
  test_data.reset_index(drop=True, inplace=True)
  all_test_data = test_data

  x_test = test_data [X_INPUTS_T]
  y_test = test_data [[OUTPUT]].values.ravel()
  dtest = {'dm':xgb.DMatrix(x_test, label=y_test),'x_test':x_test,'y_test':y_test,'len':len(x_test)}

  # INTERACTION_CONSTRAINTS = [[data.columns.get_loc(j) for j in i] for i in WEIGHTS]
  # INTERACTION_CONSTRAINTS = f'[' + ", ".join(['[' + ', '.join(str(idx) for idx in group) + ']' for group in INTERACTION_CONSTRAINTS]) + ']'
  # print(INTERACTION_CONSTRAINTS)
  if TRIAL:
    best = {
      'max_depth': 0,
      'eta': 0,
      'accuracy': 0,
      'training_data': '',
      'inverse': False,
      'distance': 0,
    }
    for max_depth in range(5,101):
      for eta in np.arange(0.01, 0.91, 0.01):
        params = {
          'max_depth': max_depth,  # the maximum depth of each tree
          'eta': eta,  # the training step for each iteration
          'objective': 'binary:logistic',  # binary classification
          'eval_metric': 'logloss',  # evaluation metric
          'device': 'cuda',
          'tree_method': 'hist',
        }
        accuracy = train(db,params,dtrain,dtest,trial=True,output=OUTPUT)
        distance = accuracy - 0.5
        if distance < 0:
          inverse = True
        else:
          inverse = False
        # if accuracy > best['accuracy']:
        if abs(distance) > abs(best['distance']):
          best['max_depth'] = max_depth
          best['eta'] = eta
          best['accuracy'] = accuracy
          best['distance'] = distance
          best['inverse'] = inverse
        p_eta = f'{round(eta,2)}'.ljust(4)
        p_best_eta = f'{round(best["eta"],2)}'.ljust(4)
        adj_accuracy = accuracy if not inverse else 1-accuracy
        adj_best_accuracy = best['accuracy'] if not best['inverse'] else 1-best['accuracy']
        p_accuracy = f'{OUTPUT} Accuracy:{((adj_accuracy)*100):.2f}%|eta:{p_eta}|max_depth:{max_depth}||{"inverse" if inverse else "       "}'
        p_best = f'Best: Accuracy:{(adj_best_accuracy*100):.2f}%|eta:{p_best_eta}|max_depth:{best["max_depth"]}||{"inverse" if best["inverse"] else "       "}'
        print(f'{p_accuracy}||{p_best}||{dtest["len"]}')
    best_params = {
      'max_depth': best['max_depth'],
      'eta': best['eta'],
      'objective': 'binary:logistic',
      'eval_metric': params['eval_metric'],
      'device': params['device'],
      'tree_method': params['tree_method'],
    }
    p_best_eta = f'{round(best["eta"],2)}'.ljust(4)
    p_best_accuracy = best['accuracy'] if not best['inverse'] else 1-best['accuracy']
    p_best = f'{OUTPUT} Best: Accuracy:{(p_best_accuracy*100):.2f}%|eta:{p_best_eta}|max_depth:{best["max_depth"]}'
    print('Inputs:', X_INPUTS_T)
    print('Output:', OUTPUT)
    print('Params:', best_params)
    train(db,best_params,dtrain,dtest,inverse=best['inverse'],trial=False,output=OUTPUT)
  else:
    print('Inputs:', X_INPUTS_T)
    print('Output:', OUTPUT)
    train(db,GENERAL_PARAMS,dtrain,dtest,trial=False)