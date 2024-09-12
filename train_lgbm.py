import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from constants.inputConstants import X_INPUTS, X_INPUTS_S, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
from training_input import training_input, test_input
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]


SEASONS = [
  20052006,
  20062007,
  20072008,
  20082009,
  20092010,
  20102011,
  20112012,
  20122013,
  20132014,
  20142015,
  20152016,
  20162017,
  20172018,
  20182019,
  20192020,
  20202021,
  20212022,
  20222023,
]

USE_X_INPUTS = X_INPUTS_S

OUTPUT = 'winnerB'

TRIAL = False
OPTIMIZE = True
DRY_RUN = True

NUM_BOOST_ROUND = 100

PARAMS = {
    # 'device': 'gpu',
    # 'gpu_device_id': 0,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}


def train(db,params,dtrain,x_test,y_test,num_boost_round=NUM_BOOST_ROUND,trial=False):
  if not trial:
    print('Inputs:', len(USE_X_INPUTS))
    print('Output:', OUTPUT)
    print('Params:', params)

  gbm = lgb.train(params, dtrain, num_boost_round=num_boost_round)
  preds = gbm.predict(x_test, num_iteration=gbm.best_iteration)
  predictions = np.where(preds > 0.5, 1, 0)
  accuracy = accuracy_score(y_test, predictions)

  if not trial:
    model_data = f"Accuracy: {'%.2f%%' % (accuracy * 100.0)}"
    print(model_data)

    if not DRY_RUN:
      save_path = f'models/nhl_ai_v{FILE_VERSION}_lightgbm_{OUTPUT}.joblib'
      dump(gbm, save_path)

    TrainingRecords = db['dev_training_records']

    timestamp = time.time()
    TrainingRecords.insert_one({
      'sport': 'nhl',
      'savedAt': timestamp,
      'version': VERSION,
      'inputs': USE_X_INPUTS,
      'randomState': RANDOM_STATE,
      'startingSeason': START_SEASON,
      'finalSeason': END_SEASON,
      'seasons': SEASONS,
      'model': 'LightGBM Classifier',
      'params': PARAMS,
      'accuracies': {
        OUTPUT: accuracy,
      },
    })
  return accuracy





if __name__ == '__main__':
  TRAINING_DATA = training_input(SEASONS)
  data = pd.DataFrame(TRAINING_DATA)

  x_train = data [USE_X_INPUTS]
  y_train = data [[OUTPUT]].to_numpy()

  TEST_DATA = test_input(no_format=True)
  test_data = pd.DataFrame(TEST_DATA)

  x_test = test_data [USE_X_INPUTS]
  y_test = test_data [[OUTPUT]].to_numpy()

  dtrain = lgb.Dataset(x_train, label=y_train)

  
  if OPTIMIZE:
    MAX_EVALS = 1500
    print(f'Starting Optimization')
    class_keys = list(data.value_counts(OUTPUT).keys())
    space = {
      'feature_fraction': hp.quniform('feature_fraction', 0.1, 1.0, 0.1),
      'bagging_fraction': hp.quniform('bagging_fraction', 0.1, 1.0, 0.1),
      'bagging_freq': hp.uniform('bagging_freq', 0,100),
      'num_leaves': hp.uniform('num_leaves', 20,300),
      'learning_rate': hp.quniform('learning_rate', 0.001, 0.3, 0.001),
      'num_boost_round': hp.uniform('num_boost_round', 100,1000),
    }
    def objective(params):
      model = lgb.train(
        {
          # 'device': 'gpu',
          # 'gpu_device_id': 0,
          'objective': 'binary',
          'metric': 'binary_logloss',
          'boosting': 'gbdt',
          'feature_fraction': params['feature_fraction'],
          'bagging_fraction': params['bagging_fraction'],
          'bagging_freq': int(params['bagging_freq']),
          'num_leaves': int(params['num_leaves']),
          'learning_rate': params['learning_rate'],
          'verbose': '-1',
        },
        dtrain,
        num_boost_round=int(params['num_boost_round'])
      )
      preds = model.predict(x_test, num_iteration=model.best_iteration)
      predictions = np.where(preds > 0.5, 1, 0)
      accuracy = accuracy_score(y_test, predictions)
      return {'loss': -accuracy, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
    print(f"Best parameters: {best}")
    exclude_keys = {'num_boost_rounds'}
    params = {
      # 'device': 'gpu',
      # 'gpu_device_id': 0,
      'objective': 'binary',
      'metric': 'binary_logloss',
      'boosting': 'gbdt',
      'feature_fraction': best['feature_fraction'],
      'bagging_fraction': best['bagging_fraction'],
      'bagging_freq': int(best['bagging_freq']),
      'num_leaves': int(best['num_leaves']),
      'learning_rate': best['learning_rate'],
    }
    train(db,params,dtrain,x_test,y_test,num_boost_round=int(best['num_boost_round']),trial=False)

  elif TRIAL:

    best = {
      'num_leaves': 0,
      'learning_rate': 0.0,
      'feature_fraction': 0.0,
      'bagging_fraction': 0.0,
      'bagging_freq': 0,
      'accuracy': 0.0,
    }
    for feature_fraction in np.arange(0.1, 1.0, 0.1):
      for bagging_fraction in np.arange(0.1, 1.0, 0.1):
        for bagging_freq in range(0,100,5):
          for num_leaves in range(20,301):
            for learning_rate in np.arange(0.001, 0.3, 0.001):
              params = {
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'bagging_freq': bagging_freq,
                # 'device': 'gpu',
                # 'gpu_device_id': 0,
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting': 'gbdt',
                'verbose': -1,
              }
              accuracy = train(db,params,dtrain,x_test,y_test,trial=True)
              if accuracy > best['accuracy']:
                best['num_leaves'] = num_leaves
                best['learning_rate'] = learning_rate
                best['feature_fraction'] = feature_fraction
                best['bagging_fraction'] = bagging_fraction
                best['bagging_freq'] = bagging_freq
                best['accuracy'] = accuracy
              
              p_accuracy = f'Accuracy:{(accuracy*100):.2f}%'
              p_num_leaves = f'num_leaves:{num_leaves}'.ljust(3)
              p_learning_rate = f'learning_rate:{round(learning_rate,3)}'.ljust(5)
              p_feature_fraction = f'feature_fraction:{round(feature_fraction,1)}'.ljust(3)
              p_bagging_fraction = f'bagging_fraction:{round(bagging_fraction,1)}'.ljust(3)
              p_bagging_freq = f'bagging_freq:{bagging_freq}'.ljust(3)

              p_best_accuracy = f'Accuracy:{(best["accuracy"]*100):.2f}%'
              p_best_num_leaves = f'num_leaves:{best["num_leaves"]}'.ljust(3)
              p_best_learning_rate = f'learning_rate:{round(best["learning_rate"],3)}'.ljust(5)
              p_best_feature_fraction = f'feature_fraction:{round(best["feature_fraction"],1)}'.ljust(3)
              p_best_bagging_fraction = f'bagging_fraction:{round(best["bagging_fraction"],1)}'.ljust(3)
              p_best_bagging_freq = f'bagging_freq:{best["bagging_freq"]}'.ljust(3)

              p_round = f'{p_accuracy}|{p_num_leaves}|{p_learning_rate}|{p_feature_fraction}|{p_bagging_fraction}|{p_bagging_freq}'
              p_best = f'{p_best_accuracy}|{p_best_num_leaves}|{p_best_learning_rate}|{p_best_feature_fraction}|{p_best_bagging_fraction}|{p_best_bagging_freq}'
              print(f'[{OUTPUT}]{p_round}||{p_best}')
    best_params = {
      'num_leaves': best['num_leaves'],
      'learning_rate': best['learning_rate'],
      'feature_fraction': best['feature_fraction'],
      'bagging_fraction': best['bagging_fraction'],
      'bagging_freq': best['bagging_freq'],
      # 'device': params['device'],
      # 'gpu_device_id': params['gpu_device_id'],
      'objective': params['objective'],
      'metric': params['metric'],
      'boosting': params['boosting'],
    }
    train(db,best_params,dtrain,x_test,y_test,trial=False)
  else:
    train(db,PARAMS,dtrain,x_test,y_test,trial=False)