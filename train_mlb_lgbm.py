import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
import lightgbm as lgb
from constants.constants import MLB_LGBM_VERSION, MLB_LGBM_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from pages.mlb.inputs import X_INPUTS_MLB_S, X_INPUTS_MLB, ENCODE_COLUMNS, mlb_training_input, mlb_test_input


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
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

USE_X_INPUTS_MLB = X_INPUTS_MLB_S

OUTPUT = 'winner'

USE_TEST_SPLIT = True

TRIAL = False
OPTIMIZE = True
DRY_RUN = True

NUM_BOOST_ROUND = 100

PARAMS = {
    'device': 'gpu',
    'gpu_device_id': 0,
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
    print('Inputs:', len(USE_X_INPUTS_MLB))
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
      save_path = f'models/mlb_ai_v{MLB_LGBM_FILE_VERSION}_lightgbm_{OUTPUT}.joblib'
      dump(gbm, save_path)

    TrainingRecords = db['dev_training_records']

    timestamp = time.time()
    TrainingRecords.insert_one({
      'sport': 'mlb',
      'savedAt': timestamp,
      'version': MLB_LGBM_VERSION,
      'inputs': USE_X_INPUTS_MLB,
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

  x_train = data [USE_X_INPUTS_MLB]
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

    x_test = test_data [USE_X_INPUTS_MLB]
    y_test = test_data [[OUTPUT]].values.ravel()

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
          'device': 'gpu',
          'gpu_device_id': 0,
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
      'device': 'gpu',
      'gpu_device_id': 0,
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
                'device': 'gpu',
                'gpu_device_id': 0,
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
      'device': params['device'],
      'gpu_device_id': params['gpu_device_id'],
      'objective': params['objective'],
      'metric': params['metric'],
      'boosting': params['boosting'],
    }
    train(db,best_params,dtrain,x_test,y_test,trial=False)
  else:
    train(db,PARAMS,dtrain,x_test,y_test,trial=False)