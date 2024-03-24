import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
import lightgbm as lgb
from constants.constants import MLB_CAT_VERSION, MLB_CAT_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from util.helpers import safe_chain
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from pages.mlb.inputs import X_INPUTS_MLB_S, X_INPUTS_MLB, ENCODE_COLUMNS, mlb_training_input, mlb_test_input
from catboost import CatBoostClassifier

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

USE_TEST_SPLIT = False

TRIAL = True
OPTIMIZE = False
DRY_RUN = True

MAX_DEPTH = 10000

PARAMS = {
  'iterations': 1000, 
  'learning_rate': 0.1, 
  'depth': 6,
  'task_type':'GPU',
  'devices':'0:1',
  'loss_function': 'Logloss', 
  'verbose': 100, 
  'eval_metric': 'Accuracy'
}


def train(db,params,x_train,y_train,x_test,y_test,trial=False):
  if not trial:
    print('Inputs:', len(USE_X_INPUTS_MLB))
    print('Output:', OUTPUT)
    print('Params:', params)

  if trial:
    model = CatBoostClassifier(
      iterations=params['iterations'],
      learning_rate=params['learning_rate'],
      depth=params['depth'],
      task_type=params['task_type'],
      devices=params['devices'],
      loss_function=params['loss_function'],
      logging_level='Silent',
      eval_metric=params['eval_metric'],
    )
  else:
    model = CatBoostClassifier(
      iterations=params['iterations'],
      learning_rate=params['learning_rate'],
      depth=params['depth'],
      task_type=params['task_type'],
      devices=params['devices'],
      loss_function=params['loss_function'],
      verbose=params['verbose'],
      eval_metric=params['eval_metric'],
    )


  model.fit(x_train, y_train, eval_set=(x_test, y_test), use_best_model=True)

  predictions = model.predict(x_test)
  accuracy = accuracy_score(y_test, predictions)

  if not trial:
    model_data = f"Accuracy: {'%.2f%%' % (accuracy * 100.0)}"
    print(model_data)

    if not DRY_RUN:
      save_path = f'models/mlb_ai_v{MLB_CAT_FILE_VERSION}_catboost_{OUTPUT}.joblib'
      dump(model, save_path)

    TrainingRecords = db['dev_training_records']

    timestamp = time.time()
    TrainingRecords.insert_one({
      'sport': 'mlb',
      'savedAt': timestamp,
      'version': MLB_CAT_VERSION,
      'inputs': USE_X_INPUTS_MLB,
      'randomState': RANDOM_STATE,
      'startingSeason': START_SEASON,
      'finalSeason': END_SEASON,
      'seasons': SEASONS,
      'model': 'CatBoost Classifier',
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
    dump(test_df,f'pages/mlb/data/test_data_catboost.joblib')
  
  y_train = y_train.values.ravel()
  
  if not USE_TEST_SPLIT:
    TEST_DATA = mlb_test_input(TEST_SEASONS)
    test_data = pd.DataFrame(TEST_DATA)
    dump(test_data,f'pages/mlb/data/test_data_catboost.joblib')
    for column in ENCODE_COLUMNS:
      encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
      test_data = test_data[test_data[column] != -1]
      test_data[column] = encoder.transform(test_data[column])

    x_test = test_data [USE_X_INPUTS_MLB]
    y_test = test_data [[OUTPUT]].values.ravel()

  
  if OPTIMIZE:
    MAX_EVALS = 1500
    print(f'Starting Optimization')
    class_keys = list(data.value_counts(OUTPUT).keys())
    space = {
      'iterations': hp.uniform('iterations', 100, 3000),
      'learning_rate': hp.quniform('learning_rate', 0.01, 0.3, 0.01),
      'depth': hp.uniform('depth', 4, 10),
    }
    def objective(params):

      model = CatBoostClassifier(
        iterations=int(params['iterations']),
        learning_rate=params['learning_rate'],
        depth=int(params['depth']),
        task_type='GPU',
        devices='0:1',
        loss_function='Logloss',
        eval_metric='Accuracy',
        logging_level='Silent',
      )

      model.fit(x_train, y_train, eval_set=(x_test, y_test), use_best_model=True)

      preds = model.predict(x_test)
      predictions = np.where(preds > 0.5, 1, 0)
      accuracy = accuracy_score(y_test, predictions)
      return {'loss': -accuracy, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
    print(f"Best parameters: {best}")
    exclude_keys = {'num_boost_rounds'}
    params = {
      'iterations': int(best['iterations']),
      'learning_rate': best['learning_rate'],
      'depth': int(best['depth']),
      'task_type': 'GPU',
      'devices': '0:1',
      'loss_function': 'Logloss',
      'verbose': 100,
      'eval_metric': 'Accuracy',
    }
    train(db,params,x_train,y_train,x_test,y_test,num_boost_round=int(best['num_boost_round']),trial=False)

  elif TRIAL:

    best = {
      'iterations': 0,
      'learning_rate': 0.0,
      'depth': 0,
      'accuracy': 0.0,
    }
    iterations_start = 100
    iterations_end = 3000
    depth_start = 4
    depth_end = 10
    learning_rate_start = 0.01
    learning_rate_end = 0.31
    learning_rate_step = 0.01
    iterations_len = iterations_end-iterations_start
    depth_len = depth_end-depth_start
    learning_rate_len = (learning_rate_end-learning_rate_start)/learning_rate_step
    loop_len = MAX_DEPTH if MAX_DEPTH else round(iterations_len * depth_len * learning_rate_len)
    loop_count = 0
    for iterations in range(iterations_start, iterations_end):
      for depth in range(depth_start, depth_end):
        for learning_rate in np.arange(learning_rate_start, learning_rate_end, learning_rate_step):
          loop_count += 1
          params = {
            'learning_rate': learning_rate,
            'iterations': iterations,
            'depth': depth,
            'task_type': 'GPU',
            'devices': '0:1',
            'loss_function': 'Logloss',
            'logging_level': 'Silent',
            'eval_metric': 'Accuracy',
          }
          accuracy = train(db,params,x_train,y_train,x_test,y_test,trial=True)
          if accuracy > best['accuracy']:
            best['learning_rate'] = learning_rate
            best['iterations'] = iterations
            best['depth'] = depth
            best['accuracy'] = accuracy

          p_loop_count = f'{str(loop_count).rjust(len(str(loop_len)))}/{str(loop_len)}'
          p_accuracy = f'Accuracy:{(accuracy*100):.2f}%'
          p_iterations = f'iterations:{str(iterations).ljust(4)}'
          p_learning_rate = f'learning_rate:{str(round(learning_rate,2)).ljust(4)}'
          p_depth = f'depth:{str(depth).ljust(3)}'
          
          p_best_accuracy = f'Accuracy:{(best["accuracy"]*100):.2f}%'
          p_best_iterations = f'iterations:{str(best["iterations"]).ljust(4)}'
          p_best_learning_rate = f'learning_rate:{str(round(best["learning_rate"],2)).ljust(4)}'
          p_best_depth = f'depth:{str(best["depth"]).ljust(3)}'

          p_round = f'{p_accuracy}|{p_iterations}|{p_learning_rate}|{p_depth}'
          p_best = f'{p_best_accuracy}|{p_best_iterations}|{p_best_learning_rate}|{p_best_depth}'
          print(f'[{p_loop_count}][{OUTPUT}]{p_round}||{p_best}')
          if MAX_DEPTH:
            if loop_count >= MAX_DEPTH:
              break
        if MAX_DEPTH:
          if loop_count >= MAX_DEPTH:
            break
      if MAX_DEPTH:
        if loop_count >= MAX_DEPTH:
          break
    
    best_params = {
      'learning_rate': best['learning_rate'],
      'iterations': best['iterations'],
      'depth': best['depth'],
      'task_type': params['task_type'],
      'devices': params['devices'],
      'loss_function': params['loss_function'],
      'verbose': 100,
      'eval_metric': params['eval_metric'],
    }
    train(db,best_params,x_train,y_train,x_test,y_test,trial=False)
  else:
    train(db,PARAMS,x_train,y_train,x_test,y_test,trial=False)