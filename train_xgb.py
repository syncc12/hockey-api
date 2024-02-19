import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
import xgboost as xgb
import json
import optuna


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
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
  'eval_metric': 'logloss',  # evaluation metric
  'device': 'cuda',
  'tree_method': 'hist',
}
EPOCHS = 10  # the number of training iterations
THRESHOLD = 0.5

# Best So Far: WinnerB: Accuracy: 60.43% | eta: 0.18 | max_depth: 23 | epochs: 10

# records = []

# f = open('records/xgboost_records.txt', 'a')

def train(db,params,trial):
  if not trial:
    print('Inputs:', X_INPUTS)
    print('Output:', OUTPUT)
  data = pd.DataFrame(TRAINING_DATA)
  test_data = pd.DataFrame(TEST_DATA)
  x_train = data [X_INPUTS]
  y_train = data [[OUTPUT]].values.ravel()
  x_test = test_data [X_INPUTS]
  y_test = test_data [[OUTPUT]].values.ravel()

  dtrain = xgb.DMatrix(x_train, label=y_train)
  dtest = xgb.DMatrix(x_test, label=y_test)

  bst = xgb.train(params, dtrain, EPOCHS)

  preds = bst.predict(dtest)

  # Convert probabilities to binary output with a threshold of 0.5
  predictions = [1 if i > THRESHOLD else 0 for i in preds]
  # predictions = [round(i) for i in preds]

  accuracy = accuracy_score(y_test, predictions)
  # model_data = f"Accuracy: {'%.2f%%' % (accuracy * 100.0)} | eta: {eta} | max_depth: {max_depth} | epochs: {epochs}"
  if not trial:
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
      'threshold': THRESHOLD,
      'params': PARAMS,
      'epochs': EPOCHS,
      'accuracies': {
        OUTPUT: accuracy,
      },
    })

    dump(bst, f'models/nhl_ai_v{FILE_VERSION}_xgboost_{OUTPUT}.joblib')
  return accuracy


def objective(trial):
  data = pd.DataFrame(TRAINING_DATA)
  test_data = pd.DataFrame(TEST_DATA)
  x_train = data [X_INPUTS]
  y_train = data [[OUTPUT]].values.ravel()
  x_test = test_data [X_INPUTS]
  y_test = test_data [[OUTPUT]].values.ravel()

  dtrain = xgb.DMatrix(x_train, label=y_train)
  dtest = xgb.DMatrix(x_test, label=y_test)
  # Define the hyperparameter space
  param = {
    'verbosity': 0,
    'device': 'cuda',
    'objective': 'binary:logistic',
    # 'objective': 'reg:absoluteerror',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
    # 'booster': trial.suggest_categorical('booster', ['gbtree']),
    'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
    'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True)
  }

  # if param['booster'] == 'gbtree':
  #   param['max_depth'] = trial.suggest_int('max_depth', 10, 100)
  #   param['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
  #   param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
  #   param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

  if param['booster'] == 'gbtree' or param['booster'] == 'dart':
    param['max_depth'] = trial.suggest_int('max_depth', 10, 100)
    param['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
    param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
    param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

  if param['booster'] == 'dart':
    param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
    param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
    param['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
    param['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)

  # Train the model
  bst = xgb.train(param, dtrain, num_boost_round=NUM_BOOST_ROUND)

  # Make predictions
  preds = bst.predict(dtest)
  pred_labels = [1 if i > 0.5 else 0 for i in preds]
  # pred_labels = [round(i) for i in preds]


  # Calculate and return the accuracy
  accuracy = accuracy_score(y_test, pred_labels)
  return accuracy

if TRIAL:
  best = {
    'max_depth': 0,
    'eta': 0,
    'accuracy': 0,
  }
  for max_depth in range(10,101):
    for eta in np.arange(0.01, 0.91, 0.01):
      params = {
        'max_depth': max_depth,  # the maximum depth of each tree
        'eta': eta,  # the training step for each iteration
        'objective': 'binary:logistic',  # binary classification
        'eval_metric': 'logloss',  # evaluation metric
        'device': 'cuda',
        'tree_method': 'hist',
      }
      accuracy = train(db,params,True)
      if accuracy > best['accuracy']:
        best['max_depth'] = max_depth
        best['eta'] = eta
        best['accuracy'] = accuracy
      print(f'{OUTPUT} Accuracy: {(accuracy*100):.2f}% | eta: {eta} | max_depth: {max_depth} | Best So Far: Accuracy: {(best["accuracy"]*100):.2f}% | eta: {best["eta"]} | max_depth: {best["max_depth"]}')
  best_params = {
    'max_depth': best['max_depth'],
    'eta': best['eta'],
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cuda',
    'tree_method': 'hist',
  }
  train(db,best_params,False)


  # # Create a study object and optimize the objective function
  # study = optuna.create_study(direction='maximize')
  # study.optimize(objective, n_trials=N_TRIALS)

  # # Best trial
  # trial = study.best_trial
  # print(f'Accuracy: {trial.value}')
  # print("Best hyperparameters: {}".format(trial.params))
else:
  train(db)

