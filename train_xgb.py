import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, TEST_DATA_VERSION,TEST_DATA_FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
import xgboost as xgb
from util.helpers import all_combinations
from util.xgb_helpers import mcc_eval
from itertools import combinations
import json
import optuna


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
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
  # 20172018,
  20182019,
  20192020,
  20202021,
  20212022,
  20222023,
]
ALL_SEASONS = all_combinations(SEASONS)
print_seasons = [','.join([str(season)[-2:] for season in seasons]) for seasons in ALL_SEASONS]
TRAINING_DATA_DICT = {season: load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib') for season in SEASONS}
print('Seasons Loaded')
# TRAINING_DATAS = [load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib') for season in SEASONS]
# TRAINING_DATA = np.concatenate(TRAINING_DATAS).tolist()
# ALL_TRAINING_DATAS = [[load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib') for season in seasons] for seasons in ALL_SEASONS]
# ALL_TRAINING_DATA = [np.concatenate([TRAINING_DATA_DICT[season] for season in seasons]).tolist() for seasons in ALL_SEASONS]
# print('Seasons Compiled')
# ALL_TRAINING_DATA = [np.concatenate(datas).tolist() for datas in ALL_TRAINING_DATAS]
# TRAINING_DATA = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
TEST_DATA = load(f'test_data/test_data_v{TEST_DATA_FILE_VERSION}.joblib')


OUTPUT = 'winnerB'

TRIAL = True
DRY_RUN = False

NUM_BOOST_ROUND = 500
N_TRIALS = 100
EARLY_STOPPING_ROUNDS = 10

PARAMS = {
  # 'max_depth': 23,  # the maximum depth of each tree
  # 'eta': 0.18,  # the training step for each iteration
  'max_depth': 8,  # the maximum depth of each tree
  'eta': 0.08,  # the training step for each iteration
  'objective': 'binary:logistic',  # binary classification
  'eval_metric': 'logloss',  # evaluation metric
  # 'sampling_method': 'gradient_based',  # 'uniform', 'gradient_based'
  # 'subsample': 0.1,
  # 'colsample_bytree': 1,
  # 'colsample_bylevel': 1,
  # 'colsample_bynode': 1,
  # 'alpha': 0.00001,
  # 'lambda': 1.1,
  'device': 'cuda',
  'tree_method': 'hist',
}
EPOCHS = 10  # the number of training iterations
THRESHOLD = 0.5

# Best So Far: WinnerB: Accuracy: 60.43% | eta: 0.18 | max_depth: 23 | epochs: 10

# records = []

# f = open('records/xgboost_records.txt', 'a')

def train(db,params,dtrain,dtest,trial=False):
  if not trial:
    print('Inputs:', X_INPUTS)
    print('Output:', OUTPUT)
    print('Params:', params)

    # evallist = [(dtest, 'eval')]
    # evals_result = {}
    # bst = xgb.train(params, dtrain, EPOCHS, evals=[(dtest, 'test')], custom_metric=mcc_eval, evals_result=evals_result)
    bst = xgb.train(params, dtrain, EPOCHS)
    # mcc = evals_result['test']['MCC']
    # max_mcc_index = mcc.index(max(mcc))+1
    # print('MCC:', sum(mcc)/len(mcc))
    # print('Max MCC:', max(mcc))
    # print('Max MCC Index:', max_mcc_index)
    # evals_result = {}
    # bst = xgb.train(params, dtrain, max_mcc_index, evals=[(dtest, 'test')], custom_metric=mcc_eval, evals_result=evals_result)
    # mcc = evals_result['test']['MCC']
    # max_mcc_index = mcc.index(max(mcc))+1
    # print('MCC:', sum(mcc)/len(mcc))
    # print('Max MCC:', max(mcc))
    # print('Max MCC Index:', max_mcc_index)
  else:
    bst = xgb.train(params, dtrain, EPOCHS)

  preds = bst.predict(dtest)

  # Convert probabilities to binary output with a threshold of 0.5
  predictions = [1 if i > THRESHOLD else 0 for i in preds]
  # predictions = preds
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
      'version': VERSION,
      'XGBVersion': XGB_VERSION,
      'testDataVersion': TEST_DATA_VERSION,
      'inputs': X_INPUTS,
      'outputs': Y_OUTPUTS,
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
    if not DRY_RUN:
      # path_accuracy = ('%.2f%%' % (accuracy * 100.0)).replace('.','_')
      # save_path = f'models/nhl_ai_v{XGB_FILE_VERSION}_xgboost_{OUTPUT}_{path_accuracy}.joblib'
      save_path = f'models/nhl_ai_v{XGB_FILE_VERSION}_xgboost_{OUTPUT}_1.joblib'
      dump(bst, save_path)
  return accuracy


# def objective(trial):
#   data = pd.DataFrame(TRAINING_DATA)
#   test_data = pd.DataFrame(TEST_DATA)
#   x_train = data [X_INPUTS]
#   y_train = data [[OUTPUT]].values.ravel()
#   x_test = test_data [X_INPUTS]
#   y_test = test_data [[OUTPUT]].values.ravel()

#   dtrain = xgb.DMatrix(x_train, label=y_train)
#   dtest = xgb.DMatrix(x_test, label=y_test)
#   # Define the hyperparameter space
#   param = {
#     'verbosity': 0,
#     'device': 'cuda',
#     'objective': 'binary:logistic',
#     # 'objective': 'reg:absoluteerror',
#     'tree_method': 'gpu_hist',
#     'predictor': 'gpu_predictor',
#     'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
#     # 'booster': trial.suggest_categorical('booster', ['gbtree']),
#     'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
#     'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True)
#   }

#   # if param['booster'] == 'gbtree':
#   #   param['max_depth'] = trial.suggest_int('max_depth', 10, 100)
#   #   param['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
#   #   param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
#   #   param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

#   if param['booster'] == 'gbtree' or param['booster'] == 'dart':
#     param['max_depth'] = trial.suggest_int('max_depth', 10, 100)
#     param['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
#     param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
#     param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

#   if param['booster'] == 'dart':
#     param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
#     param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
#     param['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
#     param['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)

#   # Train the model
#   bst = xgb.train(param, dtrain, num_boost_round=NUM_BOOST_ROUND)

#   # Make predictions
#   preds = bst.predict(dtest)
#   pred_labels = [1 if i > 0.5 else 0 for i in preds]
#   # pred_labels = [round(i) for i in preds]


#   # Calculate and return the accuracy
#   accuracy = accuracy_score(y_test, pred_labels)
#   return accuracy

if __name__ == '__main__':
  # data = pd.DataFrame(TRAINING_DATA)
  # data = data.sort_values(by='id')
  # x_train = data [X_INPUTS]
  # y_train = data [[OUTPUT]].values.ravel()
  # dtrain = xgb.DMatrix(x_train, label=y_train)

  TRAINING_DATAS = [load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib') for season in SEASONS]
  TRAINING_DATA = np.concatenate(TRAINING_DATAS).tolist()
  data = pd.DataFrame(TRAINING_DATA)
  # data = data.sort_values(by='id')
  data = data.sample(frac=1, random_state=RANDOM_STATE)
  data.reset_index(drop=True, inplace=True)
  x_train = data [X_INPUTS]
  y_train = data [[OUTPUT]].values.ravel()
  dtrain = xgb.DMatrix(x_train, label=y_train)

  test_data = pd.DataFrame(TEST_DATA)
  test_data = test_data.sort_values(by='id')
  x_test = test_data [X_INPUTS]
  y_test = test_data [[OUTPUT]].values.ravel()
  dtest = xgb.DMatrix(x_test, label=y_test)

  # print('Preping Data')
  # datas = [pd.DataFrame(training_data) for training_data in ALL_TRAINING_DATA]
  # print('DataFrames Ready')
  # x_trains = [data[X_INPUTS] for data in datas]
  # y_trains = [data[[OUTPUT]].values.ravel() for data in datas]
  # dtrains = [xgb.DMatrix(x_train, label=y_train) for x_train,y_train in zip(x_trains,y_trains)]
  # print('Data Ready')

  if TRIAL:

    best = {
      'max_depth': 0,
      'eta': 0,
      'accuracy': 0,
      'training_data': '',
      # 'seasons': print_seasons[0],
    }
    # for training_data in dtrains:
    # for seasons in ALL_SEASONS:
    #   print(seasons)
    #   training_data = np.concatenate([TRAINING_DATA_DICT[season] for season in seasons]).tolist()
    #   data = pd.DataFrame(training_data)
    #   data = data.sort_values(by='id')
    #   x_train = data [X_INPUTS]
    #   y_train = data [[OUTPUT]].values.ravel()
    #   dtrain = xgb.DMatrix(x_train, label=y_train)
    #   p_seasons = ','.join([str(season)[-2:] for season in seasons])
    for max_depth in range(5,101):
      for eta in np.arange(0.01, 1.0, 0.01):
        params = {
          'max_depth': max_depth,  # the maximum depth of each tree
          'eta': eta,  # the training step for each iteration
          'objective': 'binary:logistic',  # binary classification
          'eval_metric': 'logloss',  # evaluation metric
          'device': 'cuda',
          'tree_method': 'hist',
        }
        accuracy = train(db,params,dtrain,dtest,trial=True)
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
    train(db,best_params,dtrain,dtest,trial=False)


    # # Create a study object and optimize the objective function
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=N_TRIALS)

    # # Best trial
    # trial = study.best_trial
    # print(f'Accuracy: {trial.value}')
    # print("Best hyperparameters: {}".format(trial.params))
  else:
    TRAINING_DATAS = [load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib') for season in SEASONS]
    TRAINING_DATA = np.concatenate(TRAINING_DATAS).tolist()
    data = pd.DataFrame(TRAINING_DATA)
    data = data.sort_values(by='id')
    # data = data.sample(frac=1, random_state=RANDOM_STATE)
    data.reset_index(drop=True, inplace=True)
    x_train = data [X_INPUTS]
    y_train = data [[OUTPUT]].values.ravel()
    dtrain = xgb.DMatrix(x_train, label=y_train)
    train(db,PARAMS,dtrain,dtest,trial=False)