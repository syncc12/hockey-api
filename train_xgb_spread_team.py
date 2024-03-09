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
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from util.helpers import team_lookup
from training_input import training_input, test_input
from util.xgb_helpers import mcc_eval
from util.team_helpers import away_rename, home_rename
from util.team_constants import PARAMS

# scikit-learn==1.3.2

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
  20172018,
  20182019,
  20192020,
  20202021,
  20212022,
  20222023,
]

OUTPUT = 'spread'

TRIAL = True
DRY_RUN = False
TEAM = False

USE_VALIDATION = True

N_TRIALS = 100
EARLY_STOPPING_ROUNDS = 10
EPOCHS = 10

OBJECTIVE = 'reg:squarederror'
EVAL_METRIC = 'rmse'
DEVICE = 'cuda'
TREE_METHOD = 'hist'

def train(db,params,dtrain,dtest,dvalidate=False,team_name='',trial=False,output=OUTPUT):

  bst = xgb.train(params, dtrain['dm'], EPOCHS)

  if trial and dvalidate:
    vals = bst.predict(dvalidate['dm'])
    y_validate = dvalidate['y_validate']
    validations = [round(i) for i in vals]
    validation_error_name = 'mae'
    validation_error_score = mean_absolute_error(y_validate, vals)
    validation = accuracy_score(y_validate, validations)
  
  preds = bst.predict(dtest['dm'])
  y_test = dtest['y_test']
  predictions = [round(i) for i in preds]
  error_name = 'mae'
  error_score = mean_absolute_error(y_test, preds)
  accuracy = accuracy_score(y_test, predictions)
  if not trial:
    model_data = f"{team_name} {output} Accuracy: {'%.2f%%' % (accuracy * 100.0)} | {error_name}: {error_score} | {dtest['len']}"
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
        'team': team_name,
        'teamId': team,
        'model': 'XGBoost Classifier (Teams)',
        'file': 'train_xgb_spread_team.py',
        'params': params,
        'epochs (num_boost_round)': EPOCHS,
        'accuracies': {
          output: {
            'accuracy': accuracy,
            'validation': validation if dvalidate else None,
            error_name: error_score,
          },
        },
      })

      # path_accuracy = ('%.2f%%' % (accuracy * 100.0)).replace('.','_')
      # save_path = f'models/nhl_ai_v{XGB_FILE_VERSION}_xgboost_{team_name}_{OUTPUT}_{path_accuracy}.joblib'
      save_path = f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_{output}.joblib'
      dump(bst, save_path)
  if trial and dvalidate:
    return accuracy, validation, error_score, error_name
  else:
    return accuracy, error_score, error_name

if __name__ == '__main__':
  teamLookup = team_lookup(db)
  TRAINING_DATA = training_input(SEASONS)
  data1 = pd.DataFrame(TRAINING_DATA)
  data2 = pd.DataFrame(TRAINING_DATA)

  data1.rename(columns=home_rename, inplace=True)
  data2.rename(columns=away_rename, inplace=True)
  data = pd.concat([data1, data2], axis=0)
  data.reset_index(drop=True, inplace=True)
  teams = data.groupby('team')

  if TEAM:
    teams = [(TEAM, teams.get_group(TEAM))]

  dtrains = {}
  dvalidations  = {}
  dtests = {}
  for team, team_data in teams:
    x_train = team_data [X_INPUTS_T]
    y_train = team_data [[OUTPUT]].values.ravel()
    if USE_VALIDATION:
      x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
      dtests[team] = {'dm':xgb.DMatrix(x_test, label=y_test),'x_test':x_test,'y_test':y_test,'len':len(x_test)}
    dtrains[team] = {'dm':xgb.DMatrix(x_train, label=y_train),'x_train':x_train,'y_train':y_train,'len':len(x_train)}

  TEST_DATA = test_input(X_INPUTS_T,[OUTPUT],no_format=True)
  test_data1 = pd.DataFrame(TEST_DATA)
  test_data2 = pd.DataFrame(TEST_DATA)
  test_data1.rename(columns=home_rename, inplace=True)
  test_data2.rename(columns=away_rename, inplace=True)
  test_data = pd.concat([test_data1, test_data2], axis=0)
  test_data.reset_index(drop=True, inplace=True)
  test_teams = test_data.groupby('team')

  if TEAM:
    test_teams = [(TEAM, test_teams.get_group(TEAM))]

  for team, team_data in test_teams:
    if USE_VALIDATION:
      x_validate = team_data [X_INPUTS_T]
      y_validate = team_data [[OUTPUT]].values.ravel()
      dvalidations[team] = {'dm':xgb.DMatrix(x_validate, label=y_validate),'x_validate':x_validate,'y_validate':y_validate,'len':len(x_validate)}
    else:
      x_test = team_data [X_INPUTS_T]
      y_test = team_data [[OUTPUT]].values.ravel()
      dtests[team] = {'dm':xgb.DMatrix(x_test, label=y_test),'x_test':x_test,'y_test':y_test,'len':len(x_test)}

  if TRIAL:
    team_bests = {}
    p_team_bests = []
    for i, (team, dtrain) in enumerate(dtrains.items()):
      best = {
        'max_depth': 0,
        'eta': 0,
        'accuracy': 0,
        'validation': 0,
        'training_data': '',
        'error_score': -1,
        'error_name': '',
      }
      dtest = dtests[team]
      team_name = teamLookup[team]['abbrev']
      for max_depth in range(5,51):
        for eta in np.arange(0.01, 0.3, 0.01):
          params = {
            'max_depth': max_depth,
            'eta': eta,
            'objective': OBJECTIVE,
            'eval_metric': EVAL_METRIC,
            'device': DEVICE,
            'tree_method': TREE_METHOD,
          }
          if USE_VALIDATION:
            dvalidate = dvalidations[team]
            accuracy, validation, error_score, error_name = train(db,params,dtrain,dtest,dvalidate=dvalidate,team_name=team_name,trial=True,output=OUTPUT)
          else:
            validation = -1
            accuracy, error_score, error_name = train(db,params,dtrain,dtest,team_name=team_name,trial=True,output=OUTPUT)
          metric_score = validation if USE_VALIDATION else accuracy
          best_metric_score = best['validation'] if USE_VALIDATION else best['accuracy']
          if metric_score > best_metric_score:
            best['max_depth'] = max_depth
            best['eta'] = eta
            best['accuracy'] = accuracy
            best['validation'] = validation
            best['error_score'] = error_score
            best['error_name'] = error_name

          p_eta = f'{round(eta,2)}'.ljust(4)
          p_best_eta = f'{round(best["eta"],2)}'.ljust(4)
          p_error = f'{error_name}:{error_score:.4f}'.ljust(6)
          p_best_error = f'{best["error_name"]}:{best["error_score"]:.4f}'.ljust(6)
          p_accuracy = f'{team_name} {OUTPUT} Accuracy:{((accuracy)*100):.2f}%|validation:{((validation)*100):.2f}%|{p_error}|eta:{p_eta}|max_depth:{max_depth}'
          p_best = f'Best: Accuracy:{(best["accuracy"]*100):.2f}%|validation:{(best["validation"]*100):.2f}%|{p_best_error}|eta:{p_best_eta}|max_depth:{best["max_depth"]}'
          p_validation_len = f'||{dvalidate["len"]}' if USE_VALIDATION else ""
          print(f'[{i+1}/{len(dtrains)}] {p_accuracy}||{p_best}||{dtest["len"]}{p_validation_len}')
      best_params = {
        'max_depth': best['max_depth'],
        'eta': best['eta'],
        'objective': OBJECTIVE,
        'eval_metric': params['eval_metric'],
        'device': params['device'],
        'tree_method': params['tree_method'],
      }
      team_bests[team_name] = {
        'accuracy': best['accuracy'],
        'validation': best['validation'],
        'output': OUTPUT,
        **best_params
      }
      p_best_eta = f'{round(best["eta"],2)}'.ljust(4)
      p_best_error = f'{best["error_name"]}:{best["error_score"]:.4f}'.ljust(6)
      p_best = f'{team_name} {OUTPUT} Best: Accuracy:{(best["accuracy"]*100):.2f}%|validation:{(best["validation"]*100):.2f}%|{p_best_error}|eta:{p_best_eta}|max_depth:{best["max_depth"]}'
      p_team_bests.append(f'{p_best}||{dtest["len"]}')
      print('Inputs:', X_INPUTS_T)
      print('Output:', OUTPUT)
      print('Params:', best_params)
      train(db,best_params,dtrain,dtest,team_name=team_name,trial=False,output=OUTPUT)
    print(team_bests)
    for p_team_best in p_team_bests:
      print(p_team_best)
  else:
    print('Inputs:', X_INPUTS_T)
    print('Output:', OUTPUT)
    for team, dtrain in dtrains.items():
      team_params = PARAMS[team]
      print('Params:', team_params)

      dtest = dtests[team]
      team_name = teamLookup[team]['abbrev']
      team_params['objective'] = OBJECTIVE
      team_params['eval_metric'] = EVAL_METRIC
      team_params['device'] = DEVICE
      team_params['tree_method'] = TREE_METHOD
      train(db,PARAMS[team],dtrain,dtest,team_name=team_name,trial=False)