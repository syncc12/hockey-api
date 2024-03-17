import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from pages.mlb.inputs import X_INPUTS_MLB_T, ENCODE_COLUMNS, mlb_training_input, mlb_test_input
from pages.mlb.mlb_helpers import team_lookup, away_rename, home_rename
from constants.constants import MLB_TEAM_VERSION, MLB_TEAM_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from util.team_constants import PARAMS

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
mlb_db = client["mlb"]
TEST_SEASONS = [
  2023,
  2022,
  2021,
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
  2015,
  2014,
  2013,
  2012,
  2011,
  2010,
  2009,
  2008,
  2007,
  2006,
  2005,
  2004,
  2003,
  2002,
  2001,
  2000,
]

OUTPUT = 'winner'

USE_TEST_SPLIT = True

TRIAL = True
DRY_RUN = False
TEAM = False
TOTAL = False

NUM_BOOST_ROUND = 500
N_TRIALS = 100
EARLY_STOPPING_ROUNDS = 10
EPOCHS = 10
THRESHOLD = 0.5

GENERAL_PARAMS = {
  'max_depth': 8,
  'eta': 0.08,
  'objective': 'binary:logistic',
  'eval_metric': 'logloss',
  'device': 'cuda',
  'tree_method': 'hist',
}

def train(db,params,dtrain,dtest,team_name='all',inverse=False,trial=False,output=OUTPUT):

  bst = xgb.train(params, dtrain['dm'], EPOCHS)

  preds = bst.predict(dtest['dm'])
  y_test = dtest['y_test']

  predictions = [1 if i < THRESHOLD else 0 for i in preds] if inverse else [1 if i > THRESHOLD else 0 for i in preds]

  accuracy = accuracy_score(y_test, predictions)
  if not trial:
    if accuracy < 0.5:
      inverse = True
      accuracy = 1 - accuracy
    model_data = f"{team_name} {output} Accuracy: {'%.2f%%' % (accuracy * 100.0)}{' INVERSE' if inverse else ''}"
    print(model_data)

    if not DRY_RUN:
      TrainingRecords = db['dev_training_records']

      timestamp = time.time()
      TrainingRecords.insert_one({
        'sport': 'mlb',
        'savedAt': timestamp,
        'version': MLB_TEAM_VERSION,
        'inputs': X_INPUTS_MLB_T,
        'outputs': OUTPUT,
        'randomState': RANDOM_STATE,
        'startingSeason': START_SEASON,
        'finalSeason': END_SEASON,
        'seasons': SEASONS,
        'testSeasons': TEST_SEASONS,
        'testSplit': USE_TEST_SPLIT,
        'team': team_name,
        'teamId': team,
        'model': 'XGBoost Classifier (Teams)',
        'file': 'train_mlb_xgb_team.py',
        'threshold': THRESHOLD,
        'params': params,
        'epochs': EPOCHS,
        'accuracies': {
          output: accuracy,
        },
      })

      save_path = f'models/mlb_ai_v{MLB_TEAM_FILE_VERSION}_xgboost_team{team}_{output}{"_F" if inverse else ""}.joblib'
      dump(bst, save_path)
  return accuracy

if __name__ == '__main__':
  teamLookup = team_lookup(mlb_db,only_active_mlb=True)
  TRAINING_DATA = mlb_training_input(SEASONS)
  data1 = pd.DataFrame(TRAINING_DATA)
  data2 = pd.DataFrame(TRAINING_DATA)

  data1.rename(columns=home_rename, inplace=True)
  data2.rename(columns=away_rename, inplace=True)
  data = pd.concat([data1, data2], axis=0)
  data.reset_index(drop=True, inplace=True)
  for column in ENCODE_COLUMNS:
    data = data[data[column] != -1]
    encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
    data[column] = encoder.transform(data[column])
  keep_teams = list(teamLookup.keys())
  data = data[data['team'].isin(keep_teams)]
  data = data[data['opponent'].isin(keep_teams)]
  all_data = data
  teams = data.groupby('team')

  if TEAM:
    teams = [(TEAM, teams.get_group(TEAM))]

  dtrains = {}
  dtests = {}
  for team, team_data in teams:
    x_train = team_data [X_INPUTS_MLB_T]
    y_train = team_data [[OUTPUT]].values.ravel()
    if USE_TEST_SPLIT:
      x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
      dtests[team] = {'dm':xgb.DMatrix(x_test, label=y_test),'x_test':x_test,'y_test':y_test,'len':len(x_test)}
    dtrains[team] = {'dm':xgb.DMatrix(x_train, label=y_train),'x_train':x_train,'y_train':y_train,'len':len(x_train)}
  
  if not USE_TEST_SPLIT:
    TEST_DATA = mlb_test_input(TEST_SEASONS)
    test_data1 = pd.DataFrame(TEST_DATA)
    test_data2 = pd.DataFrame(TEST_DATA)
    test_data1.rename(columns=home_rename, inplace=True)
    test_data2.rename(columns=away_rename, inplace=True)
    test_data = pd.concat([test_data1, test_data2], axis=0)
    test_data.reset_index(drop=True, inplace=True)
    for column in ENCODE_COLUMNS:
      test_data = test_data[test_data[column] != -1]
      encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
      test_data[column] = encoder.transform(test_data[column])
    test_data = test_data[test_data['team'].isin(keep_teams)]
    test_data = test_data[test_data['opponent'].isin(keep_teams)]
    all_test_data = test_data
    test_teams = test_data.groupby('team')

    if TEAM:
      test_teams = [(TEAM, test_teams.get_group(TEAM))]

    for team, team_data in test_teams:
      x_test = team_data [X_INPUTS_MLB_T]
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
        'training_data': '',
        'inverse': False,
        'distance': 0,
      }
      dtest = dtests[team]
      team_name = teamLookup[team]['abbrev']
      for max_depth in range(5,51):
        for eta in np.arange(0.01, 0.3, 0.01):
          params = {
            'max_depth': max_depth,
            'eta': eta,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'device': 'cuda',
            'tree_method': 'hist',
          }
          accuracy = train(db,params,dtrain,dtest,team_name,trial=True,output=OUTPUT)
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
          p_accuracy = f'{team_name} {OUTPUT} Accuracy:{((adj_accuracy)*100):.2f}%|eta:{p_eta}|max_depth:{max_depth}||{"inverse" if inverse else "       "}'
          p_best = f'Best: Accuracy:{(adj_best_accuracy*100):.2f}%|eta:{p_best_eta}|max_depth:{best["max_depth"]}||{"inverse" if best["inverse"] else "       "}'
          print(f'[{i+1}/{len(dtrains)}] {p_accuracy}||{p_best}||{dtest["len"]}')
      best_params = {
        'max_depth': best['max_depth'],
        'eta': best['eta'],
        'objective': 'binary:logistic',
        'eval_metric': params['eval_metric'],
        'device': params['device'],
        'tree_method': params['tree_method'],
      }
      team_bests[team_name] = {
        'accuracy': best['accuracy'],
        'inverse': best['inverse'],
        'distance': best['distance'],
        'output': OUTPUT,
        **best_params
      }
      p_best_eta = f'{round(best["eta"],2)}'.ljust(4)
      p_best_accuracy = best['accuracy'] if not best['inverse'] else 1-best['accuracy']
      p_best = f'{team_name} {OUTPUT} Best: Accuracy:{(p_best_accuracy*100):.2f}%|eta:{p_best_eta}|max_depth:{best["max_depth"]}'
      p_team_bests.append(f'{p_best}||{"inverse" if best["inverse"] else "       "}||{dtest["len"]}')
      print('Inputs:', X_INPUTS_MLB_T)
      print('Output:', OUTPUT)
      print('Params:', best_params)
      train(db,best_params,dtrain,dtest,team_name,inverse=best['inverse'],trial=False,output=OUTPUT)
    print(team_bests)
    for p_team_best in p_team_bests:
      print(p_team_best)
  elif TOTAL:
    x_train = all_data [X_INPUTS_MLB_T]
    y_train = all_data [[OUTPUT]].values.ravel()
    dtrain = {'dm':xgb.DMatrix(x_train, label=y_train),'x_train':x_train,'y_train':y_train,'len':len(x_train)}
    x_test = team_data [X_INPUTS_MLB_T]
    y_test = team_data [[OUTPUT]].values.ravel()
    dtest = {'dm':xgb.DMatrix(x_test, label=y_test),'x_test':x_test,'y_test':y_test,'len':len(x_test)}
    train(db,GENERAL_PARAMS,dtrain,dtest,trial=False)
  else:
    print('Inputs:', X_INPUTS_MLB_T)
    print('Output:', OUTPUT)
    for team, dtrain in dtrains.items():
      team_params = PARAMS[team]
      # print('Params:', team_params)

      dtest = dtests[team]
      team_name = teamLookup[team]['abbrev']
      team_params['objective'] = 'binary:logistic'
      team_params['eval_metric'] = 'logloss'
      team_params['device'] = 'cuda'
      team_params['tree_method'] = 'hist'
      train(db,team_params[OUTPUT],dtrain,dtest,team_name,trial=False)