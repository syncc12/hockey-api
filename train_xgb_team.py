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
from util.team_constants import PARAMS

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

OUTPUT = 'winB'
OUTPUTS = ['winB','lossB']

TRIAL = False
DRY_RUN = True
TEAM = False

NUM_BOOST_ROUND = 500
N_TRIALS = 100
EARLY_STOPPING_ROUNDS = 10
EPOCHS = 10
THRESHOLD = 0.5

# WEIGHTS = {
#   'season': 1,
#   'gameType': 1,
#   'venue': 1,
#   'team': 0.5,
#   'opponent': 1,
#   'date': 1,
#   'headCoach': 1,
#   'forwardAverage': 2,
#   'defenseAverage': 2,
#   'goalieAverage': 2.5,
#   'forwardAverageAge': 1,
#   'defenseAverageAge': 1,
#   'goalieAverageAge': 1,
#   'opponentHeadCoach': 1,
#   'opponentForwardAverage': 2,
#   'opponentDefenseAverage': 2,
#   'opponentGoalieAverage': 2.5,
#   'opponentForwardAverageAge': 1,
#   'opponentDefenseAverageAge': 1,
#   'opponentGoalieAverageAge': 1,
# }
  
WEIGHTS = [
  [
    'season',
    'gameType',
    'venue',
    'date'
  ],
  [
    'team',
    'opponent'
  ],
  [
    'headCoach','opponentHeadCoach',
    'forwardAverage','defenseAverage','goalieAverage',
    # 'forwardAverageAge','defenseAverageAge','goalieAverageAge',
    'opponentForwardAverage','opponentDefenseAverage','opponentGoalieAverage',
    # 'opponentForwardAverageAge','opponentDefenseAverageAge','opponentGoalieAverageAge',
  ],
]

def train(db,params,dtrain,dtest,team_name,inverse=False,trial=False,output=OUTPUT):

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
        'file': 'train_xgb_team.py',
        'threshold': THRESHOLD,
        'params': params,
        'epochs': EPOCHS,
        'accuracies': {
          output: accuracy,
        },
      })

      # path_accuracy = ('%.2f%%' % (accuracy * 100.0)).replace('.','_')
      # save_path = f'models/nhl_ai_v{XGB_FILE_VERSION}_xgboost_{team_name}_{OUTPUT}_{path_accuracy}.joblib'
      save_path = f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_{output}{"_F" if inverse else ""}.joblib'
      dump(bst, save_path)
  return accuracy

if __name__ == '__main__':
  teamLookup = team_lookup(db)
  TRAINING_DATA = training_input(SEASONS)
  data1 = pd.DataFrame(TRAINING_DATA)
  data2 = pd.DataFrame(TRAINING_DATA)

  data1.rename(columns=home_rename, inplace=True)
  data1['winB'] = 1 - data1['winB']
  data1['lossB'] = 1 - data1['winB']
  data2.rename(columns=away_rename, inplace=True)
  data2['lossB'] = 1 - data2['winB']
  data = pd.concat([data1, data2], axis=0)
  data.reset_index(drop=True, inplace=True)
  teams = data.groupby('team')

  if TEAM:
    teams = [(TEAM, teams.get_group(TEAM))]

  dtrains = {}
  dtrains_winB = {}
  dtrains_lossB = {}
  for team, team_data in teams:
    x_train = team_data [X_INPUTS_T]
    y_train = team_data [[OUTPUT]].values.ravel()
    y_train_winB = team_data [['winB']].values.ravel()
    y_train_lossB = team_data [['lossB']].values.ravel()
    interaction_constraints = [[x_train.columns.get_loc(j) for j in i] for i in WEIGHTS]
    interaction_constraints_str = str(interaction_constraints).replace(' ','')
    dtrains[team] = {'dm':xgb.DMatrix(x_train, label=y_train),'x_train':x_train,'y_train':y_train,'len':len(x_train),'ic':interaction_constraints_str}
    dtrains_winB[team] = {'dm':xgb.DMatrix(x_train, label=y_train_winB),'x_train':x_train,'y_train':y_train_winB,'len':len(x_train),'ic':interaction_constraints_str}
    dtrains_lossB[team] = {'dm':xgb.DMatrix(x_train, label=y_train_lossB),'x_train':x_train,'y_train':y_train_lossB,'len':len(x_train),'ic':interaction_constraints_str}

  TEST_DATA = test_input(X_INPUTS_T,[OUTPUT],no_format=True)
  test_data1 = pd.DataFrame(TEST_DATA)
  test_data2 = pd.DataFrame(TEST_DATA)
  test_data1.rename(columns=home_rename, inplace=True)
  test_data1['winB'] = 1 - test_data1['winB']
  test_data1['lossB'] = 1 - test_data1['winB']
  test_data2.rename(columns=away_rename, inplace=True)
  test_data2['lossB'] = 1 - test_data2['winB']
  test_data = pd.concat([test_data1, test_data2], axis=0)
  test_data.reset_index(drop=True, inplace=True)
  test_teams = test_data.groupby('team')

  if TEAM:
    test_teams = [(TEAM, test_teams.get_group(TEAM))]

  dtests = {}
  dtests_winB = {}
  dtests_lossB = {}
  for team, team_data in test_teams:
    x_test = team_data [X_INPUTS_T]
    y_test = team_data [[OUTPUT]].values.ravel()
    y_test_winB = team_data [['winB']].values.ravel()
    y_test_lossB = team_data [['lossB']].values.ravel()
    dtests[team] = {'dm':xgb.DMatrix(x_test, label=y_test),'x_test':x_test,'y_test':y_test,'len':len(x_test)}
    dtests_winB[team] = {'dm':xgb.DMatrix(x_test, label=y_test_winB),'x_test':x_test,'y_test':y_test_winB,'len':len(x_test)}
    dtests_lossB[team] = {'dm':xgb.DMatrix(x_test, label=y_test_lossB),'x_test':x_test,'y_test':y_test_lossB,'len':len(x_test)}

  # INTERACTION_CONSTRAINTS = [[data.columns.get_loc(j) for j in i] for i in WEIGHTS]
  # INTERACTION_CONSTRAINTS = f'[' + ", ".join(['[' + ', '.join(str(idx) for idx in group) + ']' for group in INTERACTION_CONSTRAINTS]) + ']'
  # print(INTERACTION_CONSTRAINTS)
  if TRIAL:
    for output in OUTPUTS:
      team_bests = {}
      p_team_bests = []
      dtrains_output = dtrains_winB if output == 'winB' else dtrains_lossB
      for i, (team, dtrain) in enumerate(dtrains_output.items()):
        best = {
          'max_depth': 0,
          'eta': 0,
          'accuracy': 0,
          'training_data': '',
          'inverse': False,
          'distance': 0,
        }
        dtest = dtests_winB[team] if output == 'winB' else dtests_lossB[team]
        team_name = teamLookup[team]['abbrev']
        for max_depth in range(5,51):
          for eta in np.arange(0.01, 0.3, 0.01):
            params = {
              'max_depth': max_depth,  # the maximum depth of each tree
              'eta': eta,  # the training step for each iteration
              'objective': 'binary:logistic',  # binary classification
              'eval_metric': 'logloss',  # evaluation metric
              'device': 'cuda',
              'tree_method': 'hist',
            }
            accuracy = train(db,params,dtrain,dtest,team_name,trial=True,output=output)
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
            p_accuracy = f'{team_name} {output} Accuracy:{((adj_accuracy)*100):.2f}%|eta:{p_eta}|max_depth:{max_depth}||{"inverse" if inverse else "       "}'
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
          'output': output,
          **best_params
        }
        p_best_eta = f'{round(best["eta"],2)}'.ljust(4)
        p_best_accuracy = best['accuracy'] if not best['inverse'] else 1-best['accuracy']
        p_best = f'{team_name} {output} Best: Accuracy:{(p_best_accuracy*100):.2f}%|eta:{p_best_eta}|max_depth:{best["max_depth"]}'
        p_team_bests.append(f'{p_best}||{"inverse" if best["inverse"] else "       "}||{dtest["len"]}')
        print('Inputs:', X_INPUTS_T)
        print('Output:', output)
        print('Params:', best_params)
        train(db,best_params,dtrain,dtest,team_name,inverse=best['inverse'],trial=False,output=output)
    print(team_bests)
    for p_team_best in p_team_bests:
      print(p_team_best)
  else:
    print('Inputs:', X_INPUTS_T)
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
      # team_params['interaction_constraints'] = dtrain['ic']
      train(db,team_params[OUTPUT],dtrain,dtest,team_name,trial=False)