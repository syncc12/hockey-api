import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
from joblib import dump
from constants.inputConstants import X_INPUTS, X_INPUTS_T, Y_OUTPUTS
from constants.constants import LGBM_TEAM_VERSION, LGBM_TEAM_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
from training_input import training_input, test_input
from util.team_helpers import away_rename, home_rename, franchise_map, TEAM_IDS
from util.helpers import team_lookup
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
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

USE_X_INPUTS = X_INPUTS_T

OUTPUT = 'winB'

TRIAL = False
OPTIMIZE = True
DRY_RUN = False

TEAM = False
START_TEAM = False
END_TEAM = False

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

OPTIMIZE_MAX_EVALS = {
  1: {'team': 'NJD', 'max_evals': 1000},
  2: {'team': 'NYI', 'max_evals': 1000},
  3: {'team': 'NYR', 'max_evals': 1000},
  4: {'team': 'PHI', 'max_evals': 1000},
  5: {'team': 'PIT', 'max_evals': 1000},
  6: {'team': 'BOS', 'max_evals': 1000},
  7: {'team': 'BUF', 'max_evals': 1000},
  8: {'team': 'MTL', 'max_evals': 1000},
  9: {'team': 'OTT', 'max_evals': 1000},
  10: {'team': 'TOR', 'max_evals': 1000},
  12: {'team': 'CAR', 'max_evals': 1000},
  13: {'team': 'FLA', 'max_evals': 1000},
  14: {'team': 'TBL', 'max_evals': 1000},
  15: {'team': 'WSH', 'max_evals': 1000},
  16: {'team': 'CHI', 'max_evals': 1000},
  17: {'team': 'DET', 'max_evals': 1000},
  18: {'team': 'NSH', 'max_evals': 1000},
  19: {'team': 'STL', 'max_evals': 1000},
  20: {'team': 'CGY', 'max_evals': 1000},
  21: {'team': 'COL', 'max_evals': 1000},
  22: {'team': 'EDM', 'max_evals': 1000},
  23: {'team': 'VAN', 'max_evals': 1000},
  24: {'team': 'ANA', 'max_evals': 1000},
  25: {'team': 'DAL', 'max_evals': 1000},
  26: {'team': 'LAK', 'max_evals': 1000},
  28: {'team': 'SJS', 'max_evals': 1000},
  29: {'team': 'CBJ', 'max_evals': 1000},
  30: {'team': 'MIN', 'max_evals': 1000},
  52: {'team': 'WPG', 'max_evals': 1000},
  53: {'team': 'ARI', 'max_evals': 1000},
  54: {'team': 'VGK', 'max_evals': 1000},
  55: {'team': 'SEA', 'max_evals': 1000},
}

TEAM_PARAMS = {
  7: {'team': 'BUF', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.2, 'bagging_fraction': 0.4, 'bagging_freq': 19, 'num_leaves': 106, 'learning_rate': 0.047, 'verbosity': -1}},
  8: {'team': 'MTL', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.30000000000000004, 'bagging_fraction': 0.6000000000000001, 'bagging_freq': 47, 'num_leaves': 257, 'learning_rate': 0.296, 'verbosity': -1}},
  9: {'team': 'OTT', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.2, 'bagging_fraction': 0.4, 'bagging_freq': 50, 'num_leaves': 229, 'learning_rate': 0.227, 'verbosity': -1}},
  10: {'team': 'TOR', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.6000000000000001, 'bagging_fraction': 0.7000000000000001, 'bagging_freq': 57, 'num_leaves': 65, 'learning_rate': 0.27, 'verbosity': -1}},
  12: {'team': 'CAR', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.4, 'bagging_fraction': 0.2, 'bagging_freq': 62, 'num_leaves': 167, 'learning_rate': 0.069, 'verbosity': -1}},
  13: {'team': 'FLA', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.9, 'bagging_fraction': 0.5, 'bagging_freq': 44, 'num_leaves': 204, 'learning_rate': 0.139, 'verbosity': -1}},
  14: {'team': 'TBL', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.4, 'bagging_fraction': 0.2, 'bagging_freq': 34, 'num_leaves': 109, 'learning_rate': 0.20500000000000002, 'verbosity': -1}},
  15: {'team': 'WSH', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.4, 'bagging_fraction': 0.4, 'bagging_freq': 11, 'num_leaves': 208, 'learning_rate': 0.256, 'verbosity': -1}},
  16: {'team': 'CHI', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.2, 'bagging_fraction': 0.4, 'bagging_freq': 82, 'num_leaves': 180, 'learning_rate': 0.25, 'verbosity': -1}},
  17: {'team': 'DET', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.1, 'bagging_fraction': 0.5, 'bagging_freq': 69, 'num_leaves': 230, 'learning_rate': 0.201, 'verbosity': -1}},
  18: {'team': 'NSH', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.5, 'bagging_fraction': 0.7000000000000001, 'bagging_freq': 55, 'num_leaves': 112, 'learning_rate': 0.276, 'verbosity': -1}},
  19: {'team': 'STL', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.5, 'bagging_fraction': 0.6000000000000001, 'bagging_freq': 67, 'num_leaves': 228, 'learning_rate': 0.27, 'verbosity': -1}},
  20: {'team': 'CGY', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.1, 'bagging_fraction': 0.1, 'bagging_freq': 60, 'num_leaves': 80, 'learning_rate': 0.121, 'verbosity': -1}},
  21: {'team': 'COL', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.4, 'bagging_fraction': 0.9, 'bagging_freq': 68, 'num_leaves': 211, 'learning_rate': 0.227, 'verbosity': -1}},
  22: {'team': 'EDM', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.5, 'bagging_fraction': 0.6000000000000001, 'bagging_freq': 56, 'num_leaves': 154, 'learning_rate': 0.258, 'verbosity': -1}},
  23: {'team': 'VAN', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.7000000000000001, 'bagging_fraction': 0.7000000000000001, 'bagging_freq': 60, 'num_leaves': 221, 'learning_rate': 0.17300000000000001, 'verbosity': -1}},
  24: {'team': 'ANA', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.9, 'bagging_fraction': 0.7000000000000001, 'bagging_freq': 19, 'num_leaves': 108, 'learning_rate': 0.08, 'verbosity': -1}},
  25: {'team': 'DAL', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.2, 'bagging_fraction': 0.8, 'bagging_freq': 8, 'num_leaves': 216, 'learning_rate': 0.019, 'verbosity': -1}},
  26: {'team': 'LAK', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 1.0, 'bagging_fraction': 0.1, 'bagging_freq': 97, 'num_leaves': 259, 'learning_rate': 0.053, 'verbosity': -1}},
  28: {'team': 'SJS', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.5, 'bagging_fraction': 0.7000000000000001, 'bagging_freq': 23, 'num_leaves': 294, 'learning_rate': 0.178, 'verbosity': -1}},
  29: {'team': 'CBJ', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 60, 'num_leaves': 241, 'learning_rate': 0.045, 'verbosity': -1}},
  30: {'team': 'MIN', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.6000000000000001, 'bagging_fraction': 0.1, 'bagging_freq': 47, 'num_leaves': 40, 'learning_rate': 0.014, 'verbosity': -1}},
  52: {'team': 'WPG', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.8, 'bagging_fraction': 0.2, 'bagging_freq': 17, 'num_leaves': 169, 'learning_rate': 0.23900000000000002, 'verbosity': -1}},
  53: {'team': 'ARI', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 1.0, 'bagging_fraction': 0.6000000000000001, 'bagging_freq': 66, 'num_leaves': 51, 'learning_rate': 0.044, 'verbosity': -1}},
  54: {'team': 'VGK', 'params': {}},
  55: {'team': 'SEA', 'params': {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'feature_fraction': 0.2, 'bagging_fraction': 0.9, 'bagging_freq': 93, 'num_leaves': 280, 'learning_rate': 0.10400000000000001, 'verbosity': -1}},
}


def train(db,params,dtrain,x_test,y_test,num_boost_round=NUM_BOOST_ROUND,team='all',trial=False,verbose=True):
  if not trial and verbose:
    print('Inputs:', len(USE_X_INPUTS))
    print('Output:', OUTPUT)
    print('Params:', params)

  gbm = lgb.train(params, dtrain, num_boost_round=num_boost_round)
  preds = gbm.predict(x_test, num_iteration=gbm.best_iteration)
  predictions = np.where(preds > 0.5, 1, 0)
  accuracy = accuracy_score(y_test, predictions)
  brier_score = brier_score_loss(y_test, preds)

  if not trial:
    model_data = f"{team}:{teamLookup[team]['abbrev']} Accuracy: {'%.2f%%' % (accuracy * 100.0)} | Brier Score: {brier_score}"
    print(model_data)

    if not DRY_RUN:
      save_path = f'models/nhl_ai_v{LGBM_TEAM_FILE_VERSION}_lightgbm_team{team}_{OUTPUT}.joblib'
      dump(gbm, save_path)

    TrainingRecords = db['dev_training_records']

    timestamp = time.time()
    TrainingRecords.insert_one({
      'sport': 'nhl',
      'savedAt': timestamp,
      'version': LGBM_TEAM_VERSION,
      'inputs': USE_X_INPUTS,
      'randomState': RANDOM_STATE,
      'startingSeason': START_SEASON,
      'finalSeason': END_SEASON,
      'seasons': SEASONS,
      'team': team,
      'model': 'LightGBM Classifier (Team)',
      'params': PARAMS,
      'accuracies': {
        OUTPUT: accuracy,
      },
    })
  return accuracy





if __name__ == '__main__':
  teamLookup = team_lookup(db)
  TRAINING_DATA = training_input(SEASONS)
  data1 = pd.DataFrame(TRAINING_DATA)
  data2 = pd.DataFrame(TRAINING_DATA)

  data1.rename(columns=home_rename, inplace=True)
  data1['winB'] = 1 - data1['winB']
  data2.rename(columns=away_rename, inplace=True)
  for franchise in franchise_map:
    data1.loc[data1['team'] == franchise, 'team'] = franchise_map[franchise]
    data2.loc[data2['team'] == franchise, 'team'] = franchise_map[franchise]
  data = pd.concat([data1, data2], axis=0)
  data.reset_index(drop=True, inplace=True)
  all_data = data
  teams = data.groupby('team')

  if TEAM:
    teams = [(TEAM, teams.get_group(TEAM))]
  
  if START_TEAM and not END_TEAM and not TEAM:
    start_index = TEAM_IDS.index(START_TEAM)
    teams = [(team,teams.get_group(team)) for team in TEAM_IDS[start_index:]]
  
  if END_TEAM and not START_TEAM and not TEAM:
    end_index = TEAM_IDS.index(END_TEAM)
    teams = [(team,teams.get_group(team)) for team in TEAM_IDS[:end_index]]
  
  if START_TEAM and END_TEAM and not TEAM:
    start_index = TEAM_IDS.index(START_TEAM)
    end_index = TEAM_IDS.index(END_TEAM)
    teams = [(team,teams.get_group(team)) for team in TEAM_IDS[start_index:end_index]]

  dtrains = {}
  dtests = {}
  dvalidations = {}
  for team, team_data in teams:
    x_train = team_data [USE_X_INPUTS]
    y_train = team_data [[OUTPUT]].values.ravel()
    dtrains[team] = {'dm':lgb.Dataset(x_train, label=y_train),'x_train':x_train,'y_train':y_train,'len':len(x_train)}

  TEST_DATA = test_input(no_format=True)
  test_data1 = pd.DataFrame(TEST_DATA)
  test_data2 = pd.DataFrame(TEST_DATA)
  test_data1.rename(columns=home_rename, inplace=True)
  test_data1['winB'] = 1 - test_data1['winB']
  test_data2.rename(columns=away_rename, inplace=True)
  for franchise in franchise_map:
    test_data1.loc[test_data1['team'] == franchise, 'team'] = franchise_map[franchise]
    test_data2.loc[test_data2['team'] == franchise, 'team'] = franchise_map[franchise]
  test_data = pd.concat([test_data1, test_data2], axis=0)
  test_data.reset_index(drop=True, inplace=True)
  all_test_data = test_data
  test_teams = test_data.groupby('team')

  if TEAM:
    test_teams = [(TEAM, test_teams.get_group(TEAM))]
  
  if START_TEAM and not TEAM:
    start_index = TEAM_IDS.index(START_TEAM)
    test_teams = [(team,test_teams.get_group(team)) for team in TEAM_IDS[start_index:]]

  for team, team_data in test_teams:
    x_test = team_data [USE_X_INPUTS]
    y_test = team_data [[OUTPUT]].values.ravel()
    dtests[team] = {'x_test':x_test,'y_test':y_test,'len':len(x_test)}


  
  if OPTIMIZE:
    MAX_EVALS = 1500
    for team, dtrain in dtrains.items():
      if team in franchise_map:
        team = franchise_map[team]
      dtrain = dtrains[team]['dm']
      x_test = dtests[team]['x_test']
      y_test = dtests[team]['y_test']
      team_name = teamLookup[team]['abbrev']
      print(f'Starting {team}:{team_name} Optimization')
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
            'verbose': -1,
          },
          dtrain,
          num_boost_round=int(params['num_boost_round'])
        )
        preds = model.predict(x_test, num_iteration=model.best_iteration)
        # predictions = np.where(preds > 0.5, 1, 0)
        # accuracy = accuracy_score(y_test, predictions)
        brier_score = brier_score_loss(y_test, preds)
        return {'loss': brier_score, 'status': STATUS_OK}
        # return {'loss': -accuracy, 'status': STATUS_OK}
      
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
        'verbosity': -1,
      }
      train(db,params,dtrain,x_test,y_test,num_boost_round=int(best['num_boost_round']),team=team,trial=False,verbose=False)

  elif TRIAL:
    best = {}
    for i, (team, dtrain) in enumerate(dtrains.items()):
      if team in franchise_map:
        team = franchise_map[team]
      x_test = dtests[team]['x_test']
      y_test = dtests[team]['y_test']
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
                accuracy = train(db,params,dtrain,x_test,y_test,team=team,trial=True)
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
      train(db,best_params,dtrain['dm'],x_test,y_test,team=team,trial=False)
  else:
    for team, dtrain in dtrains.items():
      if team in franchise_map:
        team = franchise_map[team]
      x_test = dtests[team]['x_test']
      y_test = dtests[team]['y_test']
      train(db,TEAM_PARAMS[team]['params'],dtrain['dm'],x_test,y_test,team=team,trial=False)