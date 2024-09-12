import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
import lightgbm as lgb
from constants.constants import MLB_LGBM_TEAM_VERSION, MLB_LGBM_TEAM_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from pages.mlb.inputs import X_INPUTS_MLB_T, ENCODE_COLUMNS, mlb_training_input, mlb_test_input
from pages.mlb.mlb_helpers import team_lookup, away_rename, home_rename

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]
mlb_db = client["mlb"]

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

USE_X_INPUTS_MLB = X_INPUTS_MLB_T

OUTPUT = 'winner'

USE_TEST_SPLIT = False
USE_VALIDATION = False

TRIAL = False
OPTIMIZE = True
DRY_RUN = False
TEAM = False # 143

NUM_BOOST_ROUND = 100
OVERFIT_CEILING = 1.0

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


def train(db,params,dtrain,x_test,y_test,num_boost_round=NUM_BOOST_ROUND,team='all',trial=False):
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
      save_path = f'models/mlb_ai_v{MLB_LGBM_TEAM_FILE_VERSION}_lightgbm_team{team}_{OUTPUT}.joblib'
      dump(gbm, save_path)

    TrainingRecords = db['dev_training_records']

    timestamp = time.time()
    TrainingRecords.insert_one({
      'sport': 'mlb',
      'savedAt': timestamp,
      'version': MLB_LGBM_TEAM_VERSION,
      'inputs': USE_X_INPUTS_MLB,
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



def rename_home(col):
  return col.replace('homeTeam','team').replace('awayTeam','opponent').replace('home','').replace('away','opponent')

def rename_away(col):
  return col.replace('awayTeam','team').replace('homeTeam','opponent').replace('away','').replace('home','opponent')


if __name__ == '__main__':
  teamLookup = team_lookup(mlb_db,only_active_mlb=True)
  if not USE_TEST_SPLIT or USE_VALIDATION:
    SEASONS = [season for season in SEASONS if season not in TEST_SEASONS]
    print(f'{"Test" if not USE_VALIDATION else "Validation"} Seasons:', TEST_SEASONS)
  print('Training Seasons:', SEASONS)
  TRAINING_DATA = mlb_training_input(SEASONS)
  data1 = pd.DataFrame(TRAINING_DATA)
  data2 = pd.DataFrame(TRAINING_DATA)

  data1.rename(columns=rename_home, inplace=True)
  data1['winner'] = 1 - data1['winner']
  data2.rename(columns=rename_away, inplace=True)
  data = pd.concat([data1, data2], axis=0)
  data.reset_index(drop=True, inplace=True)
  for column in ENCODE_COLUMNS:
    encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
    data = data[data[column] != -1]
    data[column] = encoder.transform(data[column])
  keep_teams = list(teamLookup.keys())
  data = data[data['team'].isin(keep_teams)]
  data = data[data['opponent'].isin(keep_teams)]
  all_data = data
  if USE_TEST_SPLIT or USE_VALIDATION:
    data, test_data = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    dump(test_data,f'pages/mlb/data/test_data_lgbm_team.joblib')
    test_teams = data.groupby('team')
  teams = data.groupby('team')

  if TEAM:
    teams = [(TEAM, teams.get_group(TEAM))]

  dtrains = {}
  dtests = {}
  dvalidations = {}
  for team, team_data in teams:
    x_train = team_data [USE_X_INPUTS_MLB]
    y_train = team_data [[OUTPUT]].values.ravel()
    dtrains[team] = {'dm':lgb.Dataset(x_train, label=y_train),'x_train':x_train,'y_train':y_train,'len':len(x_train)}

  if USE_TEST_SPLIT or USE_VALIDATION:
    for team, test_data in test_teams:
      x_test = test_data [USE_X_INPUTS_MLB]
      y_test = test_data [[OUTPUT]].values.ravel()
      dtests[team] = {'x_test':x_test,'y_test':y_test,'len':len(x_test)}
    pass
  if not USE_TEST_SPLIT or USE_VALIDATION:
    TEST_DATA = mlb_test_input(TEST_SEASONS)
    test_data1 = pd.DataFrame(TEST_DATA)
    test_data2 = pd.DataFrame(TEST_DATA)
    test_data1.rename(columns=rename_home, inplace=True)
    test_data1['winner'] = 1 - test_data1['winner']
    test_data2.rename(columns=rename_away, inplace=True)
    test_data = pd.concat([test_data1, test_data2], axis=0)
    test_data.reset_index(drop=True, inplace=True)
    for column in ENCODE_COLUMNS:
      encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
      test_data = test_data[test_data[column] != -1]
      test_data[column] = encoder.transform(test_data[column])
    test_data = test_data[test_data['team'].isin(keep_teams)]
    test_data = test_data[test_data['opponent'].isin(keep_teams)]
    dump(test_data,f'pages/mlb/data/test_data_lgbm_team.joblib')
    all_test_data = test_data
    test_teams = test_data.groupby('team')

    if TEAM:
      test_teams = [(TEAM, test_teams.get_group(TEAM))]

    for team, team_data in test_teams:
      x_test = team_data [USE_X_INPUTS_MLB]
      y_test = team_data [[OUTPUT]].values.ravel()
      if USE_VALIDATION:
        dvalidations[team] = {'x_test':x_test,'y_test':y_test,'len':len(x_test)}
      else:
        dtests[team] = {'x_test':x_test,'y_test':y_test,'len':len(x_test)}

  
  if OPTIMIZE:
    MAX_EVALS = 500
    for team, dtrain in dtrains.items():
      dtrain = dtrains[team]['dm']
      x_test = dtests[team]['x_test']
      y_test = dtests[team]['y_test']
      team_name = teamLookup[team]['abbrev']
      print(f'Starting {team_name} Optimization')
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
            'verbose': -1,
          },
          dtrain,
          num_boost_round=int(params['num_boost_round'])
        )
        preds = model.predict(x_test, num_iteration=model.best_iteration)
        predictions = np.where(preds > 0.5, 1, 0)
        accuracy = accuracy_score(y_test, predictions)
        measured_accuracy = 0 if accuracy > OVERFIT_CEILING else accuracy
        return {'loss': -measured_accuracy, 'status': STATUS_OK}
      
      trials = Trials()
      best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
      p_best_params = {**best, 'num_boost_rounds': NUM_BOOST_ROUND}
      print(f"Best parameters: {p_best_params}")
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
        'verbose': -1,
      }
      if USE_VALIDATION:
        x_test = dvalidations[team]['x_test']
        y_test = dvalidations[team]['y_test']
      train(db,params,dtrain,x_test,y_test,num_boost_round=int(best['num_boost_round']),team=team,trial=False)

  elif TRIAL:
    best = {}
    for i, (team, dtrain) in enumerate(dtrains.items()):
      x_test = dtests[team]['x_test']
      y_test = dtests[team]['y_test']
      best[team] = {
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
      if USE_VALIDATION:
        x_test = dvalidations[team]['x_test']
        y_test = dvalidations[team]['y_test']
      train(db,best_params,dtrain['dm'],x_test,y_test,team=team,trial=False)
  else:
    for team, dtrain in dtrains.items():
      if USE_VALIDATION:
        x_test = dvalidations[team]['x_test']
        y_test = dvalidations[team]['y_test']
      else:
        x_test = dtests[team]['x_test']
        y_test = dtests[team]['y_test']
      train(db,PARAMS,dtrain['dm'],x_test,y_test,team=team,trial=False)