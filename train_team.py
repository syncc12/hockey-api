import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

import requests
import json
from pymongo import MongoClient
import math
from datetime import datetime
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from joblib import dump, load
import pandas as pd
from training_input import training_input
from multiprocessing import Pool
from util.training_data import season_training_data, game_training_data, update_training_data
from constants.inputConstants import X_INPUTS_T, Y_OUTPUTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util.helpers import latestIDs, team_lookup
import time
from constants.constants import VERSION, FILE_VERSION, TEST_DATA_VERSION, TEST_DATA_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE

RE_PULL = False
UPDATE = False
OVERRIDE = True
DRY_RUN = True

def train(db, inData):
  # imputer = SimpleImputer(strategy='constant', fill_value=-1)
  data1 = pd.DataFrame(inData)
  data2 = pd.DataFrame(inData)

  data_rename_1 = {
    'homeTeam': 'team',
    'awayTeam': 'opponent',
    'homeScore': 'score',
    'awayScore': 'opponentScore',
    'homeHeadCoach': 'headCoach',
    'awayHeadCoach': 'opponentHeadCoach',
    'homeForwardAverage': 'forwardAverage',
    'homeDefenseAverage': 'defenseAverage',
    'homeGoalieAverage': 'goalieAverage',
    'awayForwardAverage': 'opponentForwardAverage',
    'awayDefenseAverage': 'opponentDefenseAverage',
    'awayGoalieAverage': 'opponentGoalieAverage',
    'homeForwardAverageAge': 'forwardAverageAge',
    'homeDefenseAverageAge': 'defenseAverageAge',
    'homeGoalieAverageAge': 'goalieAverageAge',
    'awayForwardAverageAge': 'opponentForwardAverageAge',
    'awayDefenseAverageAge': 'opponentDefenseAverageAge',
    'awayGoalieAverageAge': 'opponentGoalieAverageAge',
    'winner': 'win',
    'winnerB': 'winB',
  }
  data_rename_2 = {
    'homeTeam': 'opponent',
    'awayTeam': 'team',
    'homeScore': 'opponentScore',
    'awayScore': 'score',
    'homeHeadCoach': 'opponentHeadCoach',
    'awayHeadCoach': 'headCoach',
    'homeForwardAverage': 'opponentForwardAverage',
    'homeDefenseAverage': 'opponentDefenseAverage',
    'homeGoalieAverage': 'opponentGoalieAverage',
    'awayForwardAverage': 'forwardAverage',
    'awayDefenseAverage': 'defenseAverage',
    'awayGoalieAverage': 'goalieAverage',
    'homeForwardAverageAge': 'opponentForwardAverageAge',
    'homeDefenseAverageAge': 'opponentDefenseAverageAge',
    'homeGoalieAverageAge': 'opponentGoalieAverageAge',
    'awayForwardAverageAge': 'forwardAverageAge',
    'awayDefenseAverageAge': 'defenseAverageAge',
    'awayGoalieAverageAge': 'goalieAverageAge',
    'winner': 'win',
    'winnerB': 'winB',
  }

  data1.rename(columns=data_rename_1, inplace=True)
  data1['winB'] = 1 - data1['winB']
  data1['lossB'] = 1 - data1['winB']
  data2.rename(columns=data_rename_2, inplace=True)
  data2['lossB'] = 1 - data2['winB']
  data = pd.concat([data1, data2], axis=0)
  data.reset_index(drop=True, inplace=True)



  TEST_DATA = load(f'test_data/test_data_v{TEST_DATA_FILE_VERSION}.joblib')
  test_data1 = pd.DataFrame(TEST_DATA)
  test_data2 = pd.DataFrame(TEST_DATA)
  test_data1.rename(columns=data_rename_1, inplace=True)
  test_data1['winB'] = 1 - test_data1['winB']
  test_data1['lossB'] = 1 - test_data1['winB']
  test_data2.rename(columns=data_rename_2, inplace=True)
  test_data2['lossB'] = 1 - test_data2['winB']
  test_data = pd.concat([test_data1, test_data2], axis=0)
  test_data.reset_index(drop=True, inplace=True)
  test_teams = test_data.groupby('team')

  test_datas = {}
  for team, team_data in test_teams:
    test_datas[team] = team_data

  teamLookup = team_lookup(db)

  teams = data.groupby('team')

  for team, team_Data in teams:
    team_name = teamLookup[team]['abbrev']
    x_train = team_Data [X_INPUTS_T]
    # imputer.fit(x)
    # x = imputer.transform(x)
    # y_train_win = team_Data [['win']].values.ravel()
    y_train_winB = team_Data [['winB']].values.ravel()
    y_train_lossB = team_Data [['lossB']].values.ravel()
    # y_train_score = team_Data [['score']].values.ravel()
    # y_train_opponentScore = team_Data [['opponentScore']].values.ravel()
    # y_train_totalGoals = team_Data [['totalGoals']].values.ravel()
    # y_train_goalDifferential = team_Data [['goalDifferential']].values.ravel()


    x_test = test_datas[team] [X_INPUTS_T]
    # y_test_win = test_data [['win']].values.ravel()
    y_test_winB = test_datas[team] [['winB']].values.ravel()
    y_test_lossB = test_datas[team] [['lossB']].values.ravel()
    # y_test_score = test_data [['score']].values.ravel()
    # y_test_opponentScore = test_data [['opponentScore']].values.ravel()
    # y_test_totalGoals = test_data [['totalGoals']].values.ravel()
    # y_test_goalDifferential = test_data [['goalDifferential']].values.ravel()

    # clf_win = RandomForestClassifier(random_state=RANDOM_STATE)
    clf_winB = RandomForestClassifier(random_state=RANDOM_STATE)
    clf_lossB = RandomForestClassifier(random_state=RANDOM_STATE)
    # clf_score = RandomForestClassifier(random_state=RANDOM_STATE)
    # clf_opponentScore = RandomForestClassifier(random_state=RANDOM_STATE)
    # clf_totalGoals = RandomForestClassifier(random_state=RANDOM_STATE)
    # clf_goalDifferential = RandomForestClassifier(random_state=RANDOM_STATE)

    # clf_win.fit(x_train,y_train_win)
    clf_winB.fit(x_train,y_train_winB)
    clf_lossB.fit(x_train,y_train_lossB)
    # clf_score.fit(x_train,y_train_score)
    # clf_opponentScore.fit(x_train,y_train_opponentScore)
    # clf_totalGoals.fit(x_train,y_train_totalGoals)
    # clf_goalDifferential.fit(x_train,y_train_goalDifferential)

    # predictions_win = clf_win.predict(x_test)
    predictions_winB = clf_winB.predict(x_test)
    predictions_lossB = clf_lossB.predict(x_test)
    # predictions_score = clf_score.predict(x_test)
    # predictions_opponentScore = clf_opponentScore.predict(x_test)
    # predictions_totalGoals = clf_totalGoals.predict(x_test)
    # predictions_goalDifferential = clf_goalDifferential.predict(x_test)

    # win_accuracy = accuracy_score(y_test_win, predictions_win)
    winB_accuracy = accuracy_score(y_test_winB, predictions_winB)
    lossB_accuracy = accuracy_score(y_test_lossB, predictions_lossB)
    # score_accuracy = accuracy_score(y_test_score, predictions_score)
    # opponentScore_accuracy = accuracy_score(y_test_opponentScore, predictions_opponentScore)
    # totalGoals_accuracy = accuracy_score(y_test_totalGoals, predictions_totalGoals)
    # goalDifferential_accuracy = accuracy_score(y_test_goalDifferential, predictions_goalDifferential)
    # print(f"{team_name} Win Accuracy:", f"{(win_accuracy*100):.2f}%")
    print(f"{team_name} Win Binary Accuracy:", f"{(winB_accuracy*100):.2f}%")
    print(f"{team_name} Loss Binary Accuracy:", f"{(lossB_accuracy*100):.2f}%")
    # print(f"{team_name} Score Accuracy:", f"{(score_accuracy*100):.2f}%")
    # print(f"{team_name} Opponent Score Accuracy:", f"{(opponentScore_accuracy*100):.2f}%")
    # print(f"{team_name} Total Goals Accuracy:", f"{(totalGoals_accuracy*100):.2f}%")
    # print(f"{team_name} Goal Differential Accuracy:", f"{(goalDifferential_accuracy*100):.2f}%")
    
    if not DRY_RUN:
      TrainingRecords = db['dev_training_records']
      # Metadata = db['dev_metadata']

      timestamp = time.time()
      TrainingRecords.insert_one({
        'savedAt': timestamp,
        'lastTrainedId': inData[len(inData)-1]['id'],
        'version': VERSION,
        'inputs': X_INPUTS_T,
        'outputs': Y_OUTPUTS,
        'randomState': RANDOM_STATE,
        'startingSeason': START_SEASON,
        'finalSeason': END_SEASON,
        'team': team,
        'teamName': team_name,
        'model': 'Random Forest Classifier (Teams)',
        'file': 'train_team.py',
        'accuracies': {
          # 'win': win_accuracy,
          'winB': winB_accuracy,
          'lossB': lossB_accuracy,
          # 'score': score_accuracy,
          # 'opponentScore': opponentScore_accuracy,
          # 'totalGoals': totalGoals_accuracy,
          # 'goalDifferential': goalDifferential_accuracy,
        },
      })

      # dump(clf, f'models/nhl_ai_v{FILE_VERSION}.joblib')
      # dump(clf_win, f'models/nhl_ai_v{FILE_VERSION}_{team_name}_win.joblib')
      dump(clf_winB, f'models/nhl_ai_v{FILE_VERSION}_{team_name}_winB.joblib')
      # dump(clf_score, f'models/nhl_ai_v{FILE_VERSION}_{team_name}_score.joblib')
      # dump(clf_opponentScore, f'models/nhl_ai_v{FILE_VERSION}_{team_name}_opponentScore.joblib')
      # dump(clf_totalGoals, f'models/nhl_ai_v{FILE_VERSION}_{team_name}_totalGoals.joblib')
      # dump(clf_goalDifferential, f'models/nhl_ai_v{FILE_VERSION}_{team_name}_goalDifferential.joblib')


dir_path = f'training_data/v{VERSION}'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

tdList = os.listdir(f'training_data/v{VERSION}')

USE_SEASONS = True
SKIP_SEASONS = [int(td.replace(f'training_data_v{FILE_VERSION}_','').replace('.joblib','')) for td in tdList] if len(tdList) > 0 and not f'training_data_v{FILE_VERSION}.joblib' in os.listdir('training_data') else []

LATEST_SEASON = 20232024
MAX_ID = 2023020514

if __name__ == '__main__':
  # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
  # client = MongoClient(db_url)
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]
  if OVERRIDE:
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
      20162017,
      20172018,
      20182019,
      20192020,
      20202021,
      20212022,
      20222023,
    ]
    # TRAINING_DATAS = [load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib') for season in SEASONS]
    # TRAINING_DATA = np.concatenate(TRAINING_DATAS).tolist()
    # print('Seasons Loaded')
    TRAINING_DATA = training_input(SEASONS)
    train(db, TRAINING_DATA)
  else:
    if RE_PULL:
      if USE_SEASONS:
        seasons = list(db["dev_seasons"].find(
          {
            'seasonId': {
              '$gte': START_SEASON,
              '$lte': END_SEASON,
            }
          },
          {'_id':0,'seasonId': 1}
        ))
        seasons = [int(season['seasonId']) for season in seasons]
        print(seasons)
        if (len(SKIP_SEASONS) > 0):
          for season in SKIP_SEASONS:
            seasons.remove(season)
          print(seasons)
      else:
        ids = latestIDs()
        startID = ids['saved']['training']
        endID = MAX_ID
        games = list(db["dev_games"].find(
          {'id':{'$gte':startID,'$lt':endID+1}},
          # {'id':{'$lt':endID+1}},
          {'id': 1, '_id': 0}
        ))
      
      pool = Pool(processes=4)
      if USE_SEASONS:
        result = pool.map(season_training_data,seasons)
      else:
        result = pool.map(game_training_data,games)
      if len(SKIP_SEASONS) > 0:
        for skip_season in SKIP_SEASONS:
          season_data = load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{skip_season}.joblib')
          result.append(season_data)
      result = np.concatenate(result).tolist()
      pool.close()
      dump(result,f'training_data/training_data_v{FILE_VERSION}.joblib')
      f = open('training_data/training_data_text.txt', 'w')
      f.write(json.dumps(result[len(result)-200:len(result)]))
    else:
      training_data_path = f'training_data/training_data_v{FILE_VERSION}.joblib'
      print(training_data_path)
      result = load(training_data_path)
      f = open('training_data/training_data_text.txt', 'w')
      f.write(json.dumps(result[len(result)-200:len(result)]))
    print('Games Collected')
    train(db, result)