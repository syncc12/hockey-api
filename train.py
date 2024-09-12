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
from multiprocessing import Pool
from util.training_data import season_training_data, game_training_data, update_training_data
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util.helpers import latestIDs, all_combinations
import time
from constants.constants import VERSION, FILE_VERSION, TEST_DATA_VERSION, TEST_DATA_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE

RE_PULL = False
UPDATE = False
OVERRIDE = True

def train(db, inData):
  imputer = SimpleImputer(strategy='constant', fill_value=-1)
  data = pd.DataFrame(inData)
  x = data [X_INPUTS]

  y_winner = data [['winner']].values.ravel()
  y_winnerB = data [['winnerB']].values.ravel()
  y_homeScore = data [['homeScore']].values.ravel()
  y_awayScore = data [['awayScore']].values.ravel()
  y_totalGoals = data [['totalGoals']].values.ravel()
  y_goalDifferential = data [['goalDifferential']].values.ravel()

  y_train_winner = y_winner
  y_train_winnerB = y_winnerB
  y_train_homeScore = y_homeScore
  y_train_awayScore = y_awayScore
  y_train_totalGoals = y_totalGoals
  y_train_goalDifferential = y_goalDifferential

  imputer.fit(x)
  x = imputer.transform(x)
  x_winner = x
  x_winnerB = x
  x_homeScore = x
  x_awayScore = x
  x_totalGoals = x
  x_goalDifferential = x

  x_train_winner = x_winner
  x_train_winnerB = x_winnerB
  x_train_homeScore = x_homeScore
  x_train_awayScore = x_awayScore
  x_train_totalGoals = x_totalGoals
  x_train_goalDifferential = x_goalDifferential

  
  TEST_DATA = load(f'test_data/test_data_v{TEST_DATA_FILE_VERSION}.joblib')
  test_data = pd.DataFrame(TEST_DATA)
  x_test_winner = test_data [X_INPUTS]
  x_test_winnerB = test_data [X_INPUTS]
  x_test_homeScore = test_data [X_INPUTS]
  x_test_awayScore = test_data [X_INPUTS]
  x_test_totalGoals = test_data [X_INPUTS]
  x_test_goalDifferential = test_data [X_INPUTS]
  y_test_winner = test_data [['winner']].values.ravel()
  y_test_winnerB = test_data [['winnerB']].values.ravel()
  y_test_homeScore = test_data [['homeScore']].values.ravel()
  y_test_awayScore = test_data [['awayScore']].values.ravel()
  y_test_totalGoals = test_data [['totalGoals']].values.ravel()
  y_test_goalDifferential = test_data [['goalDifferential']].values.ravel()

  # x_train_winner, x_test_winner, y_train_winner, y_test_winner = train_test_split(x_winner, y_winner, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_winnerB, x_test_winnerB, y_train_winnerB, y_test_winnerB = train_test_split(x_winnerB, y_winnerB, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeScore, x_test_homeScore, y_train_homeScore, y_test_homeScore = train_test_split(x_homeScore, y_homeScore, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayScore, x_test_awayScore, y_train_awayScore, y_test_awayScore = train_test_split(x_awayScore, y_awayScore, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_totalGoals, x_test_totalGoals, y_train_totalGoals, y_test_totalGoals = train_test_split(x_totalGoals, y_totalGoals, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_goalDifferential, x_test_goalDifferential, y_train_goalDifferential, y_test_goalDifferential = train_test_split(x_goalDifferential, y_goalDifferential, test_size=0.2, random_state=RANDOM_STATE)
  
  # clf = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_winner = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_winnerB = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_homeScore = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_awayScore = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_totalGoals = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_goalDifferential = RandomForestClassifier(random_state=RANDOM_STATE)

  # clf.fit(x,y)
  clf_winner.fit(x_train_winner,y_train_winner)
  clf_winnerB.fit(x_train_winnerB,y_train_winnerB)
  clf_homeScore.fit(x_train_homeScore,y_train_homeScore)
  clf_awayScore.fit(x_train_awayScore,y_train_awayScore)
  clf_totalGoals.fit(x_train_totalGoals,y_train_totalGoals)
  clf_goalDifferential.fit(x_train_goalDifferential,y_train_goalDifferential)

  predictions_winner = clf_winner.predict(x_test_winner)
  predictions_winnerB = clf_winnerB.predict(x_test_winnerB)
  predictions_homeScore = clf_homeScore.predict(x_test_homeScore)
  predictions_awayScore = clf_awayScore.predict(x_test_awayScore)
  predictions_totalGoals = clf_totalGoals.predict(x_test_totalGoals)
  predictions_goalDifferential = clf_goalDifferential.predict(x_test_goalDifferential)

  winner_accuracy = accuracy_score(y_test_winner, predictions_winner)
  winnerB_accuracy = accuracy_score(y_test_winnerB, predictions_winnerB)
  homeScore_accuracy = accuracy_score(y_test_homeScore, predictions_homeScore)
  awayScore_accuracy = accuracy_score(y_test_awayScore, predictions_awayScore)
  totalGoals_accuracy = accuracy_score(y_test_totalGoals, predictions_totalGoals)
  goalDifferential_accuracy = accuracy_score(y_test_goalDifferential, predictions_goalDifferential)
  print("Winner Accuracy:", winner_accuracy)
  print("Winner Binary Accuracy:", winnerB_accuracy)
  print("Home Score Accuracy:", homeScore_accuracy)
  print("Away Score Accuracy:", awayScore_accuracy)
  print("Total Goals Accuracy:", totalGoals_accuracy)
  print("Goal Differential Accuracy:", goalDifferential_accuracy)
  
  TrainingRecords = db['dev_training_records']
  # Metadata = db['dev_metadata']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'savedAt': timestamp,
    'lastTrainedId': inData[len(inData)-1]['id'],
    'version': VERSION,
    'inputs': X_INPUTS,
    'outputs': Y_OUTPUTS,
    'randomState': RANDOM_STATE,
    'startingSeason': START_SEASON,
    'finalSeason': END_SEASON,
    'model': 'Random Forest Classifier',
    'accuracies': {
      'winner': winner_accuracy,
      'winnerB': winnerB_accuracy,
      'homeScore': homeScore_accuracy,
      'awayScore': awayScore_accuracy,
      'totalGoals': totalGoals_accuracy,
      'goalDifferential': goalDifferential_accuracy,
    },
  })

  # dump(clf, f'models/nhl_ai_v{FILE_VERSION}.joblib')
  dump(clf_winner, f'models/nhl_ai_v{FILE_VERSION}_winner.joblib')
  dump(clf_winnerB, f'models/nhl_ai_v{FILE_VERSION}_winnerB.joblib')
  dump(clf_homeScore, f'models/nhl_ai_v{FILE_VERSION}_homeScore.joblib')
  dump(clf_awayScore, f'models/nhl_ai_v{FILE_VERSION}_awayScore.joblib')
  dump(clf_totalGoals, f'models/nhl_ai_v{FILE_VERSION}_totalGoals.joblib')
  dump(clf_goalDifferential, f'models/nhl_ai_v{FILE_VERSION}_goalDifferential.joblib')


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
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
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
      # 20162017,
      # 20172018,
      20182019,
      20192020,
      20202021,
      20212022,
      20222023,
    ]
    TRAINING_DATAS = [load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib') for season in SEASONS]
    TRAINING_DATA = np.concatenate(TRAINING_DATAS).tolist()
    print('Seasons Loaded')
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