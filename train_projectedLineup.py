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
from util.training_data import season_training_data_projectedLineup, game_training_data, update_training_data
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util.helpers import latestIDs
import time
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
from scipy.stats import mode


RE_PULL = True
UPDATE = False
# VERSION = 6

def train(db, inData):
  imputer = SimpleImputer(strategy='constant', fill_value=-1)
  winner_classifiers = []
  winnerB_classifiers = []
  homeScore_classifiers = []
  awayScore_classifiers = []
  totalGoals_classifiers = []
  goalDifferential_classifiers = []

  counter = 1
  for subset in inData:

    data = pd.DataFrame(subset)
    x = data [X_INPUTS]
    # y = data [['homeScore','awayScore','winner','totalGoals','goalDifferential']].values
    y_winner = data [['winner']].values.ravel()
    y_winnerB = data [['winnerB']].values.ravel()
    y_homeScore = data [['homeScore']].values.ravel()
    y_awayScore = data [['awayScore']].values.ravel()
    y_totalGoals = data [['totalGoals']].values.ravel()
    y_goalDifferential = data [['goalDifferential']].values.ravel()
    imputer.fit(x)
    x = imputer.transform(x)
    x_winner = x
    x_winnerB = x
    x_homeScore = x
    x_awayScore = x
    x_totalGoals = x
    x_goalDifferential = x

    x_train_winner, x_test_winner, y_train_winner, y_test_winner = train_test_split(x_winner, y_winner, test_size=0.2, random_state=RANDOM_STATE)
    x_train_winnerB, x_test_winnerB, y_train_winnerB, y_test_winnerB = train_test_split(x_winnerB, y_winnerB, test_size=0.2, random_state=RANDOM_STATE)
    x_train_homeScore, x_test_homeScore, y_train_homeScore, y_test_homeScore = train_test_split(x_homeScore, y_homeScore, test_size=0.2, random_state=RANDOM_STATE)
    x_train_awayScore, x_test_awayScore, y_train_awayScore, y_test_awayScore = train_test_split(x_awayScore, y_awayScore, test_size=0.2, random_state=RANDOM_STATE)
    x_train_totalGoals, x_test_totalGoals, y_train_totalGoals, y_test_totalGoals = train_test_split(x_totalGoals, y_totalGoals, test_size=0.2, random_state=RANDOM_STATE)
    x_train_goalDifferential, x_test_goalDifferential, y_train_goalDifferential, y_test_goalDifferential = train_test_split(x_goalDifferential, y_goalDifferential, test_size=0.2, random_state=RANDOM_STATE)
    
    clf_winner = RandomForestClassifier(random_state=RANDOM_STATE)
    clf_winnerB = RandomForestClassifier(random_state=RANDOM_STATE)
    clf_homeScore = RandomForestClassifier(random_state=RANDOM_STATE)
    clf_awayScore = RandomForestClassifier(random_state=RANDOM_STATE)
    clf_totalGoals = RandomForestClassifier(random_state=RANDOM_STATE)
    clf_goalDifferential = RandomForestClassifier(random_state=RANDOM_STATE)
    
    clf_winner.fit(x_train_winner,y_train_winner)
    print(f'winner - fit {counter}/{len(inData)}')
    clf_winnerB.fit(x_train_winnerB,y_train_winnerB)
    print(f'winnerB - fit {counter}/{len(inData)}')
    clf_homeScore.fit(x_train_homeScore,y_train_homeScore)
    print(f'homeScore - fit {counter}/{len(inData)}')
    clf_awayScore.fit(x_train_awayScore,y_train_awayScore)
    print(f'awayScore - fit {counter}/{len(inData)}')
    clf_totalGoals.fit(x_train_totalGoals,y_train_totalGoals)
    print(f'totalGoals - fit {counter}/{len(inData)}')
    clf_goalDifferential.fit(x_train_goalDifferential,y_train_goalDifferential)
    print(f'goalDifferential - fit {counter}/{len(inData)}')

    counter += 1
    
    winner_classifiers.append((clf_winner,x_test_winner))
    winnerB_classifiers.append((clf_winnerB,x_test_winnerB))
    homeScore_classifiers.append((clf_homeScore,x_test_homeScore))
    awayScore_classifiers.append((clf_awayScore,x_test_awayScore))
    totalGoals_classifiers.append((clf_totalGoals,x_test_totalGoals))
    goalDifferential_classifiers.append((clf_goalDifferential,x_test_goalDifferential))

  predictions_winner = np.array([clf.predict(x_test) for clf, x_test in winner_classifiers])
  predictions_winnerB = np.array([clf.predict(x_test) for clf, x_test in winnerB_classifiers])
  predictions_homeScore = np.array([clf.predict(x_test) for clf, x_test in homeScore_classifiers])
  predictions_awayScore = np.array([clf.predict(x_test) for clf, x_test in awayScore_classifiers])
  predictions_totalGoals = np.array([clf.predict(x_test) for clf, x_test in totalGoals_classifiers])
  predictions_goalDifferential = np.array([clf.predict(x_test) for clf, x_test in goalDifferential_classifiers])

  winner_final_predictions, _ = mode(predictions_winner, axis=0)
  winnerB_final_predictions, _ = mode(predictions_winnerB, axis=0)
  homeScore_final_predictions, _ = mode(predictions_homeScore, axis=0)
  awayScore_final_predictions, _ = mode(predictions_awayScore, axis=0)
  totalGoals_final_predictions, _ = mode(predictions_totalGoals, axis=0)
  goalDifferential_final_predictions, _ = mode(predictions_goalDifferential, axis=0)

  winner_accuracy = np.mean(winner_final_predictions.flatten() == y_test_winner)
  winnerB_accuracy = np.mean(winnerB_final_predictions.flatten() == y_test_winnerB)
  homeScore_accuracy = np.mean(homeScore_final_predictions.flatten() == y_test_homeScore)
  awayScore_accuracy = np.mean(awayScore_final_predictions.flatten() == y_test_awayScore)
  totalGoals_accuracy = np.mean(totalGoals_final_predictions.flatten() == y_test_totalGoals)
  goalDifferential_accuracy = np.mean(goalDifferential_final_predictions.flatten() == y_test_goalDifferential)


  print("Winner Accuracy:", winner_accuracy)
  print("Winner Binary Accuracy:", winnerB_accuracy)
  print("Home Score Accuracy:", homeScore_accuracy)
  print("Away Score Accuracy:", awayScore_accuracy)
  print("Total Goals Accuracy:", totalGoals_accuracy)
  print("Goal Differential Accuracy:", goalDifferential_accuracy)
  TrainingRecords = db['dev_training_records']

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
    'projectedLineup': True,
    'model': 'Stacked Hard Voting Random Forest Classifier',
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
  dump(winner_classifiers, f'models/nhl_ai_v{FILE_VERSION}_winner_projectedLineup.joblib')
  dump(winnerB_classifiers, f'models/nhl_ai_v{FILE_VERSION}_winnerB_projectedLineup.joblib')
  dump(homeScore_classifiers, f'models/nhl_ai_v{FILE_VERSION}_homeScore_projectedLineup.joblib')
  dump(awayScore_classifiers, f'models/nhl_ai_v{FILE_VERSION}_awayScore_projectedLineup.joblib')
  dump(totalGoals_classifiers, f'models/nhl_ai_v{FILE_VERSION}_totalGoals_projectedLineup.joblib')
  dump(goalDifferential_classifiers, f'models/nhl_ai_v{FILE_VERSION}_goalDifferential_projectedLineup.joblib')


dir_path = f'training_data/v{VERSION}'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

dir_path = f'training_data/v{VERSION}/projected_lineup'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

tdList = os.listdir(f'training_data/v{VERSION}/projected_lineup')

SKIP_SEASONS = [int(td.replace(f'training_data_v{FILE_VERSION}_','').replace('_projectedLineup','').replace('.joblib','')) for td in tdList] if len(tdList) > 0 and not f'training_data_v{FILE_VERSION}_projectedLineup.joblib' in os.listdir('training_data') else []

LATEST_SEASON = 20232024
MAX_ID = 2023020514

if __name__ == '__main__':
  # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
  # client = MongoClient(db_url)
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]
  if RE_PULL:
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

    pool = Pool(processes=4)
    results = pool.map(season_training_data_projectedLineup,seasons)
    if len(SKIP_SEASONS) > 0:
      for skip_season in SKIP_SEASONS:
        season_data = load(f'training_data/v{VERSION}/projected_lineup/training_data_v{FILE_VERSION}_{skip_season}_projectedLineup.joblib')
        results.append(season_data)
    result = [[],[],[],[]]
    for r in results:
      for r1 in r:
        for lu in range(0,len(r1)):
          result[lu].append(r1[lu])
    pool.close()
    dump(result,f'training_data/training_data_v{FILE_VERSION}_projectedLineup.joblib')
    f = open('training_data/training_data_text.txt', 'w')
    f.write(json.dumps(result[len(result)-200:len(result)]))
  else:
    training_data_path = f'training_data/training_data_v{FILE_VERSION}_projectedLineup.joblib'
    print(training_data_path)
    result = load(training_data_path)
    f = open('training_data/training_data_text.txt', 'w')
    f.write(json.dumps(result[len(result)-200:len(result)]))
  print('Games Collected')
  train(db, result)