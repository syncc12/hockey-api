import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\constants')

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
from constants.inputConstants import X_V4_INPUTS, Y_V4_OUTPUTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util.helpers import latestIDs
import time
from inputs.nfl.inputs import master_inputs


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["nfl"]
Games = db["games"]

VERSION = 1
RE_PULL = True

def train(inData):
  imputer = SimpleImputer(strategy='constant', fill_value=-1)
  data = pd.DataFrame(inData)
  x = data [[
    'gameId','gameType','season','week','date','venue','homeTeam','awayTeam','lineJudge','referee','videoOperator','replayAssistant','umpire','headLinesman','backJudge','sideJudge','fieldJudge'
  ]].values
  # y = data [['homeScore','awayScore','winner','totalScore','scoreDifferential']].values
  y_winner = data [['winner']].values.ravel()
  y_homeScore = data [['homeScore']].values.ravel()
  y_awayScore = data [['awayScore']].values.ravel()
  y_totalScore = data [['totalScore']].values.ravel()
  y_scoreDifferential = data [['scoreDifferential']].values.ravel()

  imputer.fit(x)
  x = imputer.transform(x)

  x_train, x_test, y_train_winner, y_test_winner = train_test_split(x, y_winner, test_size=0.2, random_state=12)
  x_train, x_test, y_train_homeScore, y_test_homeScore = train_test_split(x, y_homeScore, test_size=0.2, random_state=12)
  x_train, x_test, y_train_awayScore, y_test_awayScore = train_test_split(x, y_awayScore, test_size=0.2, random_state=12)
  x_train, x_test, y_train_totalScore, y_test_totalScore = train_test_split(x, y_totalScore, test_size=0.2, random_state=12)
  x_train, x_test, y_train_scoreDifferential, y_test_scoreDifferential = train_test_split(x, y_scoreDifferential, test_size=0.2, random_state=12)

  # clf = RandomForestClassifier(random_state=12)
  clf_winner = RandomForestClassifier(random_state=12)
  clf_homeScore = RandomForestClassifier(random_state=12)
  clf_awayScore = RandomForestClassifier(random_state=12)
  clf_totalScore = RandomForestClassifier(random_state=12)
  clf_scoreDifferential = RandomForestClassifier(random_state=12)

  # clf.fit(x,y)
  clf_winner.fit(x_train,y_train_winner)
  clf_homeScore.fit(x_train,y_train_homeScore)
  clf_awayScore.fit(x_train,y_train_awayScore)
  clf_totalScore.fit(x_train,y_train_totalScore)
  clf_scoreDifferential.fit(x_train,y_train_scoreDifferential)
  predictions_winner = clf_winner.predict(x_test)
  predictions_homeScore = clf_homeScore.predict(x_test)
  predictions_awayScore = clf_awayScore.predict(x_test)
  predictions_totalScore = clf_totalScore.predict(x_test)
  predictions_scoreDifferential = clf_scoreDifferential.predict(x_test)
  winner_accuracy = accuracy_score(y_test_winner, predictions_winner)
  homeScore_accuracy = accuracy_score(y_test_homeScore, predictions_homeScore)
  awayScore_accuracy = accuracy_score(y_test_awayScore, predictions_awayScore)
  totalScore_accuracy = accuracy_score(y_test_totalScore, predictions_totalScore)
  scoreDifferential_accuracy = accuracy_score(y_test_scoreDifferential, predictions_scoreDifferential)
  print("Winner Accuracy:", winner_accuracy)
  print("Home Score Accuracy:", homeScore_accuracy)
  print("Away Score Accuracy:", awayScore_accuracy)
  print("Total Score Accuracy:", totalScore_accuracy)
  print("Score Differential Accuracy:", scoreDifferential_accuracy)

  TrainingRecords = db['training_records']
  Metadata = db['metadata']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'savedAt': timestamp,
    'lastTrainedId': inData[len(inData)]['id'],
    'accuracies': {
      'winner': winner_accuracy,
      'homeScore': homeScore_accuracy,
      'awayScore': awayScore_accuracy,
      'totalScore': totalScore_accuracy,
      'scoreDifferential': scoreDifferential_accuracy,
    }
  })

  # dump(clf, f'models/nfl_ai_v{VERSION}.joblib')
  dump(clf_winner, f'models/nfl_ai_v{VERSION}_winner.joblib')
  dump(clf_homeScore, f'models/nfl_ai_v{VERSION}_homeScore.joblib')
  dump(clf_awayScore, f'models/nfl_ai_v{VERSION}_awayScore.joblib')
  dump(clf_totalScore, f'models/nfl_ai_v{VERSION}_totalScore.joblib')
  dump(clf_scoreDifferential, f'models/nfl_ai_v{VERSION}_scoreDifferential.joblib')



if __name__ == '__main__':
  if RE_PULL:
    # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
    # client = MongoClient(db_url)
    client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
    db = client["nfl"]
    games = list(db["games"].find(
      {}
    ))
    
    pool = Pool(processes=4)
    result = pool.map(master_inputs,games)
    pool.close()
    dump(result,f'training_data/nfl/training_data_v{VERSION}.joblib')
    f = open('training_data/training_data_text.txt', 'w')
    f.write(json.dumps(result[len(result)-200:len(result)]))
  else:
    training_data_path = f'training_data/nfl/training_data_v{VERSION}.joblib'
    print(training_data_path)
    result = load(training_data_path)
    f = open('training_data/nfl/training_data_text.txt', 'w')
    f.write(json.dumps(result[len(result)-200:len(result)]))
  print('Games Collected')
  train(result)