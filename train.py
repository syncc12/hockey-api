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
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util.helpers import latestIDs
import time
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON

RE_PULL = True
UPDATE = False
# VERSION = 6

def xPlayerData(homeAway,position,index,isGoalie=False,gamesBack=-1):
  if isGoalie:
    playerTitle = f'{homeAway}{index}{position}'
    playerKeys = [
      f'{playerTitle}',
      f'{playerTitle}Catches',
      f'{playerTitle}Age',
      f'{playerTitle}Height',
      f'{playerTitle}Weight',
    ]
  else:
    playerTitle = f'{homeAway}{position}{index}'
    playerKeys = [
      f'{playerTitle}',
      f'{playerTitle}Position',
      f'{playerTitle}Age',
      f'{playerTitle}Shoots',
    ]
  
  if gamesBack > -1:
    for i in range(0,gamesBack):
      playerKeys.append(f'{playerTitle}Back{i+1}GameId')
      playerKeys.append(f'{playerTitle}Back{i+1}GameDate')
      playerKeys.append(f'{playerTitle}Back{i+1}GameTeam')
      playerKeys.append(f'{playerTitle}Back{i+1}GameHomeAway')
      playerKeys.append(f'{playerTitle}Back{i+1}GamePlayer')
      playerKeys.append(f'{playerTitle}Back{i+1}GamePosition')
      playerKeys.append(f'{playerTitle}Back{i+1}GamePIM')
      playerKeys.append(f'{playerTitle}Back{i+1}GameTOI')
      if isGoalie:
        playerKeys.append(f'{playerTitle}Back{i+1}GameEvenStrengthShotsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePowerPlayShotsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShorthandedShotsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GameSaves')
        playerKeys.append(f'{playerTitle}Back{i+1}GameSavePercentage')
        playerKeys.append(f'{playerTitle}Back{i+1}GameEvenStrengthGoalsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePowerPlayGoalsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShorthandedGoalsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GameGoalsAgainst')
      else:
        playerKeys.append(f'{playerTitle}Back{i+1}GameGoals')
        playerKeys.append(f'{playerTitle}Back{i+1}GameAssists')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePoints')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePlusMinus')
        playerKeys.append(f'{playerTitle}Back{i+1}GameHits')
        playerKeys.append(f'{playerTitle}Back{i+1}GameBlockedShots')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePowerPlayGoals')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePowerPlayPoints')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShorthandedGoals')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShorthandedPoints')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShots')
        playerKeys.append(f'{playerTitle}Back{i+1}GameFaceoffs')
        playerKeys.append(f'{playerTitle}Back{i+1}GameFaceoffWinPercentage')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePowerPlayTOI')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShorthandedTOI')
  return playerKeys

def xTeamData(homeAway,gamesBack=-1):
  teamKeys = [f'{homeAway}Team']
  for i in range(0,gamesBack):
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameId')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameDate')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameType')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameVenue')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameStartTime')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameEasternOffset')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameVenueOffset')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOutcome')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameHomeAway')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameFinalPeriod')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameScore')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameShots')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameFaceoffWinPercentage')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GamePowerPlays')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GamePowerPlayPercentage')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GamePIM')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameHits')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameBlocks')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponent')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentScore')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentShots')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentFaceoffWinPercentage')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentPowerPlays')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentPowerPlayPercentage')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentPIM')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentHits')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentBlocks')
  return teamKeys

def train(db, inData):
  imputer = SimpleImputer(strategy='constant', fill_value=-1)
  data = pd.DataFrame(inData)
  # x = data [[
  #   'id','season','gameType','venue','neutralSite','homeTeam','awayTeam','startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1',
  #   'awayForward1','awayForward1Age','awayForward1GamesPlayed','awayForward2','awayForward2Age','awayForward2GamesPlayed','awayForward3','awayForward3Age','awayForward3GamesPlayed','awayForward4','awayForward4Age','awayForward4GamesPlayed','awayForward5','awayForward5Age','awayForward5GamesPlayed',
  #   'awayForward6','awayForward6Age','awayForward6GamesPlayed','awayForward7','awayForward7Age','awayForward7GamesPlayed','awayForward8','awayForward8Age','awayForward8GamesPlayed','awayForward9','awayForward9Age','awayForward9GamesPlayed','awayForward10','awayForward10Age','awayForward10GamesPlayed',
  #   'awayForward11','awayForward11Age','awayForward11GamesPlayed','awayForward12','awayForward12Age','awayForward12GamesPlayed','awayForward13','awayForward13Age','awayForward13GamesPlayed',
  #   'awayDefenseman1','awayDefenseman1Age','awayDefenseman1GamesPlayed','awayDefenseman2','awayDefenseman2Age','awayDefenseman2GamesPlayed','awayDefenseman3','awayDefenseman3Age','awayDefenseman3GamesPlayed','awayDefenseman4','awayDefenseman4Age','awayDefenseman4GamesPlayed','awayDefenseman5',
  #   'awayDefenseman5Age','awayDefenseman5GamesPlayed','awayDefenseman6','awayDefenseman6Age','awayDefenseman6GamesPlayed','awayDefenseman7','awayDefenseman7Age','awayDefenseman7GamesPlayed',
  #   'awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieAge','awayStartingGoalieGamesPlayed','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieAge','awayBackupGoalieGamesPlayed',
  #   'homeForward1','homeForward1Age','homeForward1GamesPlayed','homeForward2','homeForward2Age','homeForward2GamesPlayed','homeForward3','homeForward3Age','homeForward3GamesPlayed','homeForward4','homeForward4Age','homeForward4GamesPlayed','homeForward5','homeForward5Age','homeForward5GamesPlayed',
  #   'homeForward6','homeForward6Age','homeForward6GamesPlayed','homeForward7','homeForward7Age','homeForward7GamesPlayed','homeForward8','homeForward8Age','homeForward8GamesPlayed','homeForward9','homeForward9Age','homeForward9GamesPlayed','homeForward10','homeForward10Age','homeForward10GamesPlayed',
  #   'homeForward11','homeForward11Age','homeForward11GamesPlayed','homeForward12','homeForward12Age','homeForward12GamesPlayed','homeForward13','homeForward13Age','homeForward13GamesPlayed',
  #   'homeDefenseman1','homeDefenseman1Age','homeDefenseman1GamesPlayed','homeDefenseman2','homeDefenseman2Age','homeDefenseman2GamesPlayed','homeDefenseman3','homeDefenseman3Age','homeDefenseman3GamesPlayed','homeDefenseman4','homeDefenseman4Age','homeDefenseman4GamesPlayed','homeDefenseman5',
  #   'homeDefenseman5Age','homeDefenseman5GamesPlayed','homeDefenseman6','homeDefenseman6Age','homeDefenseman6GamesPlayed','homeDefenseman7','homeDefenseman7Age','homeDefenseman7GamesPlayed',
  #   'homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieAge','homeStartingGoalieGamesPlayed','homeBackupGoalie','homeBackupGoalieCatches','homeBackupGoalieAge','homeBackupGoalieGamesPlayed'
  # ]].values
  x = data [X_INPUTS]
  # y = data [['homeScore','awayScore','winner','totalGoals','goalDifferential']].values
  y_winner = data [['winner']].values.ravel()
  y_homeScore = data [['homeScore']].values.ravel()
  y_awayScore = data [['awayScore']].values.ravel()
  y_totalGoals = data [['totalGoals']].values.ravel()
  y_goalDifferential = data [['goalDifferential']].values.ravel()

  imputer.fit(x)
  x = imputer.transform(x)

  x_train, x_test, y_train_winner, y_test_winner = train_test_split(x, y_winner, test_size=0.2, random_state=RANDOM_STATE)
  x_train, x_test, y_train_homeScore, y_test_homeScore = train_test_split(x, y_homeScore, test_size=0.2, random_state=RANDOM_STATE)
  x_train, x_test, y_train_awayScore, y_test_awayScore = train_test_split(x, y_awayScore, test_size=0.2, random_state=RANDOM_STATE)
  x_train, x_test, y_train_totalGoals, y_test_totalGoals = train_test_split(x, y_totalGoals, test_size=0.2, random_state=RANDOM_STATE)
  x_train, x_test, y_train_goalDifferential, y_test_goalDifferential = train_test_split(x, y_goalDifferential, test_size=0.2, random_state=RANDOM_STATE)

  # clf = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_winner = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_homeScore = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_awayScore = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_totalGoals = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_goalDifferential = RandomForestClassifier(random_state=RANDOM_STATE)

  # clf.fit(x,y)
  clf_winner.fit(x_train,y_train_winner)
  clf_homeScore.fit(x_train,y_train_homeScore)
  clf_awayScore.fit(x_train,y_train_awayScore)
  clf_totalGoals.fit(x_train,y_train_totalGoals)
  clf_goalDifferential.fit(x_train,y_train_goalDifferential)
  predictions_winner = clf_winner.predict(x_test)
  predictions_homeScore = clf_homeScore.predict(x_test)
  predictions_awayScore = clf_awayScore.predict(x_test)
  predictions_totalGoals = clf_totalGoals.predict(x_test)
  predictions_goalDifferential = clf_goalDifferential.predict(x_test)
  winner_accuracy = accuracy_score(y_test_winner, predictions_winner)
  homeScore_accuracy = accuracy_score(y_test_homeScore, predictions_homeScore)
  awayScore_accuracy = accuracy_score(y_test_awayScore, predictions_awayScore)
  totalGoals_accuracy = accuracy_score(y_test_totalGoals, predictions_totalGoals)
  goalDifferential_accuracy = accuracy_score(y_test_goalDifferential, predictions_goalDifferential)
  print("Winner Accuracy:", winner_accuracy)
  print("Home Score Accuracy:", homeScore_accuracy)
  print("Away Score Accuracy:", awayScore_accuracy)
  print("Total Goals Accuracy:", totalGoals_accuracy)
  print("Goal Differential Accuracy:", goalDifferential_accuracy)
  
  validation_winner = clf_winner.predict(x_train)
  validation_homeScore = clf_homeScore.predict(x_train)
  validation_awayScore = clf_awayScore.predict(x_train)
  validation_totalGoals = clf_totalGoals.predict(x_train)
  validation_goalDifferential = clf_goalDifferential.predict(x_train)
  winner_accuracy_validation = accuracy_score(y_train_winner, validation_winner)
  homeScore_accuracy_validation = accuracy_score(y_train_homeScore, validation_homeScore)
  awayScore_accuracy_validation = accuracy_score(y_train_awayScore, validation_awayScore)
  totalGoals_accuracy_validation = accuracy_score(y_train_totalGoals, validation_totalGoals)
  goalDifferential_accuracy_validation = accuracy_score(y_train_goalDifferential, validation_goalDifferential)
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
  print("Winner Validation:", winner_accuracy_validation)
  print("Home Score Validation:", homeScore_accuracy_validation)
  print("Away Score Validation:", awayScore_accuracy_validation)
  print("Total Goals Validation:", totalGoals_accuracy_validation)
  print("Goal Differential Validation:", goalDifferential_accuracy_validation)

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
    'accuracies': {
      'winner': winner_accuracy,
      'homeScore': homeScore_accuracy,
      'awayScore': awayScore_accuracy,
      'totalGoals': totalGoals_accuracy,
      'goalDifferential': goalDifferential_accuracy,
    },
    'validation': {
      'winner': winner_accuracy_validation,
      'homeScore': homeScore_accuracy_validation,
      'awayScore': awayScore_accuracy_validation,
      'totalGoals': totalGoals_accuracy_validation,
      'goalDifferential': goalDifferential_accuracy_validation,
    },
  })

  # dump(clf, f'models/nhl_ai_v{FILE_VERSION}.joblib')
  dump(clf_winner, f'models/nhl_ai_v{FILE_VERSION}_winner.joblib')
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
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]
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