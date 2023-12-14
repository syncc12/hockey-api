import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

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
from util.training_data import season_training_data, game_training_data

RE_PULL = False

VERSION = 3

def xPlayerData(homeAway,position,index,isGoalie=False,gamesBack=5):
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

def xTeamData(homeAway,gamesBack=5):
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
def train(inData):
  imputer = SimpleImputer(strategy='constant', fill_value=-1)
  data = pd.DataFrame(inData)
  # x = data [['id','season','gameType','venue','neutralSite','homeTeam','awayTeam','awaySplitSquad','homeSplitSquad','startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1','awayForward1','awayForward1Position','awayForward1Age','awayForward1Shoots','awayForward2','awayForward2Position','awayForward2Age','awayForward2Shoots','awayForward3','awayForward3Position','awayForward3Age','awayForward3Shoots','awayForward4','awayForward4Position','awayForward4Age','awayForward4Shoots','awayForward5','awayForward5Position','awayForward5Age','awayForward5Shoots','awayForward6','awayForward6Position','awayForward6Age','awayForward6Shoots','awayForward7','awayForward7Position','awayForward7Age','awayForward7Shoots','awayForward8','awayForward8Position','awayForward8Age','awayForward8Shoots','awayForward9','awayForward9Position','awayForward9Age','awayForward9Shoots','awayForward10','awayForward10Position','awayForward10Age','awayForward10Shoots','awayForward11','awayForward11Position','awayForward11Age','awayForward11Shoots','awayForward12','awayForward12Position','awayForward12Age','awayForward12Shoots','awayForward13','awayForward13Position','awayForward13Age','awayForward13Shoots','awayDefenseman1','awayDefenseman1Position','awayDefenseman1Age','awayDefenseman1Shoots','awayDefenseman2','awayDefenseman2Position','awayDefenseman2Age','awayDefenseman2Shoots','awayDefenseman3','awayDefenseman3Position','awayDefenseman3Age','awayDefenseman3Shoots','awayDefenseman4','awayDefenseman4Position','awayDefenseman4Age','awayDefenseman4Shoots','awayDefenseman5','awayDefenseman5Position','awayDefenseman5Age','awayDefenseman5Shoots','awayDefenseman6','awayDefenseman6Position','awayDefenseman6Age','awayDefenseman6Shoots','awayDefenseman7','awayDefenseman7Position','awayDefenseman7Age','awayDefenseman7Shoots','awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieAge','awayStartingGoalieHeight','awayStartingGoalieWeight','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieAge','awayBackupGoalieHeight','awayBackupGoalieWeight','awayThirdGoalie','awayThirdGoalieCatches','awayThirdGoalieAge','awayThirdGoalieHeight','awayThirdGoalieWeight','homeForward1','homeForward1Position','homeForward1Age','homeForward1Shoots','homeForward2','homeForward2Position','homeForward2Age','homeForward2Shoots','homeForward3','homeForward3Position','homeForward3Age','homeForward3Shoots','homeForward4','homeForward4Position','homeForward4Age','homeForward4Shoots','homeForward5','homeForward5Position','homeForward5Age','homeForward5Shoots','homeForward6','homeForward6Position','homeForward6Age','homeForward6Shoots','homeForward7','homeForward7Position','homeForward7Age','homeForward7Shoots','homeForward8','homeForward8Position','homeForward8Age','homeForward8Shoots','homeForward9','homeForward9Position','homeForward9Age','homeForward9Shoots','homeForward10','homeForward10Position','homeForward10Age','homeForward10Shoots','homeForward11','homeForward11Position','homeForward11Age','homeForward11Shoots','homeForward12','homeForward12Position','homeForward12Age','homeForward12Shoots','homeForward13','homeForward13Position','homeForward13Age','homeForward13Shoots','homeDefenseman1','homeDefenseman1Position','homeDefenseman1Age','homeDefenseman1Shoots','homeDefenseman2','homeDefenseman2Position','homeDefenseman2Age','homeDefenseman2Shoots','homeDefenseman3','homeDefenseman3Position','homeDefenseman3Age','homeDefenseman3Shoots','homeDefenseman4','homeDefenseman4Position','homeDefenseman4Age','homeDefenseman4Shoots','homeDefenseman5','homeDefenseman5Position','homeDefenseman5Age','homeDefenseman5Shoots','homeDefenseman6','homeDefenseman6Position','homeDefenseman6Age','homeDefenseman6Shoots','homeDefenseman7','homeDefenseman7Position','homeDefenseman7Age','homeDefenseman7Shoots','homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieAge','homeStartingGoalieHeight','homeStartingGoalieWeight','homeBackupGoalie','homeBackupGoalieCatches','homeBackupGoalieAge','homeBackupGoalieHeight','homeBackupGoalieWeight','homeThirdGoalie','homeThirdGoalieCatches','homeThirdGoalieAge','homeThirdGoalieHeight','homeThirdGoalieWeight']].values
  # x = data [['id','season','gameType','venue','neutralSite','homeTeam','awayTeam','startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1','awayForward1','awayForward1Position','awayForward1Age','awayForward1Shoots','awayForward2','awayForward2Position','awayForward2Age','awayForward2Shoots','awayForward3','awayForward3Position','awayForward3Age','awayForward3Shoots','awayForward4','awayForward4Position','awayForward4Age','awayForward4Shoots','awayForward5','awayForward5Position','awayForward5Age','awayForward5Shoots','awayForward6','awayForward6Position','awayForward6Age','awayForward6Shoots','awayForward7','awayForward7Position','awayForward7Age','awayForward7Shoots','awayForward8','awayForward8Position','awayForward8Age','awayForward8Shoots','awayForward9','awayForward9Position','awayForward9Age','awayForward9Shoots','awayForward10','awayForward10Position','awayForward10Age','awayForward10Shoots','awayForward11','awayForward11Position','awayForward11Age','awayForward11Shoots','awayForward12','awayForward12Position','awayForward12Age','awayForward12Shoots','awayForward13','awayForward13Position','awayForward13Age','awayForward13Shoots','awayDefenseman1','awayDefenseman1Position','awayDefenseman1Age','awayDefenseman1Shoots','awayDefenseman2','awayDefenseman2Position','awayDefenseman2Age','awayDefenseman2Shoots','awayDefenseman3','awayDefenseman3Position','awayDefenseman3Age','awayDefenseman3Shoots','awayDefenseman4','awayDefenseman4Position','awayDefenseman4Age','awayDefenseman4Shoots','awayDefenseman5','awayDefenseman5Position','awayDefenseman5Age','awayDefenseman5Shoots','awayDefenseman6','awayDefenseman6Position','awayDefenseman6Age','awayDefenseman6Shoots','awayDefenseman7','awayDefenseman7Position','awayDefenseman7Age','awayDefenseman7Shoots','awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieAge','awayStartingGoalieHeight','awayStartingGoalieWeight','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieAge','awayBackupGoalieHeight','awayBackupGoalieWeight','awayThirdGoalie','awayThirdGoalieCatches','awayThirdGoalieAge','awayThirdGoalieHeight','awayThirdGoalieWeight','homeForward1','homeForward1Position','homeForward1Age','homeForward1Shoots','homeForward2','homeForward2Position','homeForward2Age','homeForward2Shoots','homeForward3','homeForward3Position','homeForward3Age','homeForward3Shoots','homeForward4','homeForward4Position','homeForward4Age','homeForward4Shoots','homeForward5','homeForward5Position','homeForward5Age','homeForward5Shoots','homeForward6','homeForward6Position','homeForward6Age','homeForward6Shoots','homeForward7','homeForward7Position','homeForward7Age','homeForward7Shoots','homeForward8','homeForward8Position','homeForward8Age','homeForward8Shoots','homeForward9','homeForward9Position','homeForward9Age','homeForward9Shoots','homeForward10','homeForward10Position','homeForward10Age','homeForward10Shoots','homeForward11','homeForward11Position','homeForward11Age','homeForward11Shoots','homeForward12','homeForward12Position','homeForward12Age','homeForward12Shoots','homeForward13','homeForward13Position','homeForward13Age','homeForward13Shoots','homeDefenseman1','homeDefenseman1Position','homeDefenseman1Age','homeDefenseman1Shoots','homeDefenseman2','homeDefenseman2Position','homeDefenseman2Age','homeDefenseman2Shoots','homeDefenseman3','homeDefenseman3Position','homeDefenseman3Age','homeDefenseman3Shoots','homeDefenseman4','homeDefenseman4Position','homeDefenseman4Age','homeDefenseman4Shoots','homeDefenseman5','homeDefenseman5Position','homeDefenseman5Age','homeDefenseman5Shoots','homeDefenseman6','homeDefenseman6Position','homeDefenseman6Age','homeDefenseman6Shoots','homeDefenseman7','homeDefenseman7Position','homeDefenseman7Age','homeDefenseman7Shoots','homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieAge','homeStartingGoalieHeight','homeStartingGoalieWeight','homeBackupGoalie','homeBackupGoalieCatches','homeBackupGoalieAge','homeBackupGoalieHeight','homeBackupGoalieWeight','homeThirdGoalie','homeThirdGoalieCatches','homeThirdGoalieAge','homeThirdGoalieHeight','homeThirdGoalieWeight']].values
  x = data [
    ['id','season','gameType','venue','neutralSite'] +
    xTeamData('home') +
    xTeamData('away') +
    ['startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1'] +
    xPlayerData('away','Forward',1) +
    xPlayerData('away','Forward',2) +
    xPlayerData('away','Forward',3) +
    xPlayerData('away','Forward',4) +
    xPlayerData('away','Forward',5) +
    xPlayerData('away','Forward',6) +
    xPlayerData('away','Forward',7) +
    xPlayerData('away','Forward',8) +
    xPlayerData('away','Forward',9) +
    xPlayerData('away','Forward',10) +
    xPlayerData('away','Forward',11) +
    xPlayerData('away','Forward',12) +
    xPlayerData('away','Forward',13) +
    xPlayerData('away','Defenseman',1) +
    xPlayerData('away','Defenseman',2) +
    xPlayerData('away','Defenseman',3) +
    xPlayerData('away','Defenseman',4) +
    xPlayerData('away','Defenseman',5) +
    xPlayerData('away','Defenseman',6) +
    xPlayerData('away','Defenseman',7) +
    xPlayerData('away','Goalie','Starting',True) +
    xPlayerData('away','Goalie','Backup',True) +
    xPlayerData('home','Forward',1) +
    xPlayerData('home','Forward',2) +
    xPlayerData('home','Forward',3) +
    xPlayerData('home','Forward',4) +
    xPlayerData('home','Forward',5) +
    xPlayerData('home','Forward',6) +
    xPlayerData('home','Forward',7) +
    xPlayerData('home','Forward',8) +
    xPlayerData('home','Forward',9) +
    xPlayerData('home','Forward',10) +
    xPlayerData('home','Forward',11) +
    xPlayerData('home','Forward',12) +
    xPlayerData('home','Forward',13) +
    xPlayerData('home','Defenseman',1) +
    xPlayerData('home','Defenseman',2) +
    xPlayerData('home','Defenseman',3) +
    xPlayerData('home','Defenseman',4) +
    xPlayerData('home','Defenseman',5) +
    xPlayerData('home','Defenseman',6) +
    xPlayerData('home','Defenseman',7) +
    xPlayerData('home','Goalie','Starting',True) +
    xPlayerData('home','Goalie','Backup',True)
  ].values
  y = data [['homeScore','awayScore','winner']].values

  imputer.fit(x)
  x = imputer.transform(x)

  clf = RandomForestClassifier(random_state=12)

  clf.fit(x,y)

  dump(clf, f'models/nhl_ai_v{VERSION}.joblib')

USE_SEASONS = True
SKIP_SEASONS = [
  '19171918','19181919','19191920','19201921','19211922','19221923','19231924','19241925','19251926','19261927','19271928','19281929','19291930','19301931','19311932',
  '19321933','19331934','19341935','19351936','19361937','19371938','19381939','19391940','19401941','19411942','19421943','19431944','19441945','19451946','19461947',
  '19471948','19481949','19491950','19501951','19511952','19521953','19531954','19541955','19551956','19561957','19571958','19581959','19591960','19601961','19611962',
  '19621963','19631964','19641965','19651966','19661967','19671968','19681969','19691970','19701971','19711972','19721973','19731974','19741975','19751976','19761977',
  '19771978','19781979','19791980','19801981','19811982','19821983','19831984','19841985','19851986','19861987','19871988','19881989','19891990','19901991','19911992',
  '19921993','19931994','19941995','19951996','19961997','19971998','19981999','19992000','20002001','20012002','20022003','20032004','20052006','20062007','20072008',
  '20082009','20092010','20102011','20112012','20122013','20132014','20142015','20152016','20162017','20172018','20182019','20192020','20202021','20212022','20232024',
]


if __name__ == '__main__':
  if RE_PULL:
    # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
    # client = MongoClient(db_url)
    client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
    db = client["hockey"]
    if USE_SEASONS:
      seasons = list(db["dev_seasons"].find(
        {},
        {'_id':0,'seasonId': 1}
      ))
      seasons = [season['seasonId'] for season in seasons]
      if (len(SKIP_SEASONS) > 0):
        for season in SKIP_SEASONS:
          seasons.remove(season)
        print(seasons)
    else:
      startID = 1924030112
      endID = 1924030114
      games = db["dev_games"].find(
        {'id':{'$gte':startID,'$lt':endID+1}},
        # {'id':{'$lt':endID+1}},
        {'id': 1, '_id': 0}
      )

    pool = Pool(processes=4)
    if USE_SEASONS:
      result = pool.map(season_training_data,seasons)
    else:
      result = pool.map(game_training_data,games)
    result = np.concatenate(result).tolist()
    pool.close()
    if len(SKIP_SEASONS) > 0:
      for skip_season in SKIP_SEASONS:
        season_data = load(f'training_data/training_data_v{VERSION}_{skip_season}.joblib')
        for line in season_data:
          result.append(line)
    dump(result,f'training_data/training_data_v{VERSION}.joblib')
    # f = open('training_data/training_data_text.txt', 'w')
    # f.write(json.dumps(result[60000:60500]))
  else:
    training_data_path = f'training_data/training_data_v{VERSION}.joblib'
    print(training_data_path)
    result = load(training_data_path)
    # f = open('training_data/training_data_text.txt', 'w')
    # f.write(json.dumps(result[60000:60500]))
  print('Games Collected')
  train(result)