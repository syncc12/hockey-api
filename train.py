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

RE_PULL = True

def train(inData):
  imputer = SimpleImputer(strategy='constant', fill_value=-1)
  data = pd.DataFrame(inData)
  # x = data [['id','season','gameType','venue','neutralSite','homeTeam','awayTeam','awaySplitSquad','homeSplitSquad','startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1','awayForward1','awayForward1Position','awayForward1Age','awayForward1Shoots','awayForward2','awayForward2Position','awayForward2Age','awayForward2Shoots','awayForward3','awayForward3Position','awayForward3Age','awayForward3Shoots','awayForward4','awayForward4Position','awayForward4Age','awayForward4Shoots','awayForward5','awayForward5Position','awayForward5Age','awayForward5Shoots','awayForward6','awayForward6Position','awayForward6Age','awayForward6Shoots','awayForward7','awayForward7Position','awayForward7Age','awayForward7Shoots','awayForward8','awayForward8Position','awayForward8Age','awayForward8Shoots','awayForward9','awayForward9Position','awayForward9Age','awayForward9Shoots','awayForward10','awayForward10Position','awayForward10Age','awayForward10Shoots','awayForward11','awayForward11Position','awayForward11Age','awayForward11Shoots','awayForward12','awayForward12Position','awayForward12Age','awayForward12Shoots','awayForward13','awayForward13Position','awayForward13Age','awayForward13Shoots','awayDefenseman1','awayDefenseman1Position','awayDefenseman1Age','awayDefenseman1Shoots','awayDefenseman2','awayDefenseman2Position','awayDefenseman2Age','awayDefenseman2Shoots','awayDefenseman3','awayDefenseman3Position','awayDefenseman3Age','awayDefenseman3Shoots','awayDefenseman4','awayDefenseman4Position','awayDefenseman4Age','awayDefenseman4Shoots','awayDefenseman5','awayDefenseman5Position','awayDefenseman5Age','awayDefenseman5Shoots','awayDefenseman6','awayDefenseman6Position','awayDefenseman6Age','awayDefenseman6Shoots','awayDefenseman7','awayDefenseman7Position','awayDefenseman7Age','awayDefenseman7Shoots','awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieAge','awayStartingGoalieHeight','awayStartingGoalieWeight','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieAge','awayBackupGoalieHeight','awayBackupGoalieWeight','awayThirdGoalie','awayThirdGoalieCatches','awayThirdGoalieAge','awayThirdGoalieHeight','awayThirdGoalieWeight','homeForward1','homeForward1Position','homeForward1Age','homeForward1Shoots','homeForward2','homeForward2Position','homeForward2Age','homeForward2Shoots','homeForward3','homeForward3Position','homeForward3Age','homeForward3Shoots','homeForward4','homeForward4Position','homeForward4Age','homeForward4Shoots','homeForward5','homeForward5Position','homeForward5Age','homeForward5Shoots','homeForward6','homeForward6Position','homeForward6Age','homeForward6Shoots','homeForward7','homeForward7Position','homeForward7Age','homeForward7Shoots','homeForward8','homeForward8Position','homeForward8Age','homeForward8Shoots','homeForward9','homeForward9Position','homeForward9Age','homeForward9Shoots','homeForward10','homeForward10Position','homeForward10Age','homeForward10Shoots','homeForward11','homeForward11Position','homeForward11Age','homeForward11Shoots','homeForward12','homeForward12Position','homeForward12Age','homeForward12Shoots','homeForward13','homeForward13Position','homeForward13Age','homeForward13Shoots','homeDefenseman1','homeDefenseman1Position','homeDefenseman1Age','homeDefenseman1Shoots','homeDefenseman2','homeDefenseman2Position','homeDefenseman2Age','homeDefenseman2Shoots','homeDefenseman3','homeDefenseman3Position','homeDefenseman3Age','homeDefenseman3Shoots','homeDefenseman4','homeDefenseman4Position','homeDefenseman4Age','homeDefenseman4Shoots','homeDefenseman5','homeDefenseman5Position','homeDefenseman5Age','homeDefenseman5Shoots','homeDefenseman6','homeDefenseman6Position','homeDefenseman6Age','homeDefenseman6Shoots','homeDefenseman7','homeDefenseman7Position','homeDefenseman7Age','homeDefenseman7Shoots','homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieAge','homeStartingGoalieHeight','homeStartingGoalieWeight','homeBackupGoalie','homeBackupGoalieCatches','homeBackupGoalieAge','homeBackupGoalieHeight','homeBackupGoalieWeight','homeThirdGoalie','homeThirdGoalieCatches','homeThirdGoalieAge','homeThirdGoalieHeight','homeThirdGoalieWeight']].values
  x = data [['id','season','gameType','venue','neutralSite','homeTeam','awayTeam','startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1','awayForward1','awayForward1Position','awayForward1Age','awayForward1Shoots','awayForward2','awayForward2Position','awayForward2Age','awayForward2Shoots','awayForward3','awayForward3Position','awayForward3Age','awayForward3Shoots','awayForward4','awayForward4Position','awayForward4Age','awayForward4Shoots','awayForward5','awayForward5Position','awayForward5Age','awayForward5Shoots','awayForward6','awayForward6Position','awayForward6Age','awayForward6Shoots','awayForward7','awayForward7Position','awayForward7Age','awayForward7Shoots','awayForward8','awayForward8Position','awayForward8Age','awayForward8Shoots','awayForward9','awayForward9Position','awayForward9Age','awayForward9Shoots','awayForward10','awayForward10Position','awayForward10Age','awayForward10Shoots','awayForward11','awayForward11Position','awayForward11Age','awayForward11Shoots','awayForward12','awayForward12Position','awayForward12Age','awayForward12Shoots','awayForward13','awayForward13Position','awayForward13Age','awayForward13Shoots','awayDefenseman1','awayDefenseman1Position','awayDefenseman1Age','awayDefenseman1Shoots','awayDefenseman2','awayDefenseman2Position','awayDefenseman2Age','awayDefenseman2Shoots','awayDefenseman3','awayDefenseman3Position','awayDefenseman3Age','awayDefenseman3Shoots','awayDefenseman4','awayDefenseman4Position','awayDefenseman4Age','awayDefenseman4Shoots','awayDefenseman5','awayDefenseman5Position','awayDefenseman5Age','awayDefenseman5Shoots','awayDefenseman6','awayDefenseman6Position','awayDefenseman6Age','awayDefenseman6Shoots','awayDefenseman7','awayDefenseman7Position','awayDefenseman7Age','awayDefenseman7Shoots','awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieAge','awayStartingGoalieHeight','awayStartingGoalieWeight','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieAge','awayBackupGoalieHeight','awayBackupGoalieWeight','awayThirdGoalie','awayThirdGoalieCatches','awayThirdGoalieAge','awayThirdGoalieHeight','awayThirdGoalieWeight','homeForward1','homeForward1Position','homeForward1Age','homeForward1Shoots','homeForward2','homeForward2Position','homeForward2Age','homeForward2Shoots','homeForward3','homeForward3Position','homeForward3Age','homeForward3Shoots','homeForward4','homeForward4Position','homeForward4Age','homeForward4Shoots','homeForward5','homeForward5Position','homeForward5Age','homeForward5Shoots','homeForward6','homeForward6Position','homeForward6Age','homeForward6Shoots','homeForward7','homeForward7Position','homeForward7Age','homeForward7Shoots','homeForward8','homeForward8Position','homeForward8Age','homeForward8Shoots','homeForward9','homeForward9Position','homeForward9Age','homeForward9Shoots','homeForward10','homeForward10Position','homeForward10Age','homeForward10Shoots','homeForward11','homeForward11Position','homeForward11Age','homeForward11Shoots','homeForward12','homeForward12Position','homeForward12Age','homeForward12Shoots','homeForward13','homeForward13Position','homeForward13Age','homeForward13Shoots','homeDefenseman1','homeDefenseman1Position','homeDefenseman1Age','homeDefenseman1Shoots','homeDefenseman2','homeDefenseman2Position','homeDefenseman2Age','homeDefenseman2Shoots','homeDefenseman3','homeDefenseman3Position','homeDefenseman3Age','homeDefenseman3Shoots','homeDefenseman4','homeDefenseman4Position','homeDefenseman4Age','homeDefenseman4Shoots','homeDefenseman5','homeDefenseman5Position','homeDefenseman5Age','homeDefenseman5Shoots','homeDefenseman6','homeDefenseman6Position','homeDefenseman6Age','homeDefenseman6Shoots','homeDefenseman7','homeDefenseman7Position','homeDefenseman7Age','homeDefenseman7Shoots','homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieAge','homeStartingGoalieHeight','homeStartingGoalieWeight','homeBackupGoalie','homeBackupGoalieCatches','homeBackupGoalieAge','homeBackupGoalieHeight','homeBackupGoalieWeight','homeThirdGoalie','homeThirdGoalieCatches','homeThirdGoalieAge','homeThirdGoalieHeight','homeThirdGoalieWeight']].values
  y = data [['homeScore','awayScore','winner']].values

  imputer.fit(x)
  x = imputer.transform(x)

  clf = RandomForestClassifier(random_state=12)

  clf.fit(x,y)

  dump(clf, 'nhl_ai.joblib')

if __name__ == '__main__':
  if RE_PULL:
    # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
    # client = MongoClient(db_url)
    client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
    db = client["hockey"]
    seasons = db["dev_seasons"].find(
      {},
      {'seasonId': 1}
    )

    # startID = ''
    # endID = 2023020226
    # games = db["dev_games"].find(
    #   # {'id':{'$gte':startID,'$lt':endID+1}},
    #   {'id':{'$lt':endID+1}},
    #   {'id': 1, '_id': 0}
    # )

    pool = Pool(processes=50)
    result = pool.map(season_training_data,seasons)
    # result = pool.map(game_training_data,games)
    result = np.concatenate(result).tolist()
    pool.close()
    dump(result,'training_data/training_data.joblib')
    # f = open('training_data/training_data_text.txt', 'w')
    # f.write(json.dumps(result[60000:60500]))
  else:
    result = load('training_data/training_data.joblib')
    f = open('training_data/training_data_text.txt', 'w')
    f.write(json.dumps(result[60000:60500]))
  print('Games Collected')
  # train(result)