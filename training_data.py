import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

import json
from pymongo import MongoClient
import os
import numpy as np
from joblib import dump, load
import pandas as pd
from multiprocessing import Pool
from util.training_data import season_training_data, game_training_data, update_training_data
from util.helpers import latestIDs
from constants.constants import VERSION, FILE_VERSION, START_SEASON, END_SEASON

dir_path = f'training_data/v{VERSION}'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

tdList = os.listdir(dir_path)

USE_SEASONS = True
UPDATE = True
SKIP_SEASONS = [int(td.replace(f'training_data_v{FILE_VERSION}_','').replace('.joblib','')) for td in tdList] if len(tdList) > 0 and not f'training_data_v{FILE_VERSION}.joblib' in os.listdir('training_data') else []

LATEST_SEASON = 20232024
MAX_ID = 2023020514

if __name__ == '__main__':
  # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
  # client = MongoClient(db_url)
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]
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
      if not UPDATE:
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

  # data = pd.DataFrame(result)
  # for column in data.columns:
  #   class_counts = data[column].value_counts()
  #   print(class_counts)
  #   class_percentages = class_counts / len(data) * 100
  #   print(class_percentages)
    

  dump(result,f'training_data/training_data_v{FILE_VERSION}.joblib')
  f = open('training_data/training_data_text.txt', 'w')
  f.write(json.dumps(result[len(result)-200:len(result)]))
  print(f'Games Collected: v{VERSION}')