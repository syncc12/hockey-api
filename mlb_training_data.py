import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import requests
from joblib import dump
import pandas as pd
from util.helpers import safe_chain
from pages.mlb.inputs import base_inputs

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['mlb']

def collect_boxscores(gamePks):
  pk_len = len(gamePks)
  data = []
  for i, pk in enumerate(gamePks):
    try:
      boxscore_data = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live").json()
      base_data = base_inputs(boxscore_data)
      data.append(base_data)
      print(f"{safe_chain(boxscore_data,'gameData','game','season')} {pk} {i+1}/{pk_len}")
    except DuplicateKeyError:
      print('DUPLICATE - boxscore', pk)
      pass
  return data

if __name__ == "__main__":
  Games = db['games']
  seasons = [
    # 2023,
    # 2022,
    2021,
    2020,
    2019,
    2018,
    2017,
    2016,
    2015,
    2014,
    2013,
    2012,
    2011,
    2010,
    2009,
    2008,
    2007,
    2006,
    2005,
    2004,
    2003,
    2002,
    2001,
    2000
  ]
  for season in seasons:
    gamePks = list(Games.find(
      {'season': str(season)},
      {'_id':0,'gamePk':1}
    ))
    gamePks = list(set([pk['gamePk'] for pk in gamePks]))
    # print(gamePks)
    boxscore_season_data = collect_boxscores(gamePks)
    dump(boxscore_season_data, f'pages/mlb/data/training_data_{season}.joblib')
  print('done')