import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import requests
from joblib import dump
from inputs import base_inputs
import pandas as pd
# from util.helpers import safe_chain, false_chain

def safe_chain(obj, *keys, default=-1):
  for key in keys:
    if key == default:
      return default
    else:
      if type(key) == int:
        if len(obj) > key:
          obj = obj[key]
        else:
          return default
      else:
        try:
          obj = getattr(obj, key, default) if hasattr(obj, key) else obj[key]
        except (KeyError, TypeError, AttributeError):
          return default
  return obj

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['mlb']
Boxscores = db['boxscores']

def collect_games():
  Games = db['games']
  seasons = ['2023','2022','2021','2020','2019','2018','2017','2016','2015','2014','2013','2012','2011','2010','2009','2008','2007','2006','2005','2004','2003','2002','2001','2000']
  # seasons = ['2022','2021','2020','2019','2018','2017','2016','2015','2014','2013','2012','2011','2010','2009','2008','2007','2006','2005','2004','2003','2002','2001','2000']
  # season_data = requests.get(f"https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1&season=2023").json()
  for season in seasons:
    print(season)
    season_data = requests.get(f"https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1&season={season}").json()
    for day in season_data['dates']:
      print(day['date'],day['totalGames'])
      Games.insert_many(day['games'])

def collect_boxscores(gamePks):
  pk_len = len(gamePks)
  data = []
  for i, pk in enumerate(gamePks):
    try:
      boxscore_data = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live").json()
      # print(safe_chain(boxscore_data,'gameData','game','season'), safe_chain(boxscore_data,'gameData','datetime','officialDate'), pk)
      # Boxscores.insert_one(boxscore_data)
      base_data = base_inputs(boxscore_data)
      data.append(base_data)
      print(f"{safe_chain(boxscore_data,'gameData','game','season')} {pk} {i+1}/{pk_len}")
    except DuplicateKeyError:
      print('DUPLICATE - boxscore', pk)
      pass
  return data
# def collect_boxscores(pk):
#   try:
#     boxscore_data = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live").json()
#     Boxscores.insert_one(boxscore_data)
#     print(safe_chain(boxscore_data,'gameData','game','season'), safe_chain(boxscore_data,'gameData','datetime','officialDate'), pk)
#   except DuplicateKeyError:
#     print('DUPLICATE - boxscore', pk)
#     pass

if __name__ == "__main__":
  Games = db['games']
  seasons = [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000]
  for season in seasons:
    gamePks = list(Games.find(
      {'season': str(season)},
      {'_id':0,'gamePk':1}
    ))
    gamePks = [pk['gamePk'] for pk in gamePks]
    # print(gamePks)
    boxscore_season_data = collect_boxscores(gamePks)
    dump(boxscore_season_data, f'pages/mlb/data/training_data_{season}.joblib')
  print('done')