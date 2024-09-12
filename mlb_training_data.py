import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import requests
from joblib import dump, load
import pandas as pd
from util.helpers import safe_chain
from pages.mlb.inputs import base_inputs, mlb_training_input, raw_data, ENCODE_COLUMNS
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.preprocessing import LabelEncoder

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['mlb']

MAX_WORKERS = 10

def fetch_boxscores(i_pk_len):
  i, pk, pk_len = i_pk_len
  boxscore_data = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live").json()
  print(f"{safe_chain(boxscore_data,'gameData','game','season')} {pk} {i+1}/{pk_len}")
  return boxscore_data

def fetch_data(season):
  return load(f'pages/mlb/data/raw_data_{season}.joblib')

def process_boxscores(boxscore_data):
  return base_inputs(boxscore_data)

def collect_boxscores(season):
  games = load(f'pages/mlb/data/raw_data_{season}.joblib')
  games_len = len(games)
  data = []
  for i, game in enumerate(games):
    base_data = base_inputs(game)
    data.append(base_data)
    print(f"{safe_chain(game,'gameData','game','season')} {safe_chain(game,'gamePk')} {i+1}/{games_len}")
  dump(data, f'pages/mlb/data/training_data_{season}.joblib')
  return data

def analyze_boxscores(season):
  games = load(f'pages/mlb/data/raw_data_{season}.joblib')
  batter_len = []
  pitcher_len = []
  bench_len = []
  bullpen_len = []
  for game in games:
    if len(safe_chain(game,'liveData','boxscore','teams','away','batters')) not in batter_len:
      batter_len.append(len(safe_chain(game,'liveData','boxscore','teams','away','batters')))
    if len(safe_chain(game,'liveData','boxscore','teams','home','batters')) not in batter_len:
      batter_len.append(len(safe_chain(game,'liveData','boxscore','teams','home','batters')))
    if len(safe_chain(game,'liveData','boxscore','teams','away','pitchers')) not in pitcher_len:
      pitcher_len.append(len(safe_chain(game,'liveData','boxscore','teams','away','pitchers')))
    if len(safe_chain(game,'liveData','boxscore','teams','home','pitchers')) not in pitcher_len:
      pitcher_len.append(len(safe_chain(game,'liveData','boxscore','teams','home','pitchers')))
    if len(safe_chain(game,'liveData','boxscore','teams','away','bench')) not in bench_len:
      bench_len.append(len(safe_chain(game,'liveData','boxscore','teams','away','bench')))
    if len(safe_chain(game,'liveData','boxscore','teams','home','bench')) not in bench_len:
      bench_len.append(len(safe_chain(game,'liveData','boxscore','teams','home','bench')))
    if len(safe_chain(game,'liveData','boxscore','teams','away','bullpen')) not in bullpen_len:
      bullpen_len.append(len(safe_chain(game,'liveData','boxscore','teams','away','bullpen')))
    if len(safe_chain(game,'liveData','boxscore','teams','home','bullpen')) not in bullpen_len:
      bullpen_len.append(len(safe_chain(game,'liveData','boxscore','teams','home','bullpen')))
  print(f"batter: {batter_len}")
  print(f"pitcher: {pitcher_len}")
  print(f"bench: {bench_len}")
  print(f"bullpen: {bullpen_len}")


def process_raw_data(data):
  return raw_data(data)

def fetch_raw_data(i_pk_len):
  i, pk, pk_len = i_pk_len
  data = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live").json()
  print(f"{safe_chain(data,'gameData','game','season')} {pk} {i+1}/{pk_len}")
  return data

def raw_main(season):
  gamePks = list(Games.find(
    {'season': str(season)},
    {'_id':0,'gamePk':1}
  ))
  gamePks = list(set([pk['gamePk'] for pk in gamePks]))
  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_url = {executor.submit(fetch_raw_data, (i,pk,len(gamePks))): pk for i, pk in enumerate(gamePks)}
    
    data_for_processing = []
    for future in as_completed(future_to_url):
      data = future.result()
      data_for_processing.append(data)

  with ProcessPoolExecutor(max_workers=MAX_WORKERS) as process_executor:
    results = process_executor.map(process_raw_data, data_for_processing)
    results = list(results)
    dump(results, f'pages/mlb/data/raw_data_{season}.joblib')
  

def main(season):
  print(season, 'START')
  # gamePks = list(Games.find(
  #   {'season': str(season)},
  #   {'_id':0,'gamePk':1}
  # ))
  # gamePks = list(set([pk['gamePk'] for pk in gamePks]))
  # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
  #   future_to_url = {executor.submit(fetch_boxscores, (i,pk,len(gamePks))): pk for i, pk in enumerate(gamePks)}
    
  #   data_for_processing = []
  #   for future in as_completed(future_to_url):
  #     data = future.result()
  #     data_for_processing.append(data)
  
  data_for_processing = fetch_data(season)

  with ProcessPoolExecutor(max_workers=MAX_WORKERS) as process_executor:
    results = process_executor.map(process_boxscores, data_for_processing)
    results = list(results)
    dump(results, f'pages/mlb/data/training_data_{season}.joblib')

  print(season, 'DONE')

def encode_data(data):
  for column in ENCODE_COLUMNS:
    print(data[column].value_counts())
    data = data[data[column] != -1]
    encoder = LabelEncoder()
    encoder.fit_transform(data[column])
    dump(encoder, f'pages/mlb/encoders/{column}_encoder.joblib')


if __name__ == "__main__":
  Games = db['games']
  seasons = [
    # 2023,
    # 2022,
    # 2021,
    # 2020,
    # 2019,
    # 2018,
    # 2017,
    # 2016,
    2015,
    2014,
    2013,
    2012,
    2011,
    2010,
    2009,
    2008,
    # 2007,
    # 2006,
    # 2005,
    # 2004,
    # 2003,
    # 2002,
    # 2001,
    # 2000,
  ]
  # analyze_boxscores(2023)
  RAW = True
  if RAW:
    for season in seasons:
      raw_main(season)
  else:
    for season in seasons:
      main(season)
    input_data = mlb_training_input(seasons)
    encode_data(pd.DataFrame(input_data))
  print('done')