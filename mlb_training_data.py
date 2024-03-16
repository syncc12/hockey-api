import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import requests
from joblib import dump, load
import pandas as pd
from util.helpers import safe_chain
from pages.mlb.inputs import base_inputs, mlb_training_input, ENCODE_COLUMNS
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.preprocessing import LabelEncoder

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['mlb']

MAX_WORKERS = 10

def fetch_boxscores(i_pk_len):
  i, pk, pk_len = i_pk_len
  boxscore_data = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live").json()
  print(f"{safe_chain(boxscore_data,'gameData','game','season')} {pk} {i+1}/{pk_len}")
  return boxscore_data

def process_boxscores(boxscore_data):
  return base_inputs(boxscore_data)

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

def main(season):
  gamePks = list(Games.find(
    {'season': str(season)},
    {'_id':0,'gamePk':1}
  ))
  gamePks = list(set([pk['gamePk'] for pk in gamePks]))
  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_url = {executor.submit(fetch_boxscores, (i,pk,len(gamePks))): pk for i, pk in enumerate(gamePks)}
    
    data_for_processing = []
    for future in as_completed(future_to_url):
      data = future.result()
      data_for_processing.append(data)

  with ProcessPoolExecutor(max_workers=MAX_WORKERS) as process_executor:
    results = process_executor.map(process_boxscores, data_for_processing)
    results = list(results)
    dump(results, f'pages/mlb/data/training_data_{season}.joblib')

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
    # 2015,
    # 2014,
    # 2013,
    # 2012,
    # 2011,
    # 2010,
    # 2009,
    # 2008,
    # 2007,
    # 2006,
    # 2005,
    2004,
    2003,
    2002,
    2001,
    2000,
  ]
  for season in seasons:
    main(season)
  input_data = mlb_training_input(seasons)
  encode_data(pd.DataFrame(input_data))
  print('done')