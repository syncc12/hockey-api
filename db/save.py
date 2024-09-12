import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import requests
from joblib import dump, load
# from util.helpers import safe_chain, false_chain

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['hockey']
Boxscores = db['mlb_boxscores']

def save_mlb_boxscores(season):
  data = load(f'pages/mlb/data/boxscore_data_{season}.joblib')
  data_len = len(data)
  for i, dict_ in enumerate(data):
    try:
      dict_.pop('liveData', None)
      Boxscores.insert_one(dict_)
      print(f"{season}: [{i+1}/{data_len}] {dict_['gamePk']}")
    except DuplicateKeyError:
      print(f"DUPLICATE: [{i+1}/{data_len}] {dict_['gamePk']}")
      pass

if __name__ == "__main__":
  seasons = [2020, 2019, 2018]
            #  [2017, 2016, 2015, 2014, 2013, 2012,
            #  2011, 2010, 2009, 2008, 2007, 2006,
            #  2005, 2004, 2003, 2002, 2001, 2000]
  for season in seasons:
    save_mlb_boxscores(season)