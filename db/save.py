import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import requests
from joblib import dump, load
# from util.helpers import safe_chain, false_chain

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['mlb']
Boxscores = db['boxscores']

def save_mlb_boxscores(data):
  Boxscores.insert_many(data)

if __name__ == "__main__":
  season = 2023
  data = load(f'pages/mlb/data/boxscore_data_{season}.joblib')
  save_mlb_boxscores(data)