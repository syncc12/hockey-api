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
db = client['mlb']
Boxscores = db['boxscores']
# Seasons = db['dev_seasons']

# seasons = list(Seasons.find({},{'_id':0,'seasonId':1}))
# seasons = sorted([season['seasonId'] for season in seasons])
# print(seasons)

def reduce_seasons():
  # clean_seasons = [19171918, 19181919, 19191920, 19201921, 19211922, 19221923, 19231924, 19241925, 19251926, 19261927, 19271928, 19281929, 19291930, 19301931, 19311932, 19321933, 19331934, 19341935, 19351936, 19361937, 19371938, 19381939, 19391940, 19401941, 19411942, 19421943, 19431944, 19441945, 19451946, 19461947, 19471948, 19481949, 19491950, 19501951, 19511952, 19521953, 19531954, 19541955, 19551956, 19561957, 19571958, 19581959, 19591960, 19601961, 19611962, 19621963, 19631964, 19641965, 19651966, 19661967, 19671968, 19681969, 19691970, 19701971, 19711972, 19721973, 19731974, 19741975, 19751976, 19761977, 19771978, 19781979, 19791980, 19801981, 19811982, 19821983, 19831984, 19841985, 19851986, 19861987, 19871988, 19881989, 19891990]
  clean_seasons = [19981999, 19981997, 19971996, 19961995, 19951994, 19941993, 19931992, 19921991, 19911990, 19901989]
  # clean_seasons = [19171918, 19181919]
  # boxscores = list(Boxscores.find({'season': {'$nin': clean_seasons}}))
  # boxscores = sorted([boxscore['id'] for boxscore in boxscores])
  # print(boxscores)
  # print(len(boxscores))
  Boxscores.delete_many({'season': {'$in': clean_seasons}})

def save_mlb_boxscores(data):
  Boxscores.insert_many(data)

if __name__ == "__main__":
  reduce_seasons()