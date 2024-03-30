from pymongo import MongoClient
import requests

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['hockey']
Boxscores = db['dev_boxscores']


def collect_game_ids(season):
  boxscores = list(Boxscores.find({'season': season},{'_id': 0, 'id': 1}))
  return [boxscore['id'] for boxscore in boxscores]

def get_new_boxscore(gameId):
  return requests.get(f"https://api-web.nhle.com/v1/gamecenter/{gameId}/boxscore").json()

def get_all_new_boxscores(gameIds):
  boxscores = []
  gameIds_len = len(gameIds)
  for i, gameId in enumerate(gameIds):
    boxscore = get_new_boxscore(gameId)
    boxscores.append(boxscore)
    season = f'{str(gameId)[:4]}{int(str(gameId)[:4])+1}'
    print(f"[{i+1}/{gameIds_len}] {season} {gameId} Collected")
  return boxscores

def replace_boxscore(gameId, boxscore):
  Boxscores.replace_one({'id': gameId}, boxscore)

def main(season):
  gameIds = collect_game_ids(season)
  print('GameIds Collected')
  gameIds_len = len(gameIds)
  boxscores = get_all_new_boxscores(gameIds)
  print('Boxscores Collected')
  for i, boxscore in enumerate(boxscores):
    gameId = boxscore['id']
    replace_boxscore(gameId, boxscore)
    print(f"[{i+1}/{gameIds_len}] {season} {gameId} replaced")

if __name__ == '__main__':
  SEASONS = [
    # 20222023,
    # 20212022,
    # 20202021,
    # 20192020,
    # 20182019,
    # 20172018,
    # 20162017,
    # 20152016,
    # 20142015,
    # 20132014,
    # 20122013,
    # 20112012,
    # 20102011,
    # 20092010,
    20082009,
    20072008,
    20062007,
    20052006,
    20032004,
    20022003,
    20012002,
    20002001,
    19992000,
    19981999,
    19971998,
    19961997,
    19951996,
    19941995,
    19931994,
    19921993,
    19911992,
    # 19901991,
  ]
  for season in SEASONS:
    main(season)
