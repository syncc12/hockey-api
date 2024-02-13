from pymongo import MongoClient
import requests

# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
# client = MongoClient(db_url)
client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
Rosters = db["dev_rosters"]
teams = [
  {'name':'car','id':26},
  {'name':'cbj','id':36},
  {'name':'njd','id':23},
  {'name':'nyi','id':22},
  {'name':'nyr','id':10},
  {'name':'phi','id':16},
  {'name':'pit','id':17},
  {'name':'wsh','id':24},
  {'name':'bos','id':6},
  {'name':'buf','id':19},
  {'name':'det','id':12},
  {'name':'fla','id':33},
  {'name':'mtl','id':1},
  {'name':'ott','id':30},
  {'name':'tbl','id':31},
  {'name':'tor','id':5},
  {'name':'ari','id':28},
  {'name':'chi','id':11},
  {'name':'col','id':27},
  {'name':'dal','id':15},
  {'name':'min','id':37},
  {'name':'nsh','id':34},
  {'name':'stl','id':18},
  {'name':'wpg','id':35},
  {'name':'ana','id':32},
  {'name':'cgy','id':21},
  {'name':'edm','id':25},
  {'name':'lak','id':14},
  {'name':'sjs','id':29},
  {'name':'sea','id':39},
  {'name':'van','id':20},
  {'name':'vgk','id':38},
]
# teams = [{'name':'car','id':26}]

for team in teams:
  team_name = team['name']
  season_url = f'https://api-web.nhle.com/v1/roster-season/{team_name}'
  seasons = requests.get(season_url).json()[-20:-10]
  # seasons = requests.get(season_url).json()[-10:]
  for season in seasons:
    roster_url = f'https://api-web.nhle.com/v1/roster/{team_name}/{season}'
    roster = requests.get(roster_url).json()
    payload = {
      'id': f"{season}{team['id']}",
      'teamId': team['id'],
      'team': team_name,
      'season': season,
      'roster': roster,
    }
    Rosters.insert_one(payload)
    print(season)
  print(team_name)