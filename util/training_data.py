import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
from datetime import datetime
from util.helpers import safe_chain, false_chain
from inputs.inputs import master_inputs
import os
from joblib import dump
from constants.constants import VERSION, FILE_VERSION

# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
# client = MongoClient(db_url)
client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
Boxscores = db["dev_boxscores"]
Games = db["dev_games"]

# VERSION = 6
def season_training_data(season):
  print('fired',season)

  training_data = []
  boxscores = list(Boxscores.find(
    {'season': int(season)},
    {'id': 1, 'season': 1, 'gameType': 1, 'gameDate': 1, 'venue': 1, 'neutralSite': 1, 'homeTeam': 1, 'awayTeam': 1, 'boxscore': 1, 'period': 1}
  ))
  games = list(Games.find(
    {'season': int(season)},
    {'id': 1, 'neutralSite': 1, 'homeTeam': 1, 'awayTeam': 1}
  ))
  for i in range(0,len(games)):
    if false_chain(boxscores,i,'id'):
      id = safe_chain(boxscores,i,'id')
    else:
      id = safe_chain(games,i,'id')
    if false_chain(boxscores,i,'homeTeam'):
      homeTeam = safe_chain(boxscores,i,'homeTeam')
    else:
      homeTeam = safe_chain(games,i,'homeTeam')
    if false_chain(boxscores,i,'awayTeam'):
      awayTeam = safe_chain(boxscores,i,'awayTeam')
    else:
      awayTeam = safe_chain(games,i,'awayTeam')
    game_data = {
      'id': id,
      'season': safe_chain(boxscores,i,'season'),
      'gameType': safe_chain(boxscores,i,'gameType'),
      'gameDate': safe_chain(boxscores,i,'gameDate'),
      'venue': safe_chain(boxscores,i,'venue'),
      'period': safe_chain(boxscores,i,'period'),
      'homeTeam': homeTeam,
      'awayTeam': awayTeam,
      'boxscore': safe_chain(boxscores,i,'boxscore'),
      'neutralSite': safe_chain(games,i,'neutralSite'),
      # 'homeSplitSquad': safe_chain(games,i,'homeTeam','homeSplitSquad'),
      # 'awaySplitSquad': safe_chain(games,i,'awayTeam','awaySplitSquad'),
    }

    boxscore_data = master_inputs(db=db, game=game_data)['data']
    if boxscore_data:
      training_data.append(boxscore_data)
    print(season,f'{i+1}/{len(games)}')
  print('DONE ',season)
  dump(training_data,f"training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib")
  return training_data

def game_training_data(gameId):
  print('fired',gameId['id'])

  training_data = []
  boxscores = list(Boxscores.find(
    {'id': int(gameId['id'])},
    {'id': 1, 'season': 1, 'gameType': 1, 'gameDate': 1, 'venue': 1, 'homeTeam': 1, 'awayTeam': 1, 'boxscore': 1, 'period':1}
  ))
  games = list(Games.find(
    {'id': int(gameId['id'])},
    {'id': 1, 'neutralSite': 1, 'homeTeam': 1, 'awayTeam': 1}
  ))

  game_data = {
    'id': safe_chain(boxscores,0,'id'),
    'season': safe_chain(boxscores,0,'season'),
    'gameType': safe_chain(boxscores,0,'gameType'),
    'gameDate': safe_chain(boxscores,0,'gameDate'),
    'venue': safe_chain(boxscores,0,'venue'),
    'period': safe_chain(boxscores,0,'period'),
    'homeTeam': safe_chain(boxscores,0,'homeTeam'),
    'awayTeam': safe_chain(boxscores,0,'awayTeam'),
    'boxscore': safe_chain(boxscores,0,'boxscore'),
    'neutralSite': safe_chain(games,0,'neutralSite'),
    'homeSplitSquad': safe_chain(games,0,'homeTeam','homeSplitSquad'),
    'awaySplitSquad': safe_chain(games,0,'awayTeam','awaySplitSquad'),
  }
  boxscore_data = master_inputs(db=db, game=game_data)

  if boxscore_data:
    training_data.append(boxscore_data)
  print('DONE ', gameId['id'])
  return training_data


# def compile_data(id,boxscores,games):
#   game_data = []
#   for boxscore in boxscores:
#     game_data.append(boxscore_data)
#   now = datetime.now()
#   print(id, len(boxscores), f'{now.hour-last_time.hour}:{now.minute-last_time.minute}:{float(f"{now.second}.{now.microsecond}")-float(f"{last_time.second}.{last_time.microsecond}")}')
#   last_time = now
#   return game_data

# def save_training_data(boxscores,neutralSite):

#   game_data = {
#     'id': safe_chain(boxscores,0,'id'),
#     'season': safe_chain(boxscores,0,'season'),
#     'gameType': safe_chain(boxscores,0,'gameType'),
#     'gameDate': safe_chain(boxscores,0,'gameDate'),
#     'venue': safe_chain(boxscores,0,'venue'),
#     'homeTeam': safe_chain(boxscores,0,'homeTeam'),
#     'awayTeam': safe_chain(boxscores,0,'awayTeam'),
#     'boxscore': safe_chain(boxscores,0,'boxscore'),
#     'neutralSite': neutralSite,
#     'homeSplitSquad': False,
#     'awaySplitSquad': False,
#   }
#   boxscore_data = compile_training_data(db=db, game=game_data)
#   return boxscore_data

def update_training_data(gameId):
  print('fired',gameId['id'])

  boxscores = list(Boxscores.find(
    {'id': int(gameId['id'])},
    {'id': 1, 'season': 1, 'gameType': 1, 'gameDate': 1, 'venue': 1, 'homeTeam': 1, 'awayTeam': 1, 'boxscore': 1, 'period':1}
  ))
  games = list(Games.find(
    {'id': int(gameId['id'])},
    {'id': 1, 'neutralSite': 1, 'homeTeam': 1, 'awayTeam': 1}
  ))

  game_data = {
    'id': safe_chain(boxscores,0,'id'),
    'season': safe_chain(boxscores,0,'season'),
    'gameType': safe_chain(boxscores,0,'gameType'),
    'gameDate': safe_chain(boxscores,0,'gameDate'),
    'venue': safe_chain(boxscores,0,'venue'),
    'period': safe_chain(boxscores,0,'period'),
    'homeTeam': safe_chain(boxscores,0,'homeTeam'),
    'awayTeam': safe_chain(boxscores,0,'awayTeam'),
    'boxscore': safe_chain(boxscores,0,'boxscore'),
    'neutralSite': safe_chain(games,0,'neutralSite'),
  }
  training_data = master_inputs(db=db, game=game_data)

  print('DONE ', gameId['id'])
  return training_data