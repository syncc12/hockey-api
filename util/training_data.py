import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
from datetime import datetime
from util.helpers import safe_chain, false_chain, getLastTeamLineups
from inputs.inputs import master_inputs
from inputs.roster import roster_inputs
import os
from joblib import dump
from constants.constants import VERSION, FILE_VERSION, PROJECTED_LINEUP_VERSION, PROJECTED_LINEUP_FILE_VERSION

# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
# client = MongoClient(db_url)
client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
Boxscores = db["dev_boxscores"]
Games = db["dev_games"]
Rosters = db["dev_rosters"]

# VERSION = 6
def season_training_data(season,test_data=False):
  print('fired',season)

  training_data = []
  boxscores = list(Boxscores.find(
    {'season': int(season)},
    {'id': 1, 'season': 1, 'gameType': 1, 'gameDate': 1, 'venue': 1, 'neutralSite': 1, 'homeTeam': 1, 'awayTeam': 1, 'playerByGameStats': 1, 'summary': 1, 'periodDescriptor': 1}
  ))
  games = list(Games.find(
    {'season': int(season)},
    {'id': 1, 'neutralSite': 1, 'homeTeam': 1, 'awayTeam': 1}
  ))
  for i in range(0,len(boxscores)):
    if safe_chain(boxscores,i,'gameType') != 2 and safe_chain(boxscores,i,'gameType') != 3:
      print(season,f'{i+1}/{len(boxscores)} - SKIPPED')
      continue
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
      'periodDescriptor': safe_chain(boxscores,i,'periodDescriptor'),
      'homeTeam': homeTeam,
      'awayTeam': awayTeam,
      'playerByGameStats': safe_chain(boxscores,i,'playerByGameStats'),
      'summary': safe_chain(boxscores,i,'summary'),
      'neutralSite': safe_chain(games,i,'neutralSite'),
    }

    boxscore_data = master_inputs(db=db, boxscore=game_data)['data']
    if boxscore_data:
      training_data.append(boxscore_data)
    print(season,f'{i+1}/{len(games)}')
  print('DONE ',season)
  if not test_data:
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


def season_training_data_projectedLineup(season):
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
    home_roster = Rosters.find_one({'season': season, 'team': homeTeam['abbrev'].lower()})
    away_roster = Rosters.find_one({'season': season, 'team': awayTeam['abbrev'].lower()})
    game_data = {
      'id': id,
      'season': safe_chain(boxscores,i,'season'),
      'gameType': safe_chain(boxscores,i,'gameType'),
      'gameDate': safe_chain(boxscores,i,'gameDate'),
      'venue': safe_chain(boxscores,i,'venue'),
      'homeTeam': homeTeam,
      'awayTeam': awayTeam,
      'boxscore': safe_chain(boxscores,i,'boxscore'),
      'neutralSite': safe_chain(games,i,'neutralSite'),
    }

    home_data = roster_inputs(db=db, boxscore=game_data, roster=home_roster, homeAway='homeTeam')
    away_data = roster_inputs(db=db, boxscore=game_data, roster=away_roster, homeAway='awayTeam')
    if home_data:
      training_data.append(home_data)
    if away_data:
      training_data.append(away_data)
    print(season,f'{i+1}/{len(games)}')
  print('DONE ',season)
  # dump(training_data,f"training_data/v{PROJECTED_LINEUP_VERSION}/projected_lineup/training_data_v{PROJECTED_LINEUP_FILE_VERSION}_{season}_projectedLineup.joblib")
  return training_data


def season_test_data(season):
  print('fired',season)

  test_data = []
  boxscores = list(Boxscores.find(
    {'season': int(season)},
    {'id': 1, 'season': 1, 'gameType': 1, 'gameDate': 1, 'venue': 1, 'neutralSite': 1, 'homeTeam': 1, 'awayTeam': 1, 'playerByGameStats': 1, 'summary': 1, 'periodDescriptor': 1}
  ))
  games = list(Games.find(
    {'season': int(season)},
    {'id': 1, 'neutralSite': 1, 'homeTeam': 1, 'awayTeam': 1}
  ))
  for i in range(0,len(boxscores)):
    if safe_chain(boxscores,i,'gameType') != 2 and safe_chain(boxscores,i,'gameType') != 3:
      print(season,f'{i+1}/{len(boxscores)} - SKIPPED')
      continue
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
      'periodDescriptor': safe_chain(boxscores,i,'periodDescriptor'),
      'homeTeam': homeTeam,
      'awayTeam': awayTeam,
      'playerByGameStats': safe_chain(boxscores,i,'playerByGameStats'),
      'summary': safe_chain(boxscores,i,'summary'),
      'neutralSite': safe_chain(games,i,'neutralSite'),
    }

    boxscore_data = master_inputs(db=db, boxscore=game_data)['data']
    if boxscore_data:
      test_data.append(boxscore_data)
    print(season,f'{i+1}/{len(boxscores)}')
  print('DONE ',season)
  return test_data