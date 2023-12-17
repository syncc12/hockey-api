import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\constants')

import requests
from pymongo import MongoClient
import math
from datetime import datetime
import os
from util.helpers import safe_chain, false_chain, n2n, isNaN, getAge, getPlayer, getPlayerData, projectedLineup
from inputs.inputs import master_inputs
from util.query import get_last_game_team_stats
from joblib import load, dump
from constants.inputConstants import X_V4_INPUTS, Y_V4_OUTPUTS

REPLACE_VALUE = -1

def nhl_data(game,message='',test=False):
  boxscore = boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game['id']}/boxscore").json()
  # db_username = os.getenv('DB_USERNAME')
  # db_name = os.getenv('DB_NAME')
  # db_password = os.getenv('DB_PASSWORD')
  # db_url = f"mongodb+srv://{db_username}:{db_password}@{db_name}"
  db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
  client = MongoClient(db_url)
  db = client['hockey']
  check_pbgs = false_chain(boxscore,'boxscore','playerByGameStats')
  if not check_pbgs:
    away_last_game = projectedLineup(boxscore['awayTeam']['abbrev'],game['id'])
    home_last_game = projectedLineup(boxscore['homeTeam']['abbrev'],game['id'])
    away_last_boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{away_last_game}/boxscore").json()
    home_last_boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{home_last_game}/boxscore").json()
    boxscore['boxscore'] = {
      'playerByGameStats': {
        'awayTeam': away_last_boxscore['boxscore']['playerByGameStats']['awayTeam'],
        'homeTeam': home_last_boxscore['boxscore']['playerByGameStats']['homeTeam'],
      },
      'gameInfo': {
        'awayTeam': away_last_boxscore['boxscore']['gameInfo']['awayTeam'],
        'homeTeam': home_last_boxscore['boxscore']['gameInfo']['homeTeam'],
      }
    }
    message = 'using projected lineup'
    # return {
    #   'data': {
    #     'data': [[]],
    #     'date': -1,
    #     'game_id': game['id'],
    #     'home_team': {
    #       'id': game['homeTeam']['id'],
    #       'city': game['homeTeam']['placeName']['default'],
    #       'name': -1,
    #       'abbreviation': -1,
    #     },
    #     'away_team': {
    #       'id': game['awayTeam']['id'],
    #       'city': game['awayTeam']['placeName']['default'],
    #       'name': -1,
    #       'abbreviation': -1,
    #     },
    #     'live': {
    #       'home_score': -1,
    #       'away_score': -1,
    #       'period': 0,
    #       'clock': 0,
    #       'stopped': True,
    #       'intermission': False,
    #     },
    #   },
    #   'message': 'no boxscore for game',
    # }
  # print(boxscore['homeTeam'])
  inputs = master_inputs(db=db,game=boxscore)
  # print(inputs)
  if not test:
    if inputs:
      x = [[inputs[i] for i in X_V4_INPUTS]]
    else:
      x = [[]]
    return {
      'data': {
        'data': x,
        'game_id': safe_chain(boxscore,'id'),
        'date': safe_chain(boxscore,'gameDate'),
        'state': safe_chain(boxscore,'gameState'),
        'home_team': {
          'id': safe_chain(boxscore,'homeTeam','id'),
          'city': safe_chain(game,'homeTeam','placeName','default'),
          'name': safe_chain(boxscore,'homeTeam','name','default'),
          'abbreviation': safe_chain(boxscore,'homeTeam','abbrev'),
        },
        'away_team': {
          'id': safe_chain(boxscore,'awayTeam','id'),
          'city': safe_chain(game,'awayTeam','placeName','default'),
          'name': safe_chain(boxscore,'awayTeam','name','default'),
          'abbreviation': safe_chain(boxscore,'awayTeam','abbrev'),
        },
        'live': {
          'home_score': safe_chain(boxscore,'homeTeam','score'),
          'away_score': safe_chain(boxscore,'awayTeam','score'),
          'period': safe_chain(boxscore,'period'),
          'clock': safe_chain(boxscore,'clock','timeRemaining'),
          'stopped': not safe_chain(boxscore,'clock','running'),
          'intermission': safe_chain(boxscore,'clock','inIntermission'),
        },
      },
      'message': message,
    }

# game = requests.get(f"https://api-web.nhle.com/v1/schedule/now").json()
# data = nhl_data(game=game['gameWeek'][0]['games'][0])
# print(data)