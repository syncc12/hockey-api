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
from constants.inputConstants import X_V6_INPUTS, Y_V6_OUTPUTS

REPLACE_VALUE = -1

def nhl_data(game,message='',test=False):
  db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
  client = MongoClient(db_url)
  db = client['hockey']
  if not test:
    boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game['id']}/boxscore").json()
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
  else:
    boxscore = game
    

  inputs = master_inputs(db=db,game=boxscore)

  if not test:
    input_data = {}
    if inputs:
      x = [[inputs[i] for i in X_V6_INPUTS]]
      for i in X_V6_INPUTS:
        input_data[i] = inputs[i]
    else:
      x = [[]]
    return {
      'data': {
        'data': x,
        'input_data': input_data,
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
  else:
    homeTeam = safe_chain(boxscore,'homeTeam','id')
    awayTeam = safe_chain(boxscore,'awayTeam','id')
    awayScore = safe_chain(boxscore,'awayTeam','score')
    homeScore = safe_chain(boxscore,'homeTeam','score')
    suplement_data = {
      'id': safe_chain(boxscore,'id'),
      'season': safe_chain(boxscore,'season'),
      'gameType': safe_chain(boxscore,'gameType'),
      'venue': n2n(safe_chain(boxscore,'venue','default')),
      'neutralSite': 0,
      'homeTeam': homeTeam,
      'awayTeam': awayTeam,
      'awayScore': awayScore,
      'homeScore': homeScore,
      'winner': homeTeam if homeScore > awayScore else awayTeam,
    }
    x_data = []
    y_data = []
    input_data = {}
    x = [[inputs[i] for i in X_V6_INPUTS]]
    for i in X_V6_INPUTS:
      input_data[i] = inputs[i]
    for i in X_V6_INPUTS:
      if i in suplement_data.keys():
        x_data.append(suplement_data[i])
      else:
        x_data.append(inputs[i])
    for i in Y_V6_OUTPUTS:
      if i in suplement_data.keys():
        y_data.append(suplement_data[i])
      elif i in inputs:
        y_data.append(inputs[i])

    test_data = [x_data]
    test_result = [y_data]
    return {
      'data':test_data,
      'result':test_result,
      'input_data': input_data,
      }


# game = requests.get(f"https://api-web.nhle.com/v1/schedule/now").json()
# data = nhl_data(game=game['gameWeek'][0]['games'][0])
# print(data)

def nhl_test(boxscore):
  return nhl_data(game=boxscore,test=True)