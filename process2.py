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
from constants.inputConstants import X_INPUTS, Y_OUTPUTS

REPLACE_VALUE = -1

def nhl_data(db,game,message='',test=False):
  isProjectedLineup = False
  if not test:
    Odds = db['dev_odds']
    game_odds = Odds.find_one(
      {'id':game['id']},
      {'_id':0,'odds':1}
    )
    boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game['id']}/boxscore").json()
    check_pbgs = false_chain(boxscore,'boxscore','playerByGameStats')
    if not check_pbgs:
      isProjectedLineup = True
      away_last_game, away_home_away = projectedLineup(boxscore['awayTeam']['abbrev'],game['id'])
      home_last_game, home_home_away = projectedLineup(boxscore['homeTeam']['abbrev'],game['id'])
      away_last_boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{away_last_game}/boxscore").json()
      home_last_boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{home_last_game}/boxscore").json()
      boxscore['boxscore'] = {
        'playerByGameStats': {
          'awayTeam': away_last_boxscore['boxscore']['playerByGameStats'][away_home_away],
          'homeTeam': home_last_boxscore['boxscore']['playerByGameStats'][home_home_away],
        },
        'gameInfo': {
          'awayTeam': away_last_boxscore['boxscore']['gameInfo'][away_home_away],
          'homeTeam': home_last_boxscore['boxscore']['gameInfo'][home_home_away],
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
    

  inputs = master_inputs(db=db,game=boxscore,isProjectedLineup=isProjectedLineup)

  if not test:
    input_data = {}
    if inputs:
      if safe_chain(inputs,'options','projectedLineup') and safe_chain(inputs,'options','projectedLineup') != -1:
        input_keys = list(inputs['data'].keys())
        x = {
          f'{input_keys[0]}': [[list(inputs['data'].values())[0][i] for i in X_INPUTS]],
          f'{input_keys[1]}': [[list(inputs['data'].values())[1][i] for i in X_INPUTS]],
          f'{input_keys[2]}': [[list(inputs['data'].values())[2][i] for i in X_INPUTS]],
          f'{input_keys[3]}': [[list(inputs['data'].values())[3][i] for i in X_INPUTS]],
        }
        input_data[input_keys[0]] = {}
        input_data[input_keys[1]] = {}
        input_data[input_keys[2]] = {}
        input_data[input_keys[3]] = {}
        for i in X_INPUTS:
          input_data[input_keys[0]][i] = inputs['data'][input_keys[0]][i]
          input_data[input_keys[1]][i] = inputs['data'][input_keys[1]][i]
          input_data[input_keys[2]][i] = inputs['data'][input_keys[2]][i]
          input_data[input_keys[3]][i] = inputs['data'][input_keys[3]][i]
      else:
        x = [[inputs['data'][i] for i in X_INPUTS]]
        for i in X_INPUTS:
          input_data[i] = inputs['data'][i]
    else:
      x = [[]]
    homeOdds = float(game_odds['odds']['homeTeam']) if game_odds else -1
    awayOdds = float(game_odds['odds']['awayTeam']) if game_odds else -1
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
          'odds': homeOdds,
        },
        'away_team': {
          'id': safe_chain(boxscore,'awayTeam','id'),
          'city': safe_chain(game,'awayTeam','placeName','default'),
          'name': safe_chain(boxscore,'awayTeam','name','default'),
          'abbreviation': safe_chain(boxscore,'awayTeam','abbrev'),
          'odds': awayOdds,
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
      'isProjectedLineup': inputs['options']['projectedLineup'],
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
      'totalGoals': awayScore + homeScore,
      'goalDifferential': abs(awayScore - homeScore),
      'finalPeriod':safe_chain(boxscore,'period'),
      'pastRegulation':1 if safe_chain(boxscore,'period') > 3 else 0,
      'awayShots':safe_chain(boxscore,'awayTeam','sog'),
      'homeShots':safe_chain(boxscore,'homeTeam','sog'),
      'awayShotsPeriod1':safe_chain(boxscore,'boxscore','shotsByPeriod',0,'away'),
      'homeShotsPeriod1':safe_chain(boxscore,'boxscore','shotsByPeriod',0,'home'),
      'awayShotsPeriod2':safe_chain(boxscore,'boxscore','shotsByPeriod',1,'away'),
      'homeShotsPeriod2':safe_chain(boxscore,'boxscore','shotsByPeriod',1,'home'),
      'awayShotsPeriod3':safe_chain(boxscore,'boxscore','shotsByPeriod',2,'away'),
      'homeShotsPeriod3':safe_chain(boxscore,'boxscore','shotsByPeriod',2,'home'),
      'awayShotsPeriod4':safe_chain(boxscore,'boxscore','shotsByPeriod',3,'away'),
      'homeShotsPeriod4':safe_chain(boxscore,'boxscore','shotsByPeriod',3,'home'),
      'awayShotsPeriod5':safe_chain(boxscore,'boxscore','shotsByPeriod',4,'away'),
      'homeShotsPeriod5':safe_chain(boxscore,'boxscore','shotsByPeriod',4,'home'),
      'awayScorePeriod1':safe_chain(boxscore,'boxscore','linescore','byPeriod',0,'away'),
      'homeScorePeriod1':safe_chain(boxscore,'boxscore','linescore','byPeriod',0,'home'),
      'awayScorePeriod2':safe_chain(boxscore,'boxscore','linescore','byPeriod',1,'away'),
      'homeScorePeriod2':safe_chain(boxscore,'boxscore','linescore','byPeriod',1,'home'),
      'awayScorePeriod3':safe_chain(boxscore,'boxscore','linescore','byPeriod',2,'away'),
      'homeScorePeriod3':safe_chain(boxscore,'boxscore','linescore','byPeriod',2,'home'),
      'awayScorePeriod4':safe_chain(boxscore,'boxscore','linescore','byPeriod',3,'away'),
      'homeScorePeriod4':safe_chain(boxscore,'boxscore','linescore','byPeriod',3,'home'),
      'awayScorePeriod5':safe_chain(boxscore,'boxscore','linescore','byPeriod',4,'away'),
      'homeScorePeriod5':safe_chain(boxscore,'boxscore','linescore','byPeriod',4,'home'),
      'period1PuckLine':abs(safe_chain(boxscore,'boxscore','linescore','byPeriod',0,'away') - safe_chain(boxscore,'boxscore','linescore','byPeriod',0,'home')),
      'period2PuckLine':abs(safe_chain(boxscore,'boxscore','linescore','byPeriod',1,'away') - safe_chain(boxscore,'boxscore','linescore','byPeriod',1,'home')),
      'period3PuckLine':abs(safe_chain(boxscore,'boxscore','linescore','byPeriod',2,'away') - safe_chain(boxscore,'boxscore','linescore','byPeriod',2,'home')),
    }
    x_data = []
    y_data = []
    input_data = {}
    x = [[inputs['data'][i] for i in X_INPUTS]]
    for i in X_INPUTS:
      input_data[i] = inputs['data'][i]
    for i in X_INPUTS:
      if i in suplement_data.keys():
        x_data.append(suplement_data[i])
      else:
        x_data.append(inputs['data'][i])
    for i in Y_OUTPUTS:
      if i in suplement_data.keys():
        y_data.append(suplement_data[i])
      elif i in inputs:
        y_data.append(inputs['data'][i])

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

def nhl_test(db,boxscore):
  return nhl_data(db=db,game=boxscore,test=True)