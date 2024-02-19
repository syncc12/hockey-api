import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

import requests
from pymongo import MongoClient
import math
from datetime import datetime
import pandas as pd
import os
from joblib import dump
from util.helpers import safe_chain, false_chain, n2n, isNaN, getAge, getPlayer, getPlayerData, projectedLineup, projected_roster
from inputs.inputs import master_inputs
from util.query import get_last_game_team_stats
from util.models import MODEL_NAMES
from constants.inputConstants import X_INPUTS, Y_OUTPUTS

REPLACE_VALUE = -1

def nhl_data(db,game,useProjectedLineup,message='',test=False):
  isProjectedLineup = useProjectedLineup
  if not test:
    Odds = db['dev_odds']
    game_odds = Odds.find_one(
      {'id':game['id']},
      {'_id':0,'odds':1}
    )
    boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game['id']}/boxscore").json()
    check_pbgs = false_chain(boxscore,'boxscore','playerByGameStats')
    if not check_pbgs or useProjectedLineup:
      isProjectedLineup = True
      # away_roster, home_roster = projected_roster(game['id'])
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
        },
        # 'projectedRoster': {
        #   'awayTeam': away_roster,
        #   'homeTeam': home_roster,
        # }
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
    

  inputs = master_inputs(db=db,boxscore=boxscore,isProjectedLineup=isProjectedLineup)
  
  if not test:
    input_data = {}
    if inputs:
      if safe_chain(inputs,'options','projectedLineup') and safe_chain(inputs,'options','projectedLineup') != -1:
        input_keys = list(inputs['data'].keys())
        x = {}
        
        # print(list(inputs['data'].values()))
        for i in range(0, len(input_keys)):
          input_list = list(inputs['data'].values())[i]
          x[f'{input_keys[i]}'] = [[input_list[j] for j in X_INPUTS]]

        for i in input_keys:
          input_data[i] = {}

        for i in X_INPUTS:
          for j in input_keys:
            input_data[j][i] = inputs['data'][j][i]
            
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
    # model_names_suplement = [
    #   'id',
    #   'season',
    #   'gameType',
    #   'venue',
    #   'neutralSite',
    #   'homeTeam',
    #   'awayTeam',
    # ]
    # for name in model_names_suplement:
    #   MODEL_NAMES.append(name)
    
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

    if isProjectedLineup:
      x_data = {}
      y_data = {}
      
      input_data = {}
      for data_input in inputs['data'].keys():
        x_data[data_input] = {}
        y_data[data_input] = {}
        for i in MODEL_NAMES:
          x_data[data_input][i] = []
          y_data[data_input][i] = 0

        input_data[data_input] = {}
        for i in X_INPUTS:
          input_data[data_input][i] = inputs['data'][data_input][i]
        for j in MODEL_NAMES:
          for i in X_INPUTS:
            if i in suplement_data.keys():
              x_data[data_input][j].append(suplement_data[i])
            else:
              x_data[data_input][j].append(inputs['data'][data_input][i])
        for i in Y_OUTPUTS:
          if i in suplement_data.keys():
            y_data[data_input][i] = suplement_data[i]
          elif i in inputs:
            y_data[data_input][i] = inputs['data'][data_input][i]

      test_data = x_data
      test_result = y_data

      return {
        'data':test_data,
        'result':test_result,
        'input_data': input_data,
      }
    else:
      x_data = {}
      y_data = {}
      for i in MODEL_NAMES:
        x_data[i] = []
        y_data[i] = 0

      input_data = {}
      for i in X_INPUTS:
        input_data[i] = inputs['data'][i]
      for j in MODEL_NAMES:
        for i in X_INPUTS:
          if i in suplement_data.keys():
            x_data[j].append(suplement_data[i])
          else:
            x_data[j].append(inputs['data'][i])
      for i in Y_OUTPUTS:
        if i in suplement_data.keys():
          y_data[i] = suplement_data[i]
        elif i in inputs:
          y_data[i] = inputs['data'][i]

      test_data = x_data
      test_result = y_data

      return {
        'data':test_data,
        'result':test_result,
        'input_data': input_data,
      }


# game = requests.get(f"https://api-web.nhle.com/v1/schedule/now").json()
# data = nhl_data(game=game['gameWeek'][0]['games'][0])
# print(data)
    

def nhl_data2(db,games,useProjectedLineups=[],messages=[''],test=False):
  input_data = []
  game_data = []
  extra_data = []
  for i in range(0,len(games)):
    game = games[i]
    useProjectedLineup = useProjectedLineups[i]
    isProjectedLineup = useProjectedLineup
    message = ''

    boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game['id']}/boxscore").json()
    check_pbgs = false_chain(boxscore,'boxscore','playerByGameStats')
    if not check_pbgs or useProjectedLineup:
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
        },
      }
      message = 'using projected lineup'
      

    inputs = master_inputs(db=db,boxscore=boxscore,isProjectedLineup=isProjectedLineup)
    
    line_input_data = {}
    if inputs:
      for i in X_INPUTS:
        line_input_data[i] = inputs['data'][i]

    input_data.append(line_input_data)
    game_data.append({
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
    })
    extra_data.append({
      'isProjectedLineup': inputs['options']['projectedLineup'],
      'message': message,
    })
  data = pd.DataFrame(input_data)
  return data, game_data, extra_data

def nhl_test(db,boxscore,useProjectedLineup):
  return nhl_data(db=db,game=boxscore,useProjectedLineup=useProjectedLineup,test=True)