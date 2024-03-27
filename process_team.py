import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

import requests
import pandas as pd
from util.helpers import safe_chain, false_chain, n2n, isNaN, getAge, getPlayer, getPlayerData, projectedLineup, projectedRoster
from inputs.inputs import master_inputs
# from util.query import get_last_game_team_stats
from constants.inputConstants import X_INPUTS_S, Y_OUTPUTS, X_INPUTS_T

MODEL_NAMES = Y_OUTPUTS

REPLACE_VALUE = -1
    

def nhl_data_team(db,games,useProjectedLineups=[],useProjectedRosters=[],messages=[''],test=False,no_df=False):
  input_data = []
  game_data = []
  extra_data = []
  for i in range(0,len(games)):
    game = games[i]
    useProjectedLineup = useProjectedLineups[i]
    isProjectedLineup = useProjectedLineup
    useProjectedRoster = useProjectedRosters[i]
    isProjectedRoster = useProjectedRoster
    message = ''

    boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game['id']}/boxscore").json()
    check_pbgs = false_chain(boxscore,'playerByGameStats')
    if not check_pbgs or useProjectedLineup or useProjectedRoster:
      if useProjectedRoster:
        isProjectedRoster = True
        awayRoster, homeRoster, landing = projectedRoster(db,game['id'])

        boxscore['playerByGameStats'] = {
          'awayTeam': awayRoster,
          'homeTeam': homeRoster,
        }
        boxscore['boxscore']['gameInfo'] = {}
        if false_chain(landing,'matchup'):
          boxscore['boxscore']['gameInfo']['awayTeam'] = landing['matchup']['gameInfo']['awayTeam']
          boxscore['boxscore']['gameInfo']['homeTeam'] = landing['matchup']['gameInfo']['homeTeam']
        else:
          away_last_boxscore, away_home_away = projectedLineup(boxscore['awayTeam']['abbrev'],game['id'],last_boxscore=True)
          home_last_boxscore, home_home_away = projectedLineup(boxscore['homeTeam']['abbrev'],game['id'],last_boxscore=True)  
          boxscore['boxscore']['gameInfo']['awayTeam'] = away_last_boxscore['boxscore']['gameInfo'][away_home_away]
          boxscore['boxscore']['gameInfo']['homeTeam'] = home_last_boxscore['boxscore']['gameInfo'][home_home_away]
        message = 'using projected lineup'

      else:
        isProjectedLineup = True
        away_last_boxscore, away_home_away = projectedLineup(boxscore['awayTeam']['abbrev'],game['id'], last_boxscore=True)
        home_last_boxscore, home_home_away = projectedLineup(boxscore['homeTeam']['abbrev'],game['id'], last_boxscore=True)
        
        boxscore['playerByGameStats'] = {
          'awayTeam': away_last_boxscore['playerByGameStats'][away_home_away],
          'homeTeam': home_last_boxscore['playerByGameStats'][home_home_away],
        },
        boxscore['summary'] = {
          'gameInfo': {
            'awayTeam': away_last_boxscore['summary']['gameInfo'][away_home_away],
            'homeTeam': home_last_boxscore['summary']['gameInfo'][home_home_away],
          },
        }
        message = 'using projected lineup'
      
    inputs = master_inputs(db=db,boxscore=boxscore,isProjectedLineup=isProjectedLineup,isProjectedRoster=isProjectedRoster)
    
    line_input_data = {}
    if inputs:
      for i in X_INPUTS_S:
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
        'period': safe_chain(boxscore,'periodDescriptor','number'),
        'clock': safe_chain(boxscore,'clock','timeRemaining'),
        'stopped': not safe_chain(boxscore,'clock','running'),
        'intermission': safe_chain(boxscore,'clock','inIntermission'),
      },
    })
    extra_data.append({
      'isProjectedLineup': inputs['options']['projectedLineup'],
      'message': message,
    })
  if no_df:
    data = input_data
  else:
    data = pd.DataFrame(input_data)
  return data, game_data, extra_data
