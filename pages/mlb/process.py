import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

import requests
import pandas as pd
from util.helpers import safe_chain
from pages.mlb.inputs import base_inputs, X_INPUTS_MLB
from pages.mlb.models import PREDICT_SCORE_H2H

REPLACE_VALUE = -1

def mlb_data(games,no_df=False):
  input_data = []
  game_data = []
  extra_data = []
  for i in range(0,len(games)):
    game = games[i]
    message = ''

    game = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game['gamePk']}/feed/live").json()
      
    inputs, isProjectedLineup = base_inputs(game,prediction=True)
    if isProjectedLineup:
      message='Projected Lineup'
    
    line_input_data = {}
    if inputs:
      for i in X_INPUTS_MLB:
        line_input_data[i] = inputs[i]

    input_data.append(line_input_data)
    game_data.append({
      'game_id': safe_chain(game,'gamePk'),
      'date': safe_chain(game,'gameData','datetime','originalDate'),
      'state': safe_chain(game,'gameData','status','detailedState'),
      'home_team': {
        'abbreviation': safe_chain(game,'gameData','teams','home','abbreviation'),
      },
      'away_team': {
        'abbreviation':  safe_chain(game,'gameData','teams','away','abbreviation'),
      },
    })
    extra_data.append({
      'isProjectedLineup': isProjectedLineup,
      'message': message,
    })
  if no_df:
    data = input_data
  else:
    data = pd.DataFrame(input_data)
  return data, game_data, extra_data

def ai_teams(games, wModels, simple=False, receipt=False):
  all_games = []
  data, game_data, extra_data = mlb_data(games=games, no_df=True)
  predictions,confidences = PREDICT_SCORE_H2H(data, wModels, simple_return=True)
    
  if simple:
    for i, prediction in enumerate(predictions):
      awayTeam = game_data[i]["away_team"]["name"]
      homeTeam = game_data[i]["home_team"]["name"]
      winner = homeTeam if prediction == 0 else awayTeam
      all_games.append({
        'awayTeam': awayTeam,
        'homeTeam': homeTeam,
        'winningTeam': f"{winner} - {(confidences[i]*100):.2f}%",
      })
    return all_games
  elif receipt:
    for i in range(len(predictions)):
      all_games.append(f'{"p-" if extra_data[i]["isProjectedLineup"] else ""}{game_data[i]["home_team"]["abbreviation"] if predictions[i] == 0 else game_data[i]["away_team"]["abbreviation"]} {round(confidences[i]*100)}%')
    return all_games
  else:
    return {
      'predictions':predictions,
      'confidences':confidences,
    }
