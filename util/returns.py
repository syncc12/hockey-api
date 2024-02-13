import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from util.helpers import safe_chain
from constants.constants import FLIP_R, FLIP_B


import numpy as np

def safe_len(inList):
  try:
    if type(inList) is list:
      return len(inList)
    else:
      return 0
  except:
    return 0


def winnerOffset(winnerId, homeId, awayId, homeTeam, awayTeam):
  if abs(winnerId - homeId) < abs(winnerId - awayId):
    return homeTeam, abs(winnerId - homeId)
  elif abs(winnerId - homeId) > abs(winnerId - awayId):
    return awayTeam, abs(winnerId - awayId)
  else:
    return 'Inconclusive', -1

def ai_return_dict_projectedLineup(data, prediction, confidence):
  data_keys = list(data['data']['data'].keys())
  # print('data',data)
  # print('prediction',prediction)
  # print('data_keys',data_keys)
  confidence_data = {}
  predicted_data = {}

  for d in data_keys:
    confidence_data[d] = {}
    predicted_data[d] = {}
    confidence_data[d]['confidence_winner'] = int((np.max(confidence[d]['confidence_winner'], axis=1) * 100)[0])
    # confidence_data[d]['confidence_winnerB'] = int((np.max(confidence[d]['confidence_winnerB'], axis=1) * 100)[0])
    confidence_data[d]['confidence_winnerB'] = confidence[d]['confidence_winnerB']
    confidence_data[d]['confidence_homeScore'] = int((np.max(confidence[d]['confidence_homeScore'], axis=1) * 100)[0])
    confidence_data[d]['confidence_awayScore'] = int((np.max(confidence[d]['confidence_awayScore'], axis=1) * 100)[0])
    confidence_data[d]['confidence_totalGoals'] = int((np.max(confidence[d]['confidence_totalGoals'], axis=1) * 100)[0])
    confidence_data[d]['confidence_goalDifferential'] = int((np.max(confidence[d]['confidence_goalDifferential'], axis=1) * 100)[0])

  homeId = data['data']['home_team']['id']
  awayId = data['data']['away_team']['id']
  winner_list = []
  homeScore_list = []
  awayScore_list = []
  totalGoals_list = []
  goalDifferential_list = []
  for d in data_keys:
    # print(safe_len(safe_chain(data,'data',d)) == 0 or safe_len(safe_chain(prediction,d)) == 0, safe_len(safe_chain(data,'data',d)),safe_len(safe_chain(prediction,d)))
    # if safe_len(safe_chain(data,'data',d)) == 0 or safe_len(safe_chain(prediction,d)) == 0:
    #   predicted_data[d]['prediction_winnerId'] = -1
    #   predicted_data[d]['prediction_winnerB'] = -1
    #   predicted_data[d]['prediction_homeScore'] = -1
    #   predicted_data[d]['prediction_awayScore'] = -1
    #   predicted_data[d]['prediction_totalGoals'] = -1
    #   predicted_data[d]['prediction_goalDifferential'] = -1
    #   state = 'OFF'
    #   homeTeam = data['data']['home_team']['city']
    #   awayTeam = data['data']['away_team']['city']
    #   predicted_data[d]['winner'] = -1
    #   predicted_data[d]['winnerB'] = -1
    #   predicted_data[d]['offset'] = -1
    # else:
      winnerId = int(prediction[d]['prediction_winner'])
      winnerB = int(prediction[d]['prediction_winnerB'])
      homeScore = int(prediction[d]['prediction_homeScore'])
      awayScore = int(prediction[d]['prediction_awayScore'])
      totalGoals = int(prediction[d]['prediction_totalGoals'])
      goalDifferential = int(prediction[d]['prediction_goalDifferential'])
      homeScore_list.append(homeScore)
      awayScore_list.append(awayScore)
      totalGoals_list.append(totalGoals)
      goalDifferential_list.append(goalDifferential)
      winner_list.append(winnerId)
      predicted_data[d]['prediction_winnerId'] = winnerId
      predicted_data[d]['prediction_winnerB'] = winnerB
      predicted_data[d]['prediction_homeScore'] = homeScore
      predicted_data[d]['prediction_awayScore'] = awayScore
      predicted_data[d]['prediction_totalGoals'] = totalGoals
      predicted_data[d]['prediction_goalDifferential'] = goalDifferential
      state = data['data']['state']
      homeTeam = f"{data['data']['home_team']['city']} {data['data']['home_team']['name']}"
      awayTeam = f"{data['data']['away_team']['city']} {data['data']['away_team']['name']}"
      if abs(winnerId - homeId) < abs(winnerId - awayId):
        predicted_data[d]['winner'] = homeTeam
        predicted_data[d]['offset'] = abs(winnerId - homeId)
      elif abs(winnerId - homeId) > abs(winnerId - awayId):
        predicted_data[d]['winner'] = awayTeam
        predicted_data[d]['offset'] = abs(winnerId - awayId)
      else:
        predicted_data[d]['winner'] = 'Inconclusive'
        predicted_data[d]['offset'] = -1
      if winnerB == 0:
        predicted_data[d]['winnerB'] = awayTeam if FLIP_B else homeTeam
      elif winnerB == 1:
        predicted_data[d]['winnerB'] = homeTeam if FLIP_B else awayTeam
      else:
        predicted_data[d]['winnerB'] = 'Inconclusive'


  live_data = {}

  return {
    'gameId': data['data']['game_id'],
    'date': data['data']['date'],
    'state': state,
    'homeId': homeId,
    'awayId': awayId,
    'homeTeam': homeTeam,
    'awayTeam': awayTeam,
    'prediction': predicted_data,
    'confidence': confidence_data,
    'live': live_data,
    'message': data['message'],
  }

def ai_return_dict(data, prediction, confidence=-1):

  homeId = data['data']['home_team']['id']
  awayId = data['data']['away_team']['id']
  if len(data['data']['data'][0]) == 0 or len(prediction['prediction_winner']) == 0:
    winnerId = -1
    winnerB = -1
    homeScore = -1
    awayScore = -1
    totalGoals = -1
    goalDifferential = -1
    state = 'OFF'
    homeTeam = data['data']['home_team']['city']
    awayTeam = data['data']['away_team']['city']
    live_away = -1
    live_home = -1
    live_period = -1
    live_clock = -1
    live_stopped = -1
    live_intermission = -1
    live_leaderId = -1
    live_leader = -1
    winningTeam = -1
    offset = -1
  else:
    winnerId = int(prediction['prediction_winner'])
    winnerB = int(prediction['prediction_winnerB'])
    state = data['data']['state']
    homeTeam = f"{data['data']['home_team']['city']} {data['data']['home_team']['name']}"
    awayTeam = f"{data['data']['away_team']['city']} {data['data']['away_team']['name']}"
    live_away = data['data']['live']['away_score']
    live_home = data['data']['live']['home_score']
    live_period = data['data']['live']['period']
    live_clock = data['data']['live']['clock']
    live_stopped = data['data']['live']['stopped']
    live_intermission = data['data']['live']['intermission']
    if live_away > live_home:
      live_leaderId = awayId
      live_leader = awayTeam
    elif live_home > live_away:
      live_leaderId = homeId
      live_leader = homeTeam
    else:
      live_leaderId = -1
      live_leader = 'tied'
    if abs(winnerId - homeId) < abs(winnerId - awayId):
      winningTeam = homeTeam
      offset = abs(winnerId - homeId)
    elif abs(winnerId - homeId) > abs(winnerId - awayId):
      winningTeam = awayTeam
      offset = abs(winnerId - awayId)
    else:
      winningTeam = 'Inconclusive'
      offset = -1
    if winnerB == 0:
      winningTeamB = awayTeam if FLIP_B else homeTeam
    elif winnerB == 1:
      winningTeamB = homeTeam if FLIP_B else awayTeam
    else:
      winningTeamB = 'Inconclusive'

  if data['message'] == 'using projected lineup':
    live_data = {}
  else:
    live_data = {
      'away': live_away,
      'home': live_home,
      'period': live_period,
      'clock': live_clock,
      'stopped': live_stopped,
      'intermission': live_intermission,
      'leader': live_leader,
      'leaderId': live_leaderId,
    }
  
  predicted_data = {
    **prediction,
    'winner': winningTeam,
    'winnerB': winningTeamB,
    'offset': offset,
  }

  confidence_data = confidence

  out_data = {
    'gameId': data['data']['game_id'],
    'date': data['data']['date'],
    'state': state,
    'homeId': homeId,
    'awayId': awayId,
    'homeTeam': homeTeam,
    'awayTeam': awayTeam,
    'homeOdds': data['data']['home_team']['odds'],
    'awayOdds': data['data']['away_team']['odds'],
    'prediction': predicted_data,
    'confidence': confidence_data,
    'live': live_data,
    'message': data['message'],
  }
  return out_data
