import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

from flask import Flask, request, jsonify, Response
from joblib import load
import requests
from process import nhl_ai
from process2 import nhl_data, nhl_test
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from util.helpers import latestIDs, adjusted_winner, recommended_wagers
import boto3
import io
from inputs.inputs import master_inputs

def ai_return_dict_projectedLineup(data, prediction, confidence=[-1,-1,-1,-1]):
  data_keys = list(data['data']['data'].keys())
  predicted_data = {}

  for d in range(0, len(data_keys)):
    predicted_data[data_keys[d]] = {}
    if confidence[d] != -1:
      predicted_data[data_keys[d]]['winnerConfidence'] = int((np.max(confidence[d][2], axis=1) * 100)[0])
      predicted_data[data_keys[d]]['homeScoreConfidence'] = int((np.max(confidence[d][0], axis=1) * 100)[0])
      predicted_data[data_keys[d]]['awayScoreConfidence'] = int((np.max(confidence[d][1], axis=1) * 100)[0])
      predicted_data[data_keys[d]]['totalGoalsConfidence'] = int((np.max(confidence[d][3], axis=1) * 100)[0])
      predicted_data[data_keys[d]]['goalDifferentialConfidence'] = int((np.max(confidence[d][4], axis=1) * 100)[0])
    else:
      predicted_data[data_keys[d]]['winnerConfidence'] = -1
      predicted_data[data_keys[d]]['homeScoreConfidence'] = -1
      predicted_data[data_keys[d]]['awayScoreConfidence'] = -1
      predicted_data[data_keys[d]]['totalGoalsConfidence'] = -1
      predicted_data[data_keys[d]]['goalDifferentialConfidence'] = -1

  homeId = data['data']['home_team']['id']
  awayId = data['data']['away_team']['id']
  for d in range(0, len(data_keys)):
    if len(data['data']['data'][data_keys[d]]) == 0 or len(prediction[d]) == 0:
      predicted_data[data_keys[d]]['winnerId'] = -1
      predicted_data[data_keys[d]]['homeScore'] = -1
      predicted_data[data_keys[d]]['awayScore'] = -1
      predicted_data[data_keys[d]]['totalGoals'] = -1
      predicted_data[data_keys[d]]['goalDifferential'] = -1
      state = 'OFF'
      homeTeam = data['data']['home_team']['city']
      awayTeam = data['data']['away_team']['city']
      predicted_data[data_keys[d]]['winningTeam'] = -1
      predicted_data[data_keys[d]]['offset'] = -1
    else:
      winnerId = int(prediction[d][2])
      predicted_data[data_keys[d]]['winnerId'] = winnerId
      predicted_data[data_keys[d]]['homeScore'] = int(prediction[d][0])
      predicted_data[data_keys[d]]['awayScore'] = int(prediction[d][1])
      predicted_data[data_keys[d]]['totalGoals'] = int(prediction[d][3])
      predicted_data[data_keys[d]]['goalDifferential'] = int(prediction[d][4])
      state = data['data']['state']
      homeTeam = f"{data['data']['home_team']['city']} {data['data']['home_team']['name']}"
      awayTeam = f"{data['data']['away_team']['city']} {data['data']['away_team']['name']}"
      if abs(winnerId - homeId) < abs(winnerId - awayId):
        predicted_data[data_keys[d]]['winningTeam'] = homeTeam
        predicted_data[data_keys[d]]['offset'] = abs(winnerId - homeId)
      elif abs(winnerId - homeId) > abs(winnerId - awayId):
        predicted_data[data_keys[d]]['winningTeam'] = awayTeam
        predicted_data[data_keys[d]]['offset'] = abs(winnerId - awayId)
      else:
        predicted_data[data_keys[d]]['winningTeam'] = 'Inconclusive'
        predicted_data[data_keys[d]]['offset'] = -1

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
    'live': live_data,
    'message': data['message'],
  }

def ai_return_dict(data, prediction, confidence=-1):
  if confidence != -1:
    winnerConfidence = int((np.max(confidence[0][2], axis=1) * 100)[0])
    homeScoreConfidence = int((np.max(confidence[0][0], axis=1) * 100)[0])
    awayScoreConfidence = int((np.max(confidence[0][1], axis=1) * 100)[0])
    totalGoalsConfidence = int((np.max(confidence[0][3], axis=1) * 100)[0])
    goalDifferentialConfidence = int((np.max(confidence[0][4], axis=1) * 100)[0])
  else:
    winnerConfidence = -1
    homeScoreConfidence = -1
    awayScoreConfidence = -1
    totalGoalsConfidence = -1
    goalDifferentialConfidence = -1

  homeId = data['data']['home_team']['id']
  awayId = data['data']['away_team']['id']
  if len(data['data']['data'][0]) == 0 or len(prediction[0]) == 0:
    winnerId = -1
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
    winnerId = int(prediction[0][2])
    homeScore = int(prediction[0][0])
    awayScore = int(prediction[0][1])
    totalGoals = int(prediction[0][3])
    goalDifferential = int(prediction[0][4])
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
    'winnerId': winnerId,
    'winningTeam': winningTeam,
    'homeScore': homeScore,
    'awayScore': awayScore,
    'totalGoals': totalGoals,
    'goalDifferential': goalDifferential,
    'winnerConfidence': winnerConfidence,
    'homeScoreConfidence': homeScoreConfidence,
    'awayScoreConfidence': awayScoreConfidence,
    'totalGoalsConfidence': totalGoalsConfidence,
    'goalDifferentialConfidence': goalDifferentialConfidence,
    'offset': offset,
  }

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
    'live': live_data,
    'message': data['message'],
  }
  return out_data

def ai(db, game_data, **kwargs):
  # data = nhl_ai(game_data)
  data = nhl_data(db, game_data)
  if not data['isProjectedLineup']:
    if len(data['data']['data'][0]) == 0:
      return ai_return_dict(data,[[]])

    prediction_winner = kwargs['model_winner'].predict(data['data']['data'])
    prediction_homeScore = kwargs['model_homeScore'].predict(data['data']['data'])
    prediction_awayScore = kwargs['model_awayScore'].predict(data['data']['data'])
    prediction_totalGoals = kwargs['model_totalGoals'].predict(data['data']['data'])
    prediction_goalDifferential = kwargs['model_goalDifferential'].predict(data['data']['data'])
    confidence_winner = kwargs['model_winner'].predict_proba(data['data']['data'])
    confidence_homeScore = kwargs['model_homeScore'].predict_proba(data['data']['data'])
    confidence_awayScore = kwargs['model_awayScore'].predict_proba(data['data']['data'])
    confidence_totalGoals = kwargs['model_totalGoals'].predict_proba(data['data']['data'])
    confidence_goalDifferential = kwargs['model_goalDifferential'].predict_proba(data['data']['data'])
    prediction = [[prediction_homeScore,prediction_awayScore,prediction_winner,prediction_totalGoals,prediction_goalDifferential]]
    confidence = [[confidence_homeScore,confidence_awayScore,confidence_winner,confidence_totalGoals,confidence_goalDifferential]]

    return ai_return_dict(data,prediction,confidence)

  else:
    data_keys = list(data['data']['data'].keys())
    prediction_winner_0 = kwargs['model_winner'].predict(data['data']['data'][data_keys[0]])
    prediction_homeScore_0 = kwargs['model_homeScore'].predict(data['data']['data'][data_keys[0]])
    prediction_awayScore_0 = kwargs['model_awayScore'].predict(data['data']['data'][data_keys[0]])
    prediction_totalGoals_0 = kwargs['model_totalGoals'].predict(data['data']['data'][data_keys[0]])
    prediction_goalDifferential_0 = kwargs['model_goalDifferential'].predict(data['data']['data'][data_keys[0]])
    confidence_winner_0 = kwargs['model_winner'].predict_proba(data['data']['data'][data_keys[0]])
    confidence_homeScore_0 = kwargs['model_homeScore'].predict_proba(data['data']['data'][data_keys[0]])
    confidence_awayScore_0 = kwargs['model_awayScore'].predict_proba(data['data']['data'][data_keys[0]])
    confidence_totalGoals_0 = kwargs['model_totalGoals'].predict_proba(data['data']['data'][data_keys[0]])
    confidence_goalDifferential_0 = kwargs['model_goalDifferential'].predict_proba(data['data']['data'][data_keys[0]])
    
    prediction_winner_1 = kwargs['model_winner'].predict(data['data']['data'][data_keys[1]])
    prediction_homeScore_1 = kwargs['model_homeScore'].predict(data['data']['data'][data_keys[1]])
    prediction_awayScore_1 = kwargs['model_awayScore'].predict(data['data']['data'][data_keys[1]])
    prediction_totalGoals_1 = kwargs['model_totalGoals'].predict(data['data']['data'][data_keys[1]])
    prediction_goalDifferential_1 = kwargs['model_goalDifferential'].predict(data['data']['data'][data_keys[1]])
    confidence_winner_1 = kwargs['model_winner'].predict_proba(data['data']['data'][data_keys[1]])
    confidence_homeScore_1 = kwargs['model_homeScore'].predict_proba(data['data']['data'][data_keys[1]])
    confidence_awayScore_1 = kwargs['model_awayScore'].predict_proba(data['data']['data'][data_keys[1]])
    confidence_totalGoals_1 = kwargs['model_totalGoals'].predict_proba(data['data']['data'][data_keys[1]])
    confidence_goalDifferential_1 = kwargs['model_goalDifferential'].predict_proba(data['data']['data'][data_keys[1]])

    prediction_winner_2 = kwargs['model_winner'].predict(data['data']['data'][data_keys[2]])
    prediction_homeScore_2 = kwargs['model_homeScore'].predict(data['data']['data'][data_keys[2]])
    prediction_awayScore_2 = kwargs['model_awayScore'].predict(data['data']['data'][data_keys[2]])
    prediction_totalGoals_2 = kwargs['model_totalGoals'].predict(data['data']['data'][data_keys[2]])
    prediction_goalDifferential_2 = kwargs['model_goalDifferential'].predict(data['data']['data'][data_keys[2]])
    confidence_winner_2 = kwargs['model_winner'].predict_proba(data['data']['data'][data_keys[2]])
    confidence_homeScore_2 = kwargs['model_homeScore'].predict_proba(data['data']['data'][data_keys[2]])
    confidence_awayScore_2 = kwargs['model_awayScore'].predict_proba(data['data']['data'][data_keys[2]])
    confidence_totalGoals_2 = kwargs['model_totalGoals'].predict_proba(data['data']['data'][data_keys[2]])
    confidence_goalDifferential_2 = kwargs['model_goalDifferential'].predict_proba(data['data']['data'][data_keys[2]])

    prediction_winner_3 = kwargs['model_winner'].predict(data['data']['data'][data_keys[3]])
    prediction_homeScore_3 = kwargs['model_homeScore'].predict(data['data']['data'][data_keys[3]])
    prediction_awayScore_3 = kwargs['model_awayScore'].predict(data['data']['data'][data_keys[3]])
    prediction_totalGoals_3 = kwargs['model_totalGoals'].predict(data['data']['data'][data_keys[3]])
    prediction_goalDifferential_3 = kwargs['model_goalDifferential'].predict(data['data']['data'][data_keys[3]])
    confidence_winner_3 = kwargs['model_winner'].predict_proba(data['data']['data'][data_keys[3]])
    confidence_homeScore_3 = kwargs['model_homeScore'].predict_proba(data['data']['data'][data_keys[3]])
    confidence_awayScore_3 = kwargs['model_awayScore'].predict_proba(data['data']['data'][data_keys[3]])
    confidence_totalGoals_3 = kwargs['model_totalGoals'].predict_proba(data['data']['data'][data_keys[3]])
    confidence_goalDifferential_3 = kwargs['model_goalDifferential'].predict_proba(data['data']['data'][data_keys[3]])

    prediction = [
      [prediction_homeScore_0,prediction_awayScore_0,prediction_winner_0,prediction_totalGoals_0,prediction_goalDifferential_0],
      [prediction_homeScore_1,prediction_awayScore_1,prediction_winner_1,prediction_totalGoals_1,prediction_goalDifferential_1],
      [prediction_homeScore_2,prediction_awayScore_2,prediction_winner_2,prediction_totalGoals_2,prediction_goalDifferential_2],
      [prediction_homeScore_3,prediction_awayScore_3,prediction_winner_3,prediction_totalGoals_3,prediction_goalDifferential_3],
    ]
    confidence = [
      [confidence_homeScore_0,confidence_awayScore_0,confidence_winner_0,confidence_totalGoals_0,confidence_goalDifferential_0],
      [confidence_homeScore_1,confidence_awayScore_1,confidence_winner_1,confidence_totalGoals_1,confidence_goalDifferential_1],
      [confidence_homeScore_2,confidence_awayScore_2,confidence_winner_2,confidence_totalGoals_2,confidence_goalDifferential_2],
      [confidence_homeScore_3,confidence_awayScore_3,confidence_winner_3,confidence_totalGoals_3,confidence_goalDifferential_3],
    ]
    
    return ai_return_dict_projectedLineup(data,prediction,confidence)
