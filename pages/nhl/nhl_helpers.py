import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

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
from util.models import MODEL_PREDICT, MODEL_CONFIDENCE
# import h2o

def ai_return_dict_projectedLineup(db, data, prediction, confidence):
  data_keys = list(data['data']['data'].keys())
  confidence_data = {}
  predicted_data = {}

  for d in data_keys:
    confidence_data[d] = {}
    predicted_data[d] = {}
    confidence_data[d]['confidence_winner'] = int((np.max(confidence[d]['confidence_winner'], axis=1) * 100)[0])
    confidence_data[d]['confidence_winnerB'] = int((np.max(confidence[d]['confidence_winnerB'], axis=1) * 100)[0])
    confidence_data[d]['confidence_homeScore'] = int((np.max(confidence[d]['confidence_homeScore'], axis=1) * 100)[0])
    confidence_data[d]['confidence_awayScore'] = int((np.max(confidence[d]['confidence_awayScore'], axis=1) * 100)[0])
    confidence_data[d]['confidence_totalGoals'] = int((np.max(confidence[d]['confidence_totalGoals'], axis=1) * 100)[0])
    confidence_data[d]['confidence_goalDifferential'] = int((np.max(confidence[d]['confidence_goalDifferential'], axis=1) * 100)[0])


  # for d in range(0, len(data_keys)):
  #   confidence_data[data_keys[d]] = {}
  #   predicted_data[data_keys[d]] = {}
  #   if confidence[d] != -1:
  #     confidence_data[data_keys[d]]['confidence_winner'] = int((np.max(confidence[d][2], axis=1) * 100)[0])
  #     confidence_data[data_keys[d]]['confidence_homeScore'] = int((np.max(confidence[d][0], axis=1) * 100)[0])
  #     confidence_data[data_keys[d]]['confidence_awayScore'] = int((np.max(confidence[d][1], axis=1) * 100)[0])
  #     confidence_data[data_keys[d]]['confidence_totalGoals'] = int((np.max(confidence[d][3], axis=1) * 100)[0])
  #     confidence_data[data_keys[d]]['confidence_goalDifferential'] = int((np.max(confidence[d][4], axis=1) * 100)[0])
  #   else:
  #     confidence_data[data_keys[d]]['confidence_winner'] = -1
  #     confidence_data[data_keys[d]]['confidence_homeScore'] = -1
  #     confidence_data[data_keys[d]]['confidence_awayScore'] = -1
  #     confidence_data[data_keys[d]]['confidence_totalGoals'] = -1
  #     confidence_data[data_keys[d]]['confidence_goalDifferential'] = -1

  homeId = data['data']['home_team']['id']
  awayId = data['data']['away_team']['id']
  for d in data_keys:
    if len(data['data']['data'][d]) == 0 or len(prediction[d]) == 0:
      predicted_data[d]['prediction_winnerId'] = -1
      predicted_data[d]['prediction_winnerB'] = -1
      predicted_data[d]['prediction_homeScore'] = -1
      predicted_data[d]['prediction_awayScore'] = -1
      predicted_data[d]['prediction_totalGoals'] = -1
      predicted_data[d]['prediction_goalDifferential'] = -1
      state = 'OFF'
      homeTeam = data['data']['home_team']['city']
      awayTeam = data['data']['away_team']['city']
      predicted_data[d]['winner'] = -1
      predicted_data[d]['winnerB'] = -1
      predicted_data[d]['offset'] = -1
    else:
      winnerId = int(prediction[d]['prediction_winner'])
      winnerB = int(prediction[d]['prediction_winnerB'])
      predicted_data[d]['prediction_winnerId'] = winnerId
      predicted_data[d]['prediction_winnerB'] = winnerB
      predicted_data[d]['prediction_homeScore'] = int(prediction[d]['prediction_homeScore'])
      predicted_data[d]['prediction_awayScore'] = int(prediction[d]['prediction_awayScore'])
      predicted_data[d]['prediction_totalGoals'] = int(prediction[d]['prediction_totalGoals'])
      predicted_data[d]['prediction_goalDifferential'] = int(prediction[d]['prediction_goalDifferential'])
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
      predicted_data[d]['winnerB'] = homeTeam
    elif winnerB == 1:
      predicted_data[d]['winnerB'] = awayTeam
    else:
      predicted_data[d]['winnerB'] = 'Inconclusive'



  # for d in range(0, len(data_keys)):
  #   if len(data['data']['data'][data_keys[d]]) == 0 or len(prediction[d]) == 0:
  #     predicted_data[data_keys[d]]['prediction_winnerId'] = -1
  #     predicted_data[data_keys[d]]['prediction_homeScore'] = -1
  #     predicted_data[data_keys[d]]['prediction_awayScore'] = -1
  #     predicted_data[data_keys[d]]['prediction_totalGoals'] = -1
  #     predicted_data[data_keys[d]]['prediction_goalDifferential'] = -1
  #     state = 'OFF'
  #     homeTeam = data['data']['home_team']['city']
  #     awayTeam = data['data']['away_team']['city']
  #     predicted_data[data_keys[d]]['winner'] = -1
  #     predicted_data[data_keys[d]]['offset'] = -1
  #   else:
  #     winnerId = int(prediction[d][2])
  #     predicted_data[data_keys[d]]['prediction_winnerId'] = winnerId
  #     predicted_data[data_keys[d]]['prediction_homeScore'] = int(prediction[d][0])
  #     predicted_data[data_keys[d]]['prediction_awayScore'] = int(prediction[d][1])
  #     predicted_data[data_keys[d]]['prediction_totalGoals'] = int(prediction[d][3])
  #     predicted_data[data_keys[d]]['prediction_goalDifferential'] = int(prediction[d][4])
  #     state = data['data']['state']
  #     homeTeam = f"{data['data']['home_team']['city']} {data['data']['home_team']['name']}"
  #     awayTeam = f"{data['data']['away_team']['city']} {data['data']['away_team']['name']}"
  #     if abs(winnerId - homeId) < abs(winnerId - awayId):
  #       predicted_data[data_keys[d]]['winner'] = homeTeam
  #       predicted_data[data_keys[d]]['offset'] = abs(winnerId - homeId)
  #     elif abs(winnerId - homeId) > abs(winnerId - awayId):
  #       predicted_data[data_keys[d]]['winner'] = awayTeam
  #       predicted_data[data_keys[d]]['offset'] = abs(winnerId - awayId)
  #     else:
  #       predicted_data[data_keys[d]]['winner'] = 'Inconclusive'
  #       predicted_data[data_keys[d]]['offset'] = -1

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
      winningTeamB = homeTeam
    elif winnerB == 1:
      winningTeamB = awayTeam
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

def ai(db, game_data, useProjectedLineup, models):
  # data = nhl_ai(game_data)
  data = nhl_data(db, game_data, useProjectedLineup)
  if not data['isProjectedLineup'] and not useProjectedLineup:
    if len(data['data']['data'][0]) == 0:
      return ai_return_dict(data,[[]])
    prediction = MODEL_PREDICT(models,data)
    confidence = MODEL_CONFIDENCE(models,data)

    return ai_return_dict(data,prediction,confidence)

  else:
    data_keys = list(data['data']['data'].keys())
    predictions = {}
    confidences = {}
    # print(data['data'])
    for i in data_keys:
      predictions[i] = {}
      confidences[i] = {}
      # print(i,data['data']['data'][i])
      predictions[i][f'prediction_winner'] = models['model_winner'].predict(data['data']['data'][i])
      predictions[i][f'prediction_winnerB'] = models['model_winnerB'].predict(data['data']['data'][i])
      # predictions[i][f'prediction_winnerB'] = models['model_winnerB'].predict(h2o.H2OFrame(data['data']['data'][i]))
      predictions[i][f'prediction_homeScore'] = models['model_homeScore'].predict(data['data']['data'][i])
      predictions[i][f'prediction_awayScore'] = models['model_awayScore'].predict(data['data']['data'][i])
      predictions[i][f'prediction_totalGoals'] = models['model_totalGoals'].predict(data['data']['data'][i])
      predictions[i][f'prediction_goalDifferential'] = models['model_goalDifferential'].predict(data['data']['data'][i])
      confidences[i][f'confidence_winner'] = models['model_winner'].predict_proba(data['data']['data'][i])
      confidences[i][f'confidence_winnerB'] = models['model_winnerB'].predict_proba(data['data']['data'][i])
      confidences[i][f'confidence_homeScore'] = models['model_homeScore'].predict_proba(data['data']['data'][i])
      confidences[i][f'confidence_awayScore'] = models['model_awayScore'].predict_proba(data['data']['data'][i])
      confidences[i][f'confidence_totalGoals'] = models['model_totalGoals'].predict_proba(data['data']['data'][i])
      confidences[i][f'confidence_goalDifferential'] = models['model_goalDifferential'].predict_proba(data['data']['data'][i])
    
    return ai_return_dict_projectedLineup(db, data,predictions,confidences)
