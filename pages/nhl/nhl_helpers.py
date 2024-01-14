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

def ai_return_dict_projectedLineup(db, data, prediction, confidence=[-1,-1,-1,-1]):
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
  # if confidence != -1:
  #   winnerConfidence = int((np.max(confidence[0][2], axis=1) * 100)[0])
  #   homeScoreConfidence = int((np.max(confidence[0][0], axis=1) * 100)[0])
  #   awayScoreConfidence = int((np.max(confidence[0][1], axis=1) * 100)[0])
  #   totalGoalsConfidence = int((np.max(confidence[0][3], axis=1) * 100)[0])
  #   goalDifferentialConfidence = int((np.max(confidence[0][4], axis=1) * 100)[0])
  #   finalPeriodConfidence = int((np.max(confidence[0][5], axis=1) * 100)[0])
  #   pastRegulationConfidence = int((np.max(confidence[0][6], axis=1) * 100)[0])
  #   awayShotsConfidence = int((np.max(confidence[0][7], axis=1) * 100)[0])
  #   homeShotsConfidence = int((np.max(confidence[0][8], axis=1) * 100)[0])
  #   awayShotsPeriod1Confidence = int((np.max(confidence[0][9], axis=1) * 100)[0])
  #   homeShotsPeriod1Confidence = int((np.max(confidence[0][10], axis=1) * 100)[0])
  #   awayShotsPeriod2Confidence = int((np.max(confidence[0][11], axis=1) * 100)[0])
  #   homeShotsPeriod2Confidence = int((np.max(confidence[0][12], axis=1) * 100)[0])
  #   awayShotsPeriod3Confidence = int((np.max(confidence[0][13], axis=1) * 100)[0])
  #   homeShotsPeriod3Confidence = int((np.max(confidence[0][14], axis=1) * 100)[0])
  #   awayShotsPeriod4Confidence = int((np.max(confidence[0][15], axis=1) * 100)[0])
  #   homeShotsPeriod4Confidence = int((np.max(confidence[0][16], axis=1) * 100)[0])
  #   awayShotsPeriod5Confidence = int((np.max(confidence[0][17], axis=1) * 100)[0])
  #   homeShotsPeriod5Confidence = int((np.max(confidence[0][18], axis=1) * 100)[0])
  #   awayScorePeriod1Confidence = int((np.max(confidence[0][19], axis=1) * 100)[0])
  #   homeScorePeriod1Confidence = int((np.max(confidence[0][20], axis=1) * 100)[0])
  #   awayScorePeriod2Confidence = int((np.max(confidence[0][21], axis=1) * 100)[0])
  #   homeScorePeriod2Confidence = int((np.max(confidence[0][22], axis=1) * 100)[0])
  #   awayScorePeriod3Confidence = int((np.max(confidence[0][23], axis=1) * 100)[0])
  #   homeScorePeriod3Confidence = int((np.max(confidence[0][24], axis=1) * 100)[0])
  #   awayScorePeriod4Confidence = int((np.max(confidence[0][25], axis=1) * 100)[0])
  #   homeScorePeriod4Confidence = int((np.max(confidence[0][26], axis=1) * 100)[0])
  #   awayScorePeriod5Confidence = int((np.max(confidence[0][27], axis=1) * 100)[0])
  #   homeScorePeriod5Confidence = int((np.max(confidence[0][28], axis=1) * 100)[0])
  #   period1PuckLineConfidence = int((np.max(confidence[0][29], axis=1) * 100)[0])
  #   period2PuckLineConfidence = int((np.max(confidence[0][30], axis=1) * 100)[0])
  #   period3PuckLineConfidence = int((np.max(confidence[0][31], axis=1) * 100)[0])
  # else:
  #   winnerConfidence = -1
  #   homeScoreConfidence = -1
  #   awayScoreConfidence = -1
  #   totalGoalsConfidence = -1
  #   goalDifferentialConfidence = -1
  #   finalPeriod = -1
  #   pastRegulationConfidence = -1
  #   awayShots = -1
  #   homeShots = -1
  #   awayShotsPeriod1 = -1
  #   homeShotsPeriod1 = -1
  #   awayShotsPeriod2 = -1
  #   homeShotsPeriod2 = -1
  #   awayShotsPeriod3 = -1
  #   homeShotsPeriod3 = -1
  #   awayShotsPeriod4 = -1
  #   homeShotsPeriod4 = -1
  #   awayShotsPeriod5 = -1
  #   homeShotsPeriod5 = -1
  #   awayScorePeriod1 = -1
  #   homeScorePeriod1 = -1
  #   awayScorePeriod2 = -1
  #   homeScorePeriod2 = -1
  #   awayScorePeriod3 = -1
  #   homeScorePeriod3 = -1
  #   awayScorePeriod4 = -1
  #   homeScorePeriod4 = -1
  #   awayScorePeriod5 = -1
  #   homeScorePeriod5 = -1
  #   period1PuckLine = -1
  #   period2PuckLine = -1
  #   period3PuckLine = -1

  homeId = data['data']['home_team']['id']
  awayId = data['data']['away_team']['id']
  if len(data['data']['data'][0]) == 0 or len(prediction['prediction_winner']) == 0:
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
    winnerId = int(prediction['prediction_winner'])
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
    **prediction,
    'winner': winningTeam,
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

def ai(db, game_data, models):
  # data = nhl_ai(game_data)
  data = nhl_data(db, game_data)
  if not data['isProjectedLineup']:
    if len(data['data']['data'][0]) == 0:
      return ai_return_dict(data,[[]])
    prediction_data = MODEL_PREDICT(models,data)
    # prediction_winner = models['model_winner'].predict(data['data']['data'])
    # prediction_homeScore = models['model_homeScore'].predict(data['data']['data'])
    # prediction_awayScore = models['model_awayScore'].predict(data['data']['data'])
    # prediction_totalGoals = models['model_totalGoals'].predict(data['data']['data'])
    # prediction_goalDifferential = models['model_goalDifferential'].predict(data['data']['data'])
    # prediction_finalPeriod = models['model_finalPeriod'].predict(data['data']['data'])
    # prediction_pastRegulation = models['model_pastRegulation'].predict(data['data']['data'])
    # prediction_awayShots = models['model_awayShots'].predict(data['data']['data'])
    # prediction_homeShots = models['model_homeShots'].predict(data['data']['data'])
    # prediction_awayShotsPeriod1 = models['model_awayShotsPeriod1'].predict(data['data']['data'])
    # prediction_homeShotsPeriod1 = models['model_homeShotsPeriod1'].predict(data['data']['data'])
    # prediction_awayShotsPeriod2 = models['model_awayShotsPeriod2'].predict(data['data']['data'])
    # prediction_homeShotsPeriod2 = models['model_homeShotsPeriod2'].predict(data['data']['data'])
    # prediction_awayShotsPeriod3 = models['model_awayShotsPeriod3'].predict(data['data']['data'])
    # prediction_homeShotsPeriod3 = models['model_homeShotsPeriod3'].predict(data['data']['data'])
    # prediction_awayShotsPeriod4 = models['model_awayShotsPeriod4'].predict(data['data']['data'])
    # prediction_homeShotsPeriod4 = models['model_homeShotsPeriod4'].predict(data['data']['data'])
    # prediction_awayShotsPeriod5 = models['model_awayShotsPeriod5'].predict(data['data']['data'])
    # prediction_homeShotsPeriod5 = models['model_homeShotsPeriod5'].predict(data['data']['data'])
    # prediction_awayScorePeriod1 = models['model_awayScorePeriod1'].predict(data['data']['data'])
    # prediction_homeScorePeriod1 = models['model_homeScorePeriod1'].predict(data['data']['data'])
    # prediction_awayScorePeriod2 = models['model_awayScorePeriod2'].predict(data['data']['data'])
    # prediction_homeScorePeriod2 = models['model_homeScorePeriod2'].predict(data['data']['data'])
    # prediction_awayScorePeriod3 = models['model_awayScorePeriod3'].predict(data['data']['data'])
    # prediction_homeScorePeriod3 = models['model_homeScorePeriod3'].predict(data['data']['data'])
    # prediction_awayScorePeriod4 = models['model_awayScorePeriod4'].predict(data['data']['data'])
    # prediction_homeScorePeriod4 = models['model_homeScorePeriod4'].predict(data['data']['data'])
    # prediction_awayScorePeriod5 = models['model_awayScorePeriod5'].predict(data['data']['data'])
    # prediction_homeScorePeriod5 = models['model_homeScorePeriod5'].predict(data['data']['data'])
    # prediction_period1PuckLine = models['model_period1PuckLine'].predict(data['data']['data'])
    # prediction_period2PuckLine = models['model_period2PuckLine'].predict(data['data']['data'])
    # prediction_period3PuckLine = models['model_period3PuckLine'].predict(data['data']['data'])
    confidence_data = MODEL_CONFIDENCE(models,data)
    # confidence_winner = models['model_winner'].predict_proba(data['data']['data'])
    # confidence_homeScore = models['model_homeScore'].predict_proba(data['data']['data'])
    # confidence_awayScore = models['model_awayScore'].predict_proba(data['data']['data'])
    # confidence_totalGoals = models['model_totalGoals'].predict_proba(data['data']['data'])
    # confidence_goalDifferential = models['model_goalDifferential'].predict_proba(data['data']['data'])
    # confidence_finalPeriod = models['model_finalPeriod'].predict_proba(data['data']['data'])
    # confidence_pastRegulation = models['model_pastRegulation'].predict_proba(data['data']['data'])
    # confidence_awayShots = models['model_awayShots'].predict_proba(data['data']['data'])
    # confidence_homeShots = models['model_homeShots'].predict_proba(data['data']['data'])
    # confidence_awayShotsPeriod1 = models['model_awayShotsPeriod1'].predict_proba(data['data']['data'])
    # confidence_homeShotsPeriod1 = models['model_homeShotsPeriod1'].predict_proba(data['data']['data'])
    # confidence_awayShotsPeriod2 = models['model_awayShotsPeriod2'].predict_proba(data['data']['data'])
    # confidence_homeShotsPeriod2 = models['model_homeShotsPeriod2'].predict_proba(data['data']['data'])
    # confidence_awayShotsPeriod3 = models['model_awayShotsPeriod3'].predict_proba(data['data']['data'])
    # confidence_homeShotsPeriod3 = models['model_homeShotsPeriod3'].predict_proba(data['data']['data'])
    # confidence_awayShotsPeriod4 = models['model_awayShotsPeriod4'].predict_proba(data['data']['data'])
    # confidence_homeShotsPeriod4 = models['model_homeShotsPeriod4'].predict_proba(data['data']['data'])
    # confidence_awayShotsPeriod5 = models['model_awayShotsPeriod5'].predict_proba(data['data']['data'])
    # confidence_homeShotsPeriod5 = models['model_homeShotsPeriod5'].predict_proba(data['data']['data'])
    # confidence_awayScorePeriod1 = models['model_awayScorePeriod1'].predict_proba(data['data']['data'])
    # confidence_homeScorePeriod1 = models['model_homeScorePeriod1'].predict_proba(data['data']['data'])
    # confidence_awayScorePeriod2 = models['model_awayScorePeriod2'].predict_proba(data['data']['data'])
    # confidence_homeScorePeriod2 = models['model_homeScorePeriod2'].predict_proba(data['data']['data'])
    # confidence_awayScorePeriod3 = models['model_awayScorePeriod3'].predict_proba(data['data']['data'])
    # confidence_homeScorePeriod3 = models['model_homeScorePeriod3'].predict_proba(data['data']['data'])
    # confidence_awayScorePeriod4 = models['model_awayScorePeriod4'].predict_proba(data['data']['data'])
    # confidence_homeScorePeriod4 = models['model_homeScorePeriod4'].predict_proba(data['data']['data'])
    # confidence_awayScorePeriod5 = models['model_awayScorePeriod5'].predict_proba(data['data']['data'])
    # confidence_homeScorePeriod5 = models['model_homeScorePeriod5'].predict_proba(data['data']['data'])
    # confidence_period1PuckLine = models['model_period1PuckLine'].predict_proba(data['data']['data'])
    # confidence_period2PuckLine = models['model_period2PuckLine'].predict_proba(data['data']['data'])
    # confidence_period3PuckLine = models['model_period3PuckLine'].predict_proba(data['data']['data'])
    
    prediction = prediction_data
    confidence = confidence_data

    return ai_return_dict(data,prediction,confidence)

  else:
    data_keys = list(data['data']['data'].keys())
    prediction_winner_0 = models['model_winner'].predict(data['data']['data'][data_keys[0]])
    prediction_homeScore_0 = models['model_homeScore'].predict(data['data']['data'][data_keys[0]])
    prediction_awayScore_0 = models['model_awayScore'].predict(data['data']['data'][data_keys[0]])
    prediction_totalGoals_0 = models['model_totalGoals'].predict(data['data']['data'][data_keys[0]])
    prediction_goalDifferential_0 = models['model_goalDifferential'].predict(data['data']['data'][data_keys[0]])
    confidence_winner_0 = models['model_winner'].predict_proba(data['data']['data'][data_keys[0]])
    confidence_homeScore_0 = models['model_homeScore'].predict_proba(data['data']['data'][data_keys[0]])
    confidence_awayScore_0 = models['model_awayScore'].predict_proba(data['data']['data'][data_keys[0]])
    confidence_totalGoals_0 = models['model_totalGoals'].predict_proba(data['data']['data'][data_keys[0]])
    confidence_goalDifferential_0 = models['model_goalDifferential'].predict_proba(data['data']['data'][data_keys[0]])
    
    prediction_winner_1 = models['model_winner'].predict(data['data']['data'][data_keys[1]])
    prediction_homeScore_1 = models['model_homeScore'].predict(data['data']['data'][data_keys[1]])
    prediction_awayScore_1 = models['model_awayScore'].predict(data['data']['data'][data_keys[1]])
    prediction_totalGoals_1 = models['model_totalGoals'].predict(data['data']['data'][data_keys[1]])
    prediction_goalDifferential_1 = models['model_goalDifferential'].predict(data['data']['data'][data_keys[1]])
    confidence_winner_1 = models['model_winner'].predict_proba(data['data']['data'][data_keys[1]])
    confidence_homeScore_1 = models['model_homeScore'].predict_proba(data['data']['data'][data_keys[1]])
    confidence_awayScore_1 = models['model_awayScore'].predict_proba(data['data']['data'][data_keys[1]])
    confidence_totalGoals_1 = models['model_totalGoals'].predict_proba(data['data']['data'][data_keys[1]])
    confidence_goalDifferential_1 = models['model_goalDifferential'].predict_proba(data['data']['data'][data_keys[1]])

    prediction_winner_2 = models['model_winner'].predict(data['data']['data'][data_keys[2]])
    prediction_homeScore_2 = models['model_homeScore'].predict(data['data']['data'][data_keys[2]])
    prediction_awayScore_2 = models['model_awayScore'].predict(data['data']['data'][data_keys[2]])
    prediction_totalGoals_2 = models['model_totalGoals'].predict(data['data']['data'][data_keys[2]])
    prediction_goalDifferential_2 = models['model_goalDifferential'].predict(data['data']['data'][data_keys[2]])
    confidence_winner_2 = models['model_winner'].predict_proba(data['data']['data'][data_keys[2]])
    confidence_homeScore_2 = models['model_homeScore'].predict_proba(data['data']['data'][data_keys[2]])
    confidence_awayScore_2 = models['model_awayScore'].predict_proba(data['data']['data'][data_keys[2]])
    confidence_totalGoals_2 = models['model_totalGoals'].predict_proba(data['data']['data'][data_keys[2]])
    confidence_goalDifferential_2 = models['model_goalDifferential'].predict_proba(data['data']['data'][data_keys[2]])

    prediction_winner_3 = models['model_winner'].predict(data['data']['data'][data_keys[3]])
    prediction_homeScore_3 = models['model_homeScore'].predict(data['data']['data'][data_keys[3]])
    prediction_awayScore_3 = models['model_awayScore'].predict(data['data']['data'][data_keys[3]])
    prediction_totalGoals_3 = models['model_totalGoals'].predict(data['data']['data'][data_keys[3]])
    prediction_goalDifferential_3 = models['model_goalDifferential'].predict(data['data']['data'][data_keys[3]])
    confidence_winner_3 = models['model_winner'].predict_proba(data['data']['data'][data_keys[3]])
    confidence_homeScore_3 = models['model_homeScore'].predict_proba(data['data']['data'][data_keys[3]])
    confidence_awayScore_3 = models['model_awayScore'].predict_proba(data['data']['data'][data_keys[3]])
    confidence_totalGoals_3 = models['model_totalGoals'].predict_proba(data['data']['data'][data_keys[3]])
    confidence_goalDifferential_3 = models['model_goalDifferential'].predict_proba(data['data']['data'][data_keys[3]])

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
    
    return ai_return_dict_projectedLineup(db, data,prediction,confidence)
