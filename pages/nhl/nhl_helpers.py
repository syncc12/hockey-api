import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from flask import Flask, request, jsonify, Response
from joblib import load
import requests
from process2 import nhl_data, nhl_test, nhl_data2
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from util.helpers import latestIDs, adjusted_winner, recommended_wagers
import boto3
import io
from inputs.inputs import master_inputs
from util.models import MODEL_PREDICT, MODEL_CONFIDENCE, MODEL_BATCH_PREDICT, MODEL_BATCH_CONFIDENCE, MODEL_PREDICT_CONFIDENCE_WINNER_B
from util.team_models import PREDICT_SCORE_H2H, PREDICT_H2H, PREDICT_SPREAD, PREDICT_SCORE_SPREAD, PREDICT_COVERS, PREDICT_SCORE_COVERS, PREDICT_LGBM_H2H, PREDICT_LGBM_SCORE_H2H
from util.returns import ai_return_dict_projectedLineup, ai_return_dict, ai_return_dict2
from train_torch import predict_model
import xgboost as xgb
from constants.inputConstants import X_INPUTS_T

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
      winnerB_input = xgb.DMatrix(pd.DataFrame(data['data']['input_data'][i],index=[0]))
      winnerB_probability = models['model_winnerB'].predict(winnerB_input)
      winnerB_prediction = [1 if i > 0.5 else 0 for i in winnerB_probability]
      predictions[i] = {}
      confidences[i] = {}
      # print(i,data['data']['data'][i])
      predictions[i][f'prediction_winner'] = models['model_winner'].predict(data['data']['data'][i])
      predictions[i][f'prediction_winnerB'] = winnerB_prediction[0]
      # predictions[i][f'prediction_winnerB'] = predict_model(data['data']['data'][i])
      # predictions[i][f'prediction_winnerB'] = models['model_winnerB'].predict(h2o.H2OFrame(data['data']['data'][i]))
      predictions[i][f'prediction_homeScore'] = models['model_homeScore'].predict(data['data']['data'][i])
      predictions[i][f'prediction_awayScore'] = models['model_awayScore'].predict(data['data']['data'][i])
      predictions[i][f'prediction_totalGoals'] = models['model_totalGoals'].predict(data['data']['data'][i])
      predictions[i][f'prediction_goalDifferential'] = models['model_goalDifferential'].predict(data['data']['data'][i])
      confidences[i][f'confidence_winner'] = models['model_winner'].predict_proba(data['data']['data'][i])
      confidences[i][f'confidence_winnerB'] = round(winnerB_probability[0] * 100)
      # confidences[i][f'confidence_winnerB'] = models['model_winnerB'].predict_proba(data['data']['data'][i])
      confidences[i][f'confidence_homeScore'] = models['model_homeScore'].predict_proba(data['data']['data'][i])
      confidences[i][f'confidence_awayScore'] = models['model_awayScore'].predict_proba(data['data']['data'][i])
      confidences[i][f'confidence_totalGoals'] = models['model_totalGoals'].predict_proba(data['data']['data'][i])
      confidences[i][f'confidence_goalDifferential'] = models['model_goalDifferential'].predict_proba(data['data']['data'][i])
    
    return ai_return_dict_projectedLineup(data,predictions,confidences)

def ai2(db, games, projectedLineups, models):
  data, game_data, extra_data = nhl_data2(db=db, games=games, useProjectedLineups=projectedLineups)
  predictions = MODEL_BATCH_PREDICT(models,data)
  confidences = MODEL_BATCH_CONFIDENCE(models,data)
  return ai_return_dict2(game_data,extra_data,predictions,confidences)

def ai_receipt(db, games, projectedLineups, models):
  data, game_data, extra_data = nhl_data2(db=db, games=games, useProjectedLineups=projectedLineups)
  predictions, confidences = MODEL_PREDICT_CONFIDENCE_WINNER_B(models,data)
  receipt = []
  for i in range(len(predictions)):
    receipt.append(f'{"p-" if extra_data[i]["isProjectedLineup"] else ""}{game_data[i]["home_team"]["name"] if predictions[i] == 0 else game_data[i]["away_team"]["name"]} {confidences[i]}%')
  return receipt

def ai_teams(db, games, projectedLineups, models, useModel, projectedRosters, simple=False, receipt=False, vote='hard'):
  if useModel == 'xgb':
    wModels = models['wModels']
    lModels = models['lModels']
  elif useModel == 'lgbm':
    wModels = models['wModelsLGBM']
    lModels = None
  # cModels = models['cModels']
  # sModels = models['sModels']
  all_games = []
  data, game_data, extra_data = nhl_data2(db=db, games=games, useProjectedLineups=projectedLineups, useProjectedRosters=projectedRosters, no_df=True)
  if vote == 'soft':
    if useModel == 'xgb':
      predictions,confidences = PREDICT_H2H(data, wModels, lModels, simple_return=True)
    elif useModel == 'lgbm':
      predictions,confidences = PREDICT_LGBM_H2H(data, wModels, simple_return=True)
    # spread_predictions,spread_confidences = PREDICT_SPREAD(data, sModels,simple_return=True)
    # covers_predictions,covers_confidences = PREDICT_COVERS(data, cModels,simple_return=True)
  else:
    if useModel == 'xgb':
      predictions,confidences = PREDICT_SCORE_H2H(data, wModels, lModels, simple_return=True)
    elif useModel == 'lgbm':
      predictions,confidences = PREDICT_LGBM_SCORE_H2H(data, wModels, simple_return=True)
    # spread_predictions,spread_confidences = PREDICT_SCORE_SPREAD(data, sModels,simple_return=True)
    # covers_predictions,covers_confidences = PREDICT_SCORE_COVERS(data, cModels,simple_return=True)
  if simple:
    for i, prediction in enumerate(predictions):
      awayTeam = game_data[i]["away_team"]["name"]
      homeTeam = game_data[i]["home_team"]["name"]
      winner = homeTeam if prediction == 0 else awayTeam
      all_games.append({
        'awayTeam': awayTeam,
        'homeTeam': homeTeam,
        'winningTeamB': f"{winner} - {(confidences[i]*100):.2f}%",
        # 'spread': f'{spread_predictions[i]} - {(spread_confidences[i]*100):.2f}%',
        # 'covers': f'{covers_predictions[i]} - {(covers_confidences[i]*100):.2f}%',
        # 'crosscheck': {
        #   'awayWin': f"{w_predictions_away[i]} - {(w_probabilities_away[i]*100):.2f}%",
        #   'awayLoss': f"{l_predictions_away[i]} - {(l_probabilities_away[i]*100):.2f}%",
        #   'homeWin': f"{w_predictions_home[i]} - {(w_probabilities_home[i]*100):.2f}%",
        #   'homeLoss': f"{l_predictions_home[i]} - {(l_probabilities_home[i]*100):.2f}%",
        # },
        # 'message': extra_data['message'],
        # 'id': game_data['game_id'],
        # 'live': simple_live_data,
      })
    return all_games
  elif receipt:
    for i in range(len(predictions)):
      # p_covers = f' | C - {round(covers_confidences[i]*100)}%' if covers_predictions[i] == 1 else ''
      # all_games.append(f'{"p-" if extra_data[i]["isProjectedLineup"] else ""}{game_data[i]["home_team"]["abbreviation"] if predictions[i] == 0 else game_data[i]["away_team"]["abbreviation"]} {round(confidences[i]*100)}%{p_covers} | {spread_predictions[i]} - {round(spread_confidences[i]*100)}%')
      all_games.append(f'{"p-" if extra_data[i]["isProjectedLineup"] else ""}{game_data[i]["home_team"]["abbreviation"] if predictions[i] == 0 else game_data[i]["away_team"]["abbreviation"]} {round(confidences[i]*100)}%')
    return all_games
  else:
    return {
      'predictions':predictions,
      'confidences':confidences,
    }
