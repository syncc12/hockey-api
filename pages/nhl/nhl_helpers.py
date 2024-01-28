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
from util.returns import ai_return_dict_projectedLineup, ai_return_dict

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
    
    return ai_return_dict_projectedLineup(data,predictions,confidences)
