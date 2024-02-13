import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from flask import Flask, request, jsonify, Response
from joblib import load
import requests
from process import nhl_ai
from process2 import nhl_data, nhl_test
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
from util.models import MODEL_PREDICT, MODEL_CONFIDENCE
from util.returns import ai_return_dict_projectedLineup, ai_return_dict
from train_torch import predict_model
import xgboost as xgb
from constants.inputConstants import X_INPUTS

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
