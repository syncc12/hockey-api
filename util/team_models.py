import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from joblib import dump, load
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util.helpers import adjusted_winner
from inputs.inputs import master_inputs
from util.returns import ai_return_dict_projectedLineup
from constants.inputConstants import X_INPUTS, Y_OUTPUTS, X_INPUTS_T
from constants.constants import VERSION, FILE_VERSION, XGB_TEAM_VERSION, XGB_TEAM_FILE_VERSION, TORCH_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
from inputs.projectedLineup import testProjectedLineup
from train_torch import predict_model
import xgboost as xgb
import warnings

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")

home_rename = {
  'homeTeam': 'team',
  'awayTeam': 'opponent',
  'homeScore': 'score',
  'awayScore': 'opponentScore',
  'homeHeadCoach': 'headCoach',
  'awayHeadCoach': 'opponentHeadCoach',
  'homeForwardAverage': 'forwardAverage',
  'homeDefenseAverage': 'defenseAverage',
  'homeGoalieAverage': 'goalieAverage',
  'awayForwardAverage': 'opponentForwardAverage',
  'awayDefenseAverage': 'opponentDefenseAverage',
  'awayGoalieAverage': 'opponentGoalieAverage',
  'homeForwardAverageAge': 'forwardAverageAge',
  'homeDefenseAverageAge': 'defenseAverageAge',
  'homeGoalieAverageAge': 'goalieAverageAge',
  'awayForwardAverageAge': 'opponentForwardAverageAge',
  'awayDefenseAverageAge': 'opponentDefenseAverageAge',
  'awayGoalieAverageAge': 'opponentGoalieAverageAge',
  'winner': 'win',
  'winnerB': 'winB',
}
away_rename = {
  'homeTeam': 'opponent',
  'awayTeam': 'team',
  'homeScore': 'opponentScore',
  'awayScore': 'score',
  'homeHeadCoach': 'opponentHeadCoach',
  'awayHeadCoach': 'headCoach',
  'homeForwardAverage': 'opponentForwardAverage',
  'homeDefenseAverage': 'opponentDefenseAverage',
  'homeGoalieAverage': 'opponentGoalieAverage',
  'awayForwardAverage': 'forwardAverage',
  'awayDefenseAverage': 'defenseAverage',
  'awayGoalieAverage': 'goalieAverage',
  'homeForwardAverageAge': 'opponentForwardAverageAge',
  'homeDefenseAverageAge': 'opponentDefenseAverageAge',
  'homeGoalieAverageAge': 'opponentGoalieAverageAge',
  'awayForwardAverageAge': 'forwardAverageAge',
  'awayDefenseAverageAge': 'defenseAverageAge',
  'awayGoalieAverageAge': 'goalieAverageAge',
  'winner': 'win',
  'winnerB': 'winB',
}

TEAM_IDS = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,52,53,54,55]

W_MODELS = {}
L_MODELS = {}
for team in TEAM_IDS:
  if os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB.joblib'):
    W_MODELS[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB.joblib'), 'inverse': False}
  elif os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB_F.joblib'):
    W_MODELS[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB_F.joblib'), 'inverse': True}
  elif os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB_I.joblib'):
    W_MODELS[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB_I.joblib'), 'inverse': True}
  if os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB.joblib'):
    L_MODELS[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB.joblib'), 'inverse': False}
  elif os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB_F.joblib'):
    L_MODELS[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB_F.joblib'), 'inverse': True}
  elif os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB_I.joblib'):
    L_MODELS[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB_I.joblib'), 'inverse': True}

# W_MODELS = {team: load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB.joblib') for team in TEAM_IDS}
# L_MODELS = {team: load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB.joblib') for team in TEAM_IDS}

def PREDICT(data, team, wModels, lModels):
  wm = wModels[team]['model']
  lm = lModels[team]['model']
  wmInverse = wModels[team]['inverse']
  lmInverse = lModels[team]['inverse']
  dinput = xgb.DMatrix(data)
  w_probability = wm.predict(dinput)
  l_probability = lm.predict(dinput)
  w_prediction = [1 if i < 0.5 else 0 for i in w_probability] if wmInverse else [1 if i > 0.5 else 0 for i in w_probability]
  l_prediction = [1 if i < 0.5 else 0 for i in l_probability] if lmInverse else [1 if i > 0.5 else 0 for i in l_probability]
  lReverse = [1 - i for i in l_probability]
  sv = []
  for i in range(0,len(w_probability)):
    sv.append((w_probability[i] + lReverse[i]) / 2)
  svp = [1 if i > 0.5 else 0 for i in sv]
  return svp,sv,w_prediction,w_probability,l_prediction,l_probability

def TEST_H2H(datas, wModels, lModels):
  w_probability_away = []
  l_probability_away = []
  w_probability_home = []
  l_probability_home = []
  for data in datas:
    wmAway = wModels[data['awayTeam']]['model']
    lmAway = lModels[data['awayTeam']]['model']
    wmHome = wModels[data['homeTeam']]['model']
    lmHome = lModels[data['homeTeam']]['model']
    wmAwayInverse = wModels[data['awayTeam']]['inverse']
    lmAwayInverse = lModels[data['awayTeam']]['inverse']
    wmHomeInverse = wModels[data['homeTeam']]['inverse']
    lmHomeInverse = lModels[data['homeTeam']]['inverse']

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])

    awayData.rename(columns=away_rename, inplace=True)
    awayData['lossB'] = 1 - awayData['winB']
    homeData.rename(columns=home_rename, inplace=True)
    homeData['winB'] = 1 - homeData['winB']
    homeData['lossB'] = 1 - homeData['winB']

    awayData = awayData[X_INPUTS_T]
    homeData = homeData[X_INPUTS_T]

    daway = xgb.DMatrix(awayData)
    dhome = xgb.DMatrix(homeData)

    wmAP = wmAway.predict(daway)
    lmAP = lmAway.predict(daway)
    wmHP = wmHome.predict(dhome)
    lmHP = lmHome.predict(dhome)
    w_probability_away.append(1-wmAP[0] if wmAwayInverse else wmAP[0])
    l_probability_away.append(1-lmAP[0] if lmAwayInverse else lmAP[0])
    w_probability_home.append(1-wmHP[0] if wmHomeInverse else wmHP[0])
    l_probability_home.append(1-lmHP[0] if lmHomeInverse else lmHP[0])

  away_probability = []
  for i in range(0,len(w_probability_away)):
    away_probability.append((w_probability_away[i] + (1 - l_probability_away[i])) / 2)

  home_probability = []
  for i in range(0,len(w_probability_home)):
    home_probability.append((w_probability_home[i] + (1 - l_probability_home[i])) / 2)

  w_prediction_away = [1 if i > 0.5 else 0 for i in w_probability_away]
  l_prediction_away = [1 if i > 0.5 else 0 for i in l_probability_away]
  w_prediction_home = [1 if i > 0.5 else 0 for i in w_probability_home]
  l_prediction_home = [1 if i > 0.5 else 0 for i in l_probability_home]

  away_prediction = [1 if i > 0.5 else 0 for i in away_probability]
  home_prediction = [1 if i > 0.5 else 0 for i in home_probability]
  predictions = [1 if away_prediction[i] > home_prediction[i] else 0 for i in range(0,len(away_prediction))]
  confidences = [away_probability[i] if prediction == 1 else home_probability[i] for prediction in predictions]
  
  return predictions,confidences,away_prediction,home_prediction,away_probability,home_probability,w_prediction_away,l_prediction_away,w_prediction_home,l_prediction_home,w_probability_away,l_probability_away,w_probability_home,l_probability_home

def PREDICT_H2H(datas, wModels, lModels):
  w_probability_away = []
  l_probability_away = []
  w_probability_home = []
  l_probability_home = []
  for data in datas:
    wmAway = wModels[data['awayTeam']]['model']
    lmAway = lModels[data['awayTeam']]['model']
    wmHome = wModels[data['homeTeam']]['model']
    lmHome = lModels[data['homeTeam']]['model']
    wmAwayInverse = wModels[data['awayTeam']]['inverse']
    lmAwayInverse = lModels[data['awayTeam']]['inverse']
    wmHomeInverse = wModels[data['homeTeam']]['inverse']
    lmHomeInverse = lModels[data['homeTeam']]['inverse']

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])
    awayData.rename(columns=away_rename, inplace=True)
    homeData.rename(columns=home_rename, inplace=True)

    awayData = awayData[X_INPUTS_T]
    homeData = homeData[X_INPUTS_T]

    daway = xgb.DMatrix(awayData)
    dhome = xgb.DMatrix(homeData)

    wmAP = wmAway.predict(daway)
    lmAP = lmAway.predict(daway)
    wmHP = wmHome.predict(dhome)
    lmHP = lmHome.predict(dhome)
    w_probability_away.append(1-wmAP[0] if wmAwayInverse else wmAP[0])
    l_probability_away.append(1-lmAP[0] if lmAwayInverse else lmAP[0])
    w_probability_home.append(1-wmHP[0] if wmHomeInverse else wmHP[0])
    l_probability_home.append(1-lmHP[0] if lmHomeInverse else lmHP[0])
  
  away_probability = []
  for i in range(0,len(w_probability_away)):
    away_probability.append((w_probability_away[i] + (1 - l_probability_away[i])) / 2)

  home_probability = []
  for i in range(0,len(w_probability_home)):
    home_probability.append((w_probability_home[i] + (1 - l_probability_home[i])) / 2)

  w_predictions_away = [1 if i > 0.5 else 0 for i in w_probability_away]
  l_predictions_away = [1 if i > 0.5 else 0 for i in l_probability_away]
  w_predictions_home = [1 if i > 0.5 else 0 for i in w_probability_home]
  l_predictions_home = [1 if i > 0.5 else 0 for i in l_probability_home]

  away_predictions = [1 if i > 0.5 else 0 for i in away_probability]
  home_predictions = [1 if i > 0.5 else 0 for i in home_probability]
  predictions = [1 if away_predictions[i] > home_predictions[i] else 0 for i in range(0,len(away_predictions))]
  confidences = [away_probability[i] if prediction == 1 else home_probability[i] for prediction in predictions]

  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability,w_predictions_away,l_predictions_away,w_predictions_home,l_predictions_home,w_probability_away,l_probability_away,w_probability_home,l_probability_home