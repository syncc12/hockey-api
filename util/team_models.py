import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from joblib import dump, load
import pandas as pd
import numpy as np
import os
from sklearn.calibration import CalibratedClassifierCV
from constants.inputConstants import X_INPUTS, Y_OUTPUTS, X_INPUTS_T
from constants.constants import VERSION, FILE_VERSION, XGB_TEAM_VERSION, XGB_TEAM_FILE_VERSION, TORCH_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
from team_helpers import away_rename, home_rename, team_score, XGBWrapper, XGBWrapperInverse
import xgboost as xgb
import warnings

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")

TEAM_IDS = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,52,53,54,55]

W_MODELS = {}
L_MODELS = {}
W_MODELS_C = {}
L_MODELS_C = {}
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

  if os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB_CALIBRATED.joblib'):
    W_MODELS_C[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB_CALIBRATED.joblib'), 'inverse': False}
  elif os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB_CALIBRATED_F.joblib'):
    W_MODELS_C[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB_CALIBRATED_F.joblib'), 'inverse': True}
  elif os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB_CALIBRATED_I.joblib'):
    W_MODELS_C[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_winB_CALIBRATED_I.joblib'), 'inverse': True}

  if os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB_CALIBRATED.joblib'):
    L_MODELS_C[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB_CALIBRATED.joblib'), 'inverse': False}
  elif os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB_CALIBRATED_F.joblib'):
    L_MODELS_C[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB_CALIBRATED_F.joblib'), 'inverse': True}
  elif os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB_CALIBRATED_I.joblib'):
    L_MODELS_C[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_team{team}_lossB_CALIBRATED_I.joblib'), 'inverse': True}

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

def PREDICT_H2H(datas, wModels, lModels, test=False, simple_return=False):
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
    if test:
      awayData['lossB'] = 1 - awayData['winB']
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

  w_predictions_away = [1 if i > 0.5 else 0 for i in w_probability_away]
  l_predictions_away = [1 if i > 0.5 else 0 for i in l_probability_away]
  w_predictions_home = [1 if i > 0.5 else 0 for i in w_probability_home]
  l_predictions_home = [1 if i > 0.5 else 0 for i in l_probability_home]

  away_predictions = [1 if i > 0.5 else 0 for i in away_probability]
  home_predictions = [1 if i > 0.5 else 0 for i in home_probability]
  predictions = [1 if away_predictions[i] > home_predictions[i] else 0 for i in range(0,len(away_predictions))]
  confidences = [away_probability[i] if prediction == 1 else home_probability[i] for prediction in predictions]

  if simple_return:
    return predictions,confidences

  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability,w_predictions_away,l_predictions_away,w_predictions_home,l_predictions_home,w_probability_away,l_probability_away,w_probability_home,l_probability_home

def PREDICT_SCORE_H2H(datas, wModels, lModels, test=False, simple_return=False):
  w_probability_away = []
  l_probability_away = []
  w_probability_home = []
  l_probability_home = []
  for data in datas:
    if team_score[data['awayTeam']]['score'] > team_score[data['homeTeam']]['score']:
      score_type = 'away'
    elif team_score[data['homeTeam']]['score'] > team_score[data['awayTeam']]['score']:
      score_type = 'home'
    elif team_score[data['awayTeam']]['score'] == team_score[data['homeTeam']]['score']:
      score_type = 'both'
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
    if test:
      awayData['lossB'] = 1 - awayData['winB']
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
    w_probability_away.append({'probability':1-wmAP[0] if wmAwayInverse else wmAP[0],'score_type':score_type})
    l_probability_away.append({'probability':1-lmAP[0] if lmAwayInverse else lmAP[0],'score_type':score_type})
    w_probability_home.append({'probability':1-wmHP[0] if wmHomeInverse else wmHP[0],'score_type':score_type})
    l_probability_home.append({'probability':1-lmHP[0] if lmHomeInverse else lmHP[0],'score_type':score_type})
  
  away_probability = []
  for i in range(0,len(w_probability_away)):
    away_probability.append({'probability':(w_probability_away[i]['probability'] + (1 - l_probability_away[i]['probability'])) / 2,'score_type':w_probability_away[i]['score_type']})

  home_probability = []
  for i in range(0,len(w_probability_home)):
    home_probability.append({'probability':(w_probability_home[i]['probability'] + (1 - l_probability_home[i]['probability'])) / 2,'score_type':w_probability_home[i]['score_type']})

  w_predictions_away = [{'prediction':1 if i['probability'] > 0.5 else 0, 'score_type':i['score_type']} for i in w_probability_away]
  l_predictions_away = [{'prediction':1 if i['probability'] > 0.5 else 0, 'score_type':i['score_type']} for i in l_probability_away]
  w_predictions_home = [{'prediction':1 if i['probability'] > 0.5 else 0, 'score_type':i['score_type']} for i in w_probability_home]
  l_predictions_home = [{'prediction':1 if i['probability'] > 0.5 else 0, 'score_type':i['score_type']} for i in l_probability_home]

  away_predictions = [{'prediction':1 if i['probability'] > 0.5 else 0,'score_type':i['score_type']} for i in away_probability]
  home_predictions = [{'prediction':1 if i['probability'] > 0.5 else 0,'score_type':i['score_type']} for i in home_probability]
  predictions = []
  for i in range(0,len(away_predictions)):
    if away_predictions[i]['score_type'] == 'away':
      predictions.append(away_predictions[i]['prediction'])
    elif away_predictions[i]['score_type'] == 'home':
      predictions.append(0 if home_predictions[i]['prediction'] == 1 else 1)
    elif away_predictions[i]['score_type'] == 'both':
      predictions.append(1 if away_predictions[i]['prediction'] > home_predictions[i]['prediction'] else 0)
  confidences = []
  for i in range(0,len(predictions)):
    if away_predictions[i]['score_type'] == 'away':
      confidences.append(away_probability[i]['probability'])
    elif away_predictions[i]['score_type'] == 'home':
      confidences.append(home_probability[i]['probability'])
    elif away_predictions[i]['score_type'] == 'both':
      confidences.append(away_probability[i]['probability'] if predictions[i] == 1 else home_probability[i]['probability'])
  # confidences = [1 - abs(0.5 - i) for i in confidences]
  w_probability_away = [i['probability'] for i in w_probability_away]
  l_probability_away = [i['probability'] for i in l_probability_away]
  w_probability_home = [i['probability'] for i in w_probability_home]
  l_probability_home = [i['probability'] for i in l_probability_home]
  away_probability = [i['probability'] for i in away_probability]
  home_probability = [i['probability'] for i in home_probability]
  w_predictions_away = [i['prediction'] for i in w_predictions_away]
  l_predictions_away = [i['prediction'] for i in l_predictions_away]
  w_predictions_home = [i['prediction'] for i in w_predictions_home]
  l_predictions_home = [i['prediction'] for i in l_predictions_home]
  away_predictions = [i['prediction'] for i in away_predictions]
  home_predictions = [i['prediction'] for i in home_predictions]

  if simple_return:
    return predictions,confidences
  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability,w_predictions_away,l_predictions_away,w_predictions_home,l_predictions_home,w_probability_away,l_probability_away,w_probability_home,l_probability_home

def PREDICT_CALIBRATED_H2H(datas, wModelsC, lModelsC, test=False, simple_return=False):
  w_probability_away = []
  l_probability_away = []
  w_probability_home = []
  l_probability_home = []
  for data in datas:
    wmAway = wModelsC[data['awayTeam']]['model']
    lmAway = lModelsC[data['awayTeam']]['model']
    wmHome = wModelsC[data['homeTeam']]['model']
    lmHome = lModelsC[data['homeTeam']]['model']
    wmAwayInverse = wModelsC[data['awayTeam']]['inverse']
    lmAwayInverse = lModelsC[data['awayTeam']]['inverse']
    wmHomeInverse = wModelsC[data['homeTeam']]['inverse']
    lmHomeInverse = lModelsC[data['homeTeam']]['inverse']

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])
    awayData.rename(columns=away_rename, inplace=True)
    homeData.rename(columns=home_rename, inplace=True)
    if test:
      awayData['lossB'] = 1 - awayData['winB']
      homeData['winB'] = 1 - homeData['winB']
      homeData['lossB'] = 1 - homeData['winB']

    awayData = awayData[X_INPUTS_T]
    homeData = homeData[X_INPUTS_T]

    # wmAPre = wmAway.predict(awayData)
    # lmAPre = lmAway.predict(awayData)
    # wmHPre = wmHome.predict(homeData)
    # lmHPre = lmHome.predict(homeData)
    wmAPro = max(wmAway.predict_proba(awayData)[0])
    lmAPro = max(lmAway.predict_proba(awayData)[0])
    wmHPro = max(wmHome.predict_proba(homeData)[0])
    lmHPro = max(lmHome.predict_proba(homeData)[0])
    # print('wmAPro:',wmAPro)
    # print('lmAPro:',lmAPro)
    # print('wmHPro:',wmHPro)
    # print('lmHPro:',lmHPro)
    w_probability_away.append(1-wmAPro if wmAwayInverse else wmAPro)
    l_probability_away.append(1-lmAPro if lmAwayInverse else lmAPro)
    w_probability_home.append(1-wmHPro if wmHomeInverse else wmHPro)
    l_probability_home.append(1-lmHPro if lmHomeInverse else lmHPro)
    # print('w_probability_away:',w_probability_away)
    # print('l_probability_away:',l_probability_away)
    # print('w_probability_home:',w_probability_home)
    # print('l_probability_home:',l_probability_home)
  
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

  if simple_return:
    return predictions,confidences
  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability,w_predictions_away,l_predictions_away,w_predictions_home,l_predictions_home,w_probability_away,l_probability_away,w_probability_home,l_probability_home

def PREDICT_CALIBRATED_SCORE_H2H(datas, wModelsC, lModelsC, test=False, simple_return=False):
  w_probability_away = []
  l_probability_away = []
  w_probability_home = []
  l_probability_home = []
  for data in datas:
    if team_score[data['awayTeam']]['score'] > team_score[data['homeTeam']]['score']:
      score_type = 'away'
    elif team_score[data['homeTeam']]['score'] > team_score[data['awayTeam']]['score']:
      score_type = 'home'
    elif team_score[data['awayTeam']]['score'] == team_score[data['homeTeam']]['score']:
      score_type = 'both'
    wmAway = wModelsC[data['awayTeam']]['model']
    lmAway = lModelsC[data['awayTeam']]['model']
    wmHome = wModelsC[data['homeTeam']]['model']
    lmHome = lModelsC[data['homeTeam']]['model']
    wmAwayInverse = wModelsC[data['awayTeam']]['inverse']
    lmAwayInverse = lModelsC[data['awayTeam']]['inverse']
    wmHomeInverse = wModelsC[data['homeTeam']]['inverse']
    lmHomeInverse = lModelsC[data['homeTeam']]['inverse']

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])

    awayData.rename(columns=away_rename, inplace=True)
    homeData.rename(columns=home_rename, inplace=True)
    if test:
      awayData['lossB'] = 1 - awayData['winB']
      homeData['winB'] = 1 - homeData['winB']
      homeData['lossB'] = 1 - homeData['winB']

    awayData = awayData[X_INPUTS_T]
    homeData = homeData[X_INPUTS_T]

    # wmAPre = wmAway.predict(awayData)
    # lmAPre = lmAway.predict(awayData)
    # wmHPre = wmHome.predict(homeData)
    # lmHPre = lmHome.predict(homeData)
    wmAPro = max(wmAway.predict_proba(awayData)[0])
    lmAPro = max(lmAway.predict_proba(awayData)[0])
    wmHPro = max(wmHome.predict_proba(homeData)[0])
    lmHPro = max(lmHome.predict_proba(homeData)[0])
    w_probability_away.append({'probability':1-wmAPro if wmAwayInverse else wmAPro,'score_type':score_type})
    l_probability_away.append({'probability':1-lmAPro if lmAwayInverse else lmAPro,'score_type':score_type})
    w_probability_home.append({'probability':1-wmHPro if wmHomeInverse else wmHPro,'score_type':score_type})
    l_probability_home.append({'probability':1-lmHPro if lmHomeInverse else lmHPro,'score_type':score_type})
  
  away_probability = []
  for i in range(0,len(w_probability_away)):
    away_probability.append({'probability':(w_probability_away[i]['probability'] + (1 - l_probability_away[i]['probability'])) / 2,'score_type':w_probability_away[i]['score_type']})

  home_probability = []
  for i in range(0,len(w_probability_home)):
    home_probability.append({'probability':(w_probability_home[i]['probability'] + (1 - l_probability_home[i]['probability'])) / 2,'score_type':w_probability_home[i]['score_type']})

  w_predictions_away = [{'prediction':1 if i['probability'] > 0.5 else 0, 'score_type':i['score_type']} for i in w_probability_away]
  l_predictions_away = [{'prediction':1 if i['probability'] > 0.5 else 0, 'score_type':i['score_type']} for i in l_probability_away]
  w_predictions_home = [{'prediction':1 if i['probability'] > 0.5 else 0, 'score_type':i['score_type']} for i in w_probability_home]
  l_predictions_home = [{'prediction':1 if i['probability'] > 0.5 else 0, 'score_type':i['score_type']} for i in l_probability_home]

  away_predictions = [{'prediction':1 if i['probability'] > 0.5 else 0,'score_type':i['score_type']} for i in away_probability]
  home_predictions = [{'prediction':1 if i['probability'] > 0.5 else 0,'score_type':i['score_type']} for i in home_probability]
  predictions = []
  for i in range(0,len(away_predictions)):
    if away_predictions[i]['score_type'] == 'away':
      predictions.append(away_predictions[i]['prediction'])
    elif away_predictions[i]['score_type'] == 'home':
      predictions.append(0 if home_predictions[i]['prediction'] == 1 else 1)
    elif away_predictions[i]['score_type'] == 'both':
      predictions.append(1 if away_predictions[i]['prediction'] > home_predictions[i]['prediction'] else 0)
  confidences = []
  for i in range(0,len(predictions)):
    if away_predictions[i]['score_type'] == 'away':
      confidences.append(away_probability[i]['probability'])
    elif away_predictions[i]['score_type'] == 'home':
      confidences.append(home_probability[i]['probability'])
    elif away_predictions[i]['score_type'] == 'both':
      confidences.append(away_probability[i]['probability'] if predictions[i] == 1 else home_probability[i]['probability'])
  confidences = [1 - abs(0.5 - i) for i in confidences]
  w_probability_away = [i['probability'] for i in w_probability_away]
  l_probability_away = [i['probability'] for i in l_probability_away]
  w_probability_home = [i['probability'] for i in w_probability_home]
  l_probability_home = [i['probability'] for i in l_probability_home]
  away_probability = [i['probability'] for i in away_probability]
  home_probability = [i['probability'] for i in home_probability]
  w_predictions_away = [i['prediction'] for i in w_predictions_away]
  l_predictions_away = [i['prediction'] for i in l_predictions_away]
  w_predictions_home = [i['prediction'] for i in w_predictions_home]
  l_predictions_home = [i['prediction'] for i in l_predictions_home]
  away_predictions = [i['prediction'] for i in away_predictions]
  home_predictions = [i['prediction'] for i in home_predictions]

  if simple_return:
    return predictions,confidences
  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability,w_predictions_away,l_predictions_away,w_predictions_home,l_predictions_home,w_probability_away,l_probability_away,w_probability_home,l_probability_home

