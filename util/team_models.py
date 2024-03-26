import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from joblib import dump, load
import pandas as pd
import numpy as np
import os
from sklearn.calibration import CalibratedClassifierCV
from constants.inputConstants import X_INPUTS, Y_OUTPUTS, X_INPUTS_T
from constants.constants import XGB_TEAM_FILE_VERSION, TEAM_FILE_VERSION, LGBM_TEAM_FILE_VERSION
from team_helpers import away_rename, home_rename, team_score, team_score_lgbm, team_spread_score, team_covers_score, TEAM_IDS
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import warnings

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")


W_MODELS = {}
L_MODELS = {}
S_MODELS = {}
C_MODELS = {}
W_MODELS_LGBM = {}
# W_MODELS_C = {}
# L_MODELS_C = {}
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

  S_MODELS[team] = load(f'models/nhl_ai_v{TEAM_FILE_VERSION}_team{team}_spread.joblib')
  W_MODELS_LGBM[team] = load(f'models/nhl_ai_v{LGBM_TEAM_FILE_VERSION}_lightgbm_team{team}_winB.joblib')

  if os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_spread_team{team}_covers.joblib'):
    C_MODELS[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_spread_team{team}_covers.joblib'), 'inverse': False}
  elif os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_spread_team{team}_covers_F.joblib'):
    C_MODELS[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_spread_team{team}_covers_F.joblib'), 'inverse': True}
  elif os.path.exists(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_spread_team{team}_covers_I.joblib'):
    C_MODELS[team] = {'model': load(f'models/nhl_ai_v{XGB_TEAM_FILE_VERSION}_xgboost_spread_team{team}_covers_I.joblib'), 'inverse': True}


def get_team_score(test_teams, teamLookup, models=(), model_type='xgb', score_type='moneyline'):
  accuracies = {}
  for team, team_data in test_teams:
    team_name = teamLookup[team]['abbrev']
    x_test = team_data [X_INPUTS_T]
    if score_type == 'moneyline':
      if model_type == 'xgb':
        y_test_winB = team_data [['winB']].values.ravel()
        y_test_lossB = team_data [['lossB']].values.ravel()
        dtests_winB = xgb.DMatrix(x_test, label=y_test_winB)
        dtests_lossB = xgb.DMatrix(x_test, label=y_test_lossB)
        wModels = models[0]
        lModels = models[1]
        preds_w = wModels[team]['model'].predict(dtests_winB)
        preds_l = lModels[team]['model'].predict(dtests_lossB)
        predictions_w = [1 if i < 0.5 else 0 for i in preds_w] if wModels[team]['inverse'] else [1 if i > 0.5 else 0 for i in preds_w]
        predictions_l = [1 if i < 0.5 else 0 for i in preds_l] if lModels[team]['inverse'] else [1 if i > 0.5 else 0 for i in preds_l]
        accuracy_w = accuracy_score(y_test_winB, predictions_w)
        accuracy_l = accuracy_score(y_test_lossB, predictions_l)
        wl = (accuracy_w + accuracy_l) / 2
        accuracies[team] = {'team':team_name,'winB':accuracy_w,'lossB':accuracy_l,'score':wl,'id':team}
      elif model_type == 'lgbm':
        y_test = team_data [['winB']].values.ravel()
        wModels = models
        preds = wModels[team].predict(x_test, num_iteration=wModels[team].best_iteration)
        predictions = [1 if i > 0.5 else 0 for i in preds]
        accuracy = accuracy_score(y_test, predictions)
        accuracies[team] = {'team':team_name,'winB':accuracy,'score':accuracy,'id':team}

    elif score_type == 'spread':
      y_test = team_data [['spread']].values.ravel()
      sModels = models
      predictions = sModels[team].predict(x_test)
      accuracy = accuracy_score(y_test, predictions)
      accuracies[team] = {'team':team_name,'spread':accuracy,'score':accuracy,'id':team}
    elif score_type == 'covers':
      y_test = team_data [['covers']].values.ravel()
      dtests = xgb.DMatrix(x_test, label=y_test)
      cModels = models
      preds = cModels[team]['model'].predict(dtests)
      predictions = [1 if i < 0.5 else 0 for i in preds] if cModels[team]['inverse'] else [1 if i > 0.5 else 0 for i in preds]
      accuracy = accuracy_score(y_test, predictions)
      accuracies[team] = {'team':team_name,'covers':accuracy,'score':accuracy,'id':team}
  return accuracies

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
  scores = []
  for data in datas:
    away_score = team_score[data['awayTeam']]['score']
    home_score = team_score[data['homeTeam']]['score']
    if away_score > home_score:
      score_type = 'away'
      use_score = away_score
    elif home_score > away_score:
      score_type = 'home'
      use_score = home_score
    elif away_score == home_score:
      score_type = 'both'
      use_score = away_score
    scores.append({'away':away_score,'home':home_score,'use':use_score})
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

  confidences = [confidence * scores[i]['use'] for i, confidence in enumerate(confidences)]
  confidences = [1 - abs(0.5 - i) for i in confidences]
  confidences = [1 - abs(0.8 - i) for i in confidences]
  
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

def PREDICT_SPREAD(datas, sModels, simple_return=False):
  away_probability = []
  home_probability = []
  away_predictions = []
  home_predictions = []
  for data in datas:
    smAway = sModels[data['awayTeam']]
    smHome = sModels[data['homeTeam']]

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])
    awayData.rename(columns=away_rename, inplace=True)
    homeData.rename(columns=home_rename, inplace=True)

    awayData = awayData[X_INPUTS_T]
    homeData = homeData[X_INPUTS_T]

    smAPro = smAway.predict_proba(awayData)
    smHPro = smHome.predict_proba(homeData)
    smAPre = smAway.predict(awayData)
    smHPre = smHome.predict(homeData)
    away_probability.append(np.max(smAPro[0]))
    home_probability.append(np.max(smHPro[0]))
    away_predictions.append(smAPre[0])
    home_predictions.append(smHPre[0])

  predictions = [away_predictions[i] if away_probability[i] > home_probability[i] else home_predictions[i] for i in range(0,len(away_predictions))]
  confidences = [away_probability[i] if away_probability[i] > home_probability[i] else home_probability[i] for i in range(0,len(away_predictions))]

  if simple_return:
    return predictions,confidences
  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability

def PREDICT_SCORE_SPREAD(datas, sModels, simple_return=False):
  away_data = []
  home_data = []
  scores = []
  for data in datas:
    away_score = team_spread_score[data['awayTeam']]['score']
    home_score = team_spread_score[data['homeTeam']]['score']
    if away_score > home_score:
      score_type = 'away'
      use_score = away_score
    elif home_score > away_score:
      score_type = 'home'
      use_score = home_score
    elif away_score == home_score:
      score_type = 'both'
      use_score = away_score
    scores.append({'away':away_score,'home':home_score,'use':use_score})
    smAway = sModels[data['awayTeam']]
    smHome = sModels[data['homeTeam']]

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])

    awayData.rename(columns=away_rename, inplace=True)
    homeData.rename(columns=home_rename, inplace=True)

    awayData = awayData[X_INPUTS_T]
    homeData = homeData[X_INPUTS_T]
    
    smAPro = smAway.predict_proba(awayData)
    smHPro = smHome.predict_proba(homeData)
    smAPre = smAway.predict(awayData)
    smHPre = smHome.predict(homeData)
    away_data.append({'prediction':smAPre[0],'probability':np.max(smAPro[0]),'score_type':score_type})
    home_data.append({'prediction':smHPre[0],'probability':np.max(smHPro[0]),'score_type':score_type})

  predictions = []
  for i in range(0,len(away_data)):
    if away_data[i]['score_type'] == 'away':
      predictions.append(away_data[i]['prediction'])
    elif away_data[i]['score_type'] == 'home':
      predictions.append(home_data[i]['prediction'])
    elif away_data[i]['score_type'] == 'both':
      predictions.append(away_data[i]['prediction'])

  confidences = []
  for i in range(0,len(predictions)):
    if away_data[i]['score_type'] == 'away':
      confidences.append(away_data[i]['probability'])
    elif away_data[i]['score_type'] == 'home':
      confidences.append(home_data[i]['probability'])
    elif away_data[i]['score_type'] == 'both':
      confidences.append(away_data[i]['probability'])
  
  away_probability = [i['probability'] for i in away_data]
  home_probability = [i['probability'] for i in home_data]
  away_predictions = [i['prediction'] for i in away_data]
  home_predictions = [i['prediction'] for i in home_data]

  if simple_return:
    return predictions,confidences
  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability

def PREDICT_COVERS(datas, cModels, simple_return=False):
  away_probability = []
  home_probability = []
  for data in datas:
    cmAway = cModels[data['awayTeam']]['model']
    cmHome = cModels[data['homeTeam']]['model']
    cmAwayInverse = cModels[data['awayTeam']]['inverse']
    cmHomeInverse = cModels[data['homeTeam']]['inverse']

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])
    awayData.rename(columns=away_rename, inplace=True)
    homeData.rename(columns=home_rename, inplace=True)

    awayData = awayData[X_INPUTS_T]
    homeData = homeData[X_INPUTS_T]

    daway = xgb.DMatrix(awayData)
    dhome = xgb.DMatrix(homeData)

    cmAP = cmAway.predict(daway)
    cmHP = cmHome.predict(dhome)
    away_probability.append(1-cmAP[0] if cmAwayInverse else cmAP[0])
    home_probability.append(1-cmHP[0] if cmHomeInverse else cmHP[0])

  away_predictions = [1 if i > 0.5 else 0 for i in away_probability]
  home_predictions = [1 if i > 0.5 else 0 for i in home_probability]

  predictions = [1 if away_predictions[i] > home_predictions[i] else 0 for i in range(0,len(away_predictions))]
  confidences = [away_probability[i] if prediction == 1 else home_probability[i] for i, prediction in enumerate(predictions)]

  if simple_return:
    return predictions,confidences

  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability

def PREDICT_SCORE_COVERS(datas, cModels, simple_return=False):
  away_probability = []
  home_probability = []
  scores = []
  for data in datas:
    away_score = team_covers_score[data['awayTeam']]['score']
    home_score = team_covers_score[data['homeTeam']]['score']
    if away_score > home_score:
      score_type = 'away'
      use_score = away_score
    elif home_score > away_score:
      score_type = 'home'
      use_score = home_score
    elif away_score == home_score:
      score_type = 'both'
      use_score = away_score
    scores.append({'away':away_score,'home':home_score,'use':use_score})
    cmAway = cModels[data['awayTeam']]['model']
    cmHome = cModels[data['homeTeam']]['model']
    cmAwayInverse = cModels[data['awayTeam']]['inverse']
    cmHomeInverse = cModels[data['homeTeam']]['inverse']

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])

    awayData.rename(columns=away_rename, inplace=True)
    homeData.rename(columns=home_rename, inplace=True)

    awayData = awayData[X_INPUTS_T]
    homeData = homeData[X_INPUTS_T]

    daway = xgb.DMatrix(awayData)
    dhome = xgb.DMatrix(homeData)

    cmAP = cmAway.predict(daway)
    cmHP = cmHome.predict(dhome)
    away_probability.append({'probability':1-cmAP[0] if cmAwayInverse else cmAP[0],'score_type':score_type})
    home_probability.append({'probability':1-cmHP[0] if cmHomeInverse else cmHP[0],'score_type':score_type})
  
  away_predictions = [{'prediction':1 if i['probability'] > 0.5 else 0, 'score_type':i['score_type']} for i in away_probability]
  home_predictions = [{'prediction':1 if i['probability'] > 0.5 else 0, 'score_type':i['score_type']} for i in home_probability]
  
  predictions = []
  for i in range(0,len(away_predictions)):
    if away_predictions[i]['score_type'] == 'away':
      predictions.append(away_predictions[i]['prediction'])
    elif away_predictions[i]['score_type'] == 'home':
      predictions.append(home_predictions[i]['prediction'])
    elif away_predictions[i]['score_type'] == 'both':
      predictions.append(away_predictions[i]['prediction'])
  
  confidences = []
  for i in range(0,len(predictions)):
    if away_predictions[i]['score_type'] == 'away':
      confidences.append(away_probability[i]['probability'])
    elif away_predictions[i]['score_type'] == 'home':
      confidences.append(home_probability[i]['probability'])
    elif away_predictions[i]['score_type'] == 'both':
      confidences.append(away_probability[i]['probability'])

  # confidences = [confidence * scores[i]['use'] for i, confidence in enumerate(confidences)]
  # confidences = [1 - abs(0.5 - i) for i in confidences]
  # confidences = [1 - abs(0.8 - i) for i in confidences]
  
  away_probability = [i['probability'] for i in away_probability]
  home_probability = [i['probability'] for i in home_probability]
  away_predictions = [i['prediction'] for i in away_predictions]
  home_predictions = [i['prediction'] for i in home_predictions]

  if simple_return:
    return predictions,confidences
  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability

def PREDICT_LGBM_H2H(datas, wModels, test=False, simple_return=False):
  away_probability = []
  home_probability = []
  for data in datas:
    wmAway = wModels[data['awayTeam']]['model']
    wmHome = wModels[data['homeTeam']]['model']

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])
    awayData.rename(columns=away_rename, inplace=True)
    homeData.rename(columns=home_rename, inplace=True)
    if test:
      homeData['winB'] = 1 - homeData['winB']

    daway = awayData[X_INPUTS_T]
    dhome = homeData[X_INPUTS_T]

    wmAP = wmAway.predict(daway, num_iteration=wmAway.best_iteration)
    wmHP = wmHome.predict(dhome, num_iteration=wmHome.best_iteration)
    away_probability.append(wmAP[0])
    home_probability.append(wmHP[0])

  away_predictions = [1 if i > 0.5 else 0 for i in away_probability]
  home_predictions = [1 if i > 0.5 else 0 for i in home_probability]
  predictions = [1 if away_predictions[i] > home_predictions[i] else 0 for i in range(0,len(away_predictions))]
  confidences = [away_probability[i] if prediction == 1 else home_probability[i] for i, prediction in enumerate(predictions)]

  if simple_return:
    return predictions,confidences

  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability

def PREDICT_LGBM_SCORE_H2H(datas, wModels, test=False, simple_return=False):
  away_probability = []
  home_probability = []
  scores = []
  for data in datas:
    away_score = team_score_lgbm[data['awayTeam']]['score']
    home_score = team_score_lgbm[data['homeTeam']]['score']
    if away_score > home_score:
      score_type = 'away'
      use_score = away_score
    elif home_score > away_score:
      score_type = 'home'
      use_score = home_score
    elif away_score == home_score:
      score_type = 'both'
      use_score = away_score
    scores.append({'away':away_score,'home':home_score,'use':use_score})
    wmAway = wModels[data['awayTeam']]
    wmHome = wModels[data['homeTeam']]

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])

    awayData.rename(columns=away_rename, inplace=True)
    homeData.rename(columns=home_rename, inplace=True)
    if test:
      homeData['winB'] = 1 - homeData['winB']

    daway = awayData[X_INPUTS_T]
    dhome = homeData[X_INPUTS_T]

    wmAP = wmAway.predict(daway, num_iteration=wmAway.best_iteration)
    wmHP = wmHome.predict(dhome, num_iteration=wmHome.best_iteration)
    away_probability.append({'probability':wmAP[0],'score_type':score_type})
    home_probability.append({'probability':wmHP[0],'score_type':score_type})

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

  # confidences = [confidence * scores[i]['use'] for i, confidence in enumerate(confidences)]
  # confidences = [1 - abs(0.5 - i) for i in confidences]
  # confidences = [1 - abs(0.8 - i) for i in confidences]
  
  away_probability = [i['probability'] for i in away_probability]
  home_probability = [i['probability'] for i in home_probability]
  away_predictions = [i['prediction'] for i in away_predictions]
  home_predictions = [i['prediction'] for i in home_predictions]

  if simple_return:
    return predictions,confidences
  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability
