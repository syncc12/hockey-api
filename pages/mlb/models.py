import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from joblib import load
import pandas as pd
from pages.mlb.inputs import X_INPUTS_MLB_T, ENCODE_COLUMNS
from constants.constants import MLB_TEAM_FILE_VERSION
from pages.mlb.constants import team_score
from pages.mlb.mlb_helpers import away_rename, home_rename, TEAM_IDS
import xgboost as xgb
import warnings

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")

W_MODELS = {team: load(f'models/mlb_ai_v{MLB_TEAM_FILE_VERSION}_xgboost_team{team}_winner.joblib') for team in TEAM_IDS}

def PREDICT_SCORE_H2H(datas, wModels, simple_return=False):
  away_probability = []
  home_probability = []
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
    wmAway = wModels[data['awayTeam']]
    wmHome = wModels[data['homeTeam']]

    awayData = pd.DataFrame(data, index=[0])
    homeData = pd.DataFrame(data, index=[0])

    awayData.rename(columns=away_rename, inplace=True)
    homeData.rename(columns=home_rename, inplace=True)
    
    for column in ENCODE_COLUMNS:
      encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
      awayData[column] = encoder.transform(awayData[column])
      homeData[column] = encoder.transform(homeData[column])

    awayData = awayData[X_INPUTS_MLB_T]
    homeData = homeData[X_INPUTS_MLB_T]

    daway = xgb.DMatrix(awayData)
    dhome = xgb.DMatrix(homeData)

    wmAP = wmAway.predict(daway)
    wmHP = wmHome.predict(dhome)
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
  
  away_probability = [i['probability'] for i in away_probability]
  home_probability = [i['probability'] for i in home_probability]
  away_predictions = [i['prediction'] for i in away_predictions]
  home_predictions = [i['prediction'] for i in home_predictions]

  if simple_return:
    return predictions,confidences
  return predictions,confidences,away_predictions,home_predictions,away_probability,home_probability