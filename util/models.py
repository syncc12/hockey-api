import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util.helpers import adjusted_winner
from inputs.inputs import master_inputs
from util.returns import ai_return_dict_projectedLineup
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, TORCH_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
from inputs.projectedLineup import testProjectedLineup
from train_torch import predict_model
import xgboost as xgb

# RANDOM_STATE = 12
# FILE_VERSION = 7
Y_OUTPUTS
MODEL_NAMES = Y_OUTPUTS

def batch_predict(model, data=[]):
  return model.predict(data)

def winnerOffset(winnerId, homeId, awayId):
  if abs(winnerId - homeId) < abs(winnerId - awayId):
    return homeId, abs(winnerId - homeId)
  elif abs(winnerId - homeId) > abs(winnerId - awayId):
    return awayId, abs(winnerId - awayId)
  else:
    return -1, -1

# MODELS = dict([(f'model_{i}',load(f'models/nhl_ai_v{FILE_VERSION}_{i}.joblib')) for i in MODEL_NAMES])
# MODELS = dict([(f'model_{i}',load(f'models/nhl_ai_v{FILE_VERSION}_gbc_{i}.joblib')) for i in MODEL_NAMES])
MODELS = {}
for i in MODEL_NAMES:
  if i == 'winnerB':
    MODELS[f'model_{i}'] = load(f'models/nhl_ai_v{FILE_VERSION}_xgboost_winnerB.joblib')
  # elif i == 'goalDifferential':
  #   MODELS[f'model_{i}'] = load(f'models/nhl_ai_v{FILE_VERSION}_stacked_goalDifferential.joblib')
  else:
    MODELS[f'model_{i}'] = load(f'models/nhl_ai_v{FILE_VERSION}_{i}.joblib')

# MODELS[f'model_winnerR'] = load(f'models/nhl_ai_v{FILE_VERSION}_stacked_winnerB.joblib')

def MODEL_X_INPUTS(x):
  return dict([(f'x_{i}',x) for i in MODEL_NAMES])
def MODEL_Y_OUTPUTS(data): 
  return dict([(f'y_{i}',data [[i]].values.ravel()) for i in MODEL_NAMES])

def MODEL_TRAIN_TEST_SPLIT(modelX,modelY):
  return dict([(f'tts_{i}',train_test_split(modelX[f'x_{i}'], modelY[f'y_{i}'], test_size=0.2, random_state=RANDOM_STATE)) for i in MODEL_NAMES])

MODEL_CLF = dict([(f'clf_{i}',RandomForestClassifier(random_state=RANDOM_STATE)) for i in MODEL_NAMES])
def MODEL_FIT(tts):
  for i in MODEL_NAMES:
    line_tts = tts[f'tts_{i}']
    MODEL_CLF[f'clf_{i}'] = MODEL_CLF[f'clf_{i}'].fit(line_tts[0],line_tts[2])

def MODEL_TEST(tts):
  out_dict = {}
  for i in MODEL_NAMES:
    line_tts = tts[f'tts_{i}']
    out_dict[f'predictions_{i}'] = MODEL_CLF[f'clf_{i}'].predict(line_tts[1])
  return out_dict

def MODEL_ACCURACY(tts,predictions):
  out_dict = {}
  for i in MODEL_NAMES:
    line_tts = tts[f'tts_{i}']
    out_dict[f'{i}_accuracy'] = accuracy_score(line_tts[3],predictions[f'predictions_{i}'])
  return out_dict

def MODEL_ACCURACY_PRINT(accuracies):
  for i in MODEL_NAMES:
    print(f"{i} Accuracy:", accuracies[f'{i}_accuracy'])

def MODEL_DUMP(clf):
  for i in MODEL_NAMES:
    dump(clf[f'clf_{i}'], f'models/nhl_ai_v{FILE_VERSION}_{i}.joblib')

def MODEL_PREDICT(models,data):
  out_dict = {}
  # print(data)
  for i in MODEL_NAMES:
    if i == 'winnerB':
      winnerB_input = xgb.DMatrix(pd.DataFrame(data['data']['input_data'],index=[0]))
      probability = models[f'model_{i}'].predict(winnerB_input)
      prediction = [1 if i > 0.5 else 0 for i in probability]
      # print(prediction)
      out_dict[f'prediction_{i}'] = prediction[0]
    else:
      out_dict[f'prediction_{i}'] = models[f'model_{i}'].predict(data['data']['data'])
  return out_dict

def MODEL_CONFIDENCE(models,data):
  out_dict = {}
  for i in MODEL_NAMES:
    if i == 'winnerB':
      winnerB_input = xgb.DMatrix(pd.DataFrame(data['data']['input_data'],index=[0]))
      probability = models[f'model_{i}'].predict(winnerB_input)
      out_dict[f'confidence_{i}'] = round(probability[0] * 100)
    else:
      out_dict[f'confidence_{i}'] = int((np.max(models[f'model_{i}'].predict_proba(data['data']['data']), axis=1) * 100)[0])
  return out_dict

TEST_ALL_INIT = dict([(f'all_{i}_total',0) for i in MODEL_NAMES])
TEST_LINE_INIT = dict([(f'{i}_total',0) for i in MODEL_NAMES])

def TEST_LINE_UPDATE(testLine,r):
  return dict([(f'{i}_total', testLine[f'{i}_total'] + r['results'][i]) for i in MODEL_NAMES])

def TEST_ALL_UPDATE(testAll,testLines):
  return dict([(f'all_{i}_total', testAll[f'all_{i}_total'] + testLines[f'{i}_total']) for i in MODEL_NAMES])

def TEST_PREDICTION(models,test_data):
  out_dict = {}
  for i in MODEL_NAMES:
    # if i == 'winnerR':
    #   prediction = model.predict([test_data['data'][i]])
    #   out_dict_verbose['winner'] = prediction
    #   out_dict_verbose['homeTeam'] = test_data['input_data']['homeTeam']
    #   out_dict_verbose['awayTeam'] = test_data['input_data']['awayTeam']
    #   out_dict[f'test_prediction_{i}'] = prediction
    # else:
    model = models[f'model_{i}']
    out_dict[f'test_prediction_{i}'] = model.predict([test_data['data'][i]])
  return out_dict

def TEST_CONFIDENCE(models,test_data):
  out_dict = {}
  for i in MODEL_NAMES:
    model = models[f'model_{i}']
    out_dict[f'test_confidence_{i}'] = model.predict_proba([test_data['data'][i]])
  return out_dict

def TEST_COMPARE(prediction,awayId,homeId):
  out_dict = {}
  for i in MODEL_NAMES:
    if i == 'winner':
      test_prediction_data = adjusted_winner(awayId,homeId,prediction[f'test_prediction_{i}'][0])
    else:
      test_prediction_data = prediction[f'test_prediction_{i}'][0]
    out_dict[f'predicted_{i}'] = test_prediction_data
  return out_dict

def TEST_DATA(test_data,awayId,homeId):
  out_dict = {}
  for i in MODEL_NAMES:
    # test_prediction_data = []
    if i == 'winner':
      # print(test_data['result'])
      test_prediction_data = adjusted_winner(awayId,homeId,test_data['result'][i])
    else:
      test_prediction_data = test_data['result'][i]
    out_dict[f'test_{i}'] = test_prediction_data
  return out_dict

def TEST_RESULTS(prediction,test):
  out_dict = {}
  for i in MODEL_NAMES:
    out_dict[i] = 1 if prediction[f'predicted_{i}']==test[f'test_{i}'] else 0
  return out_dict

def TEST_CONFIDENCE_RESULTS(test_confidence):
  out_dict = {}
  for i in MODEL_NAMES:
    out_dict[i] = int((np.max(test_confidence[f'test_confidence_{i}'], axis=1) * 100)[0])
  return out_dict

def TEST_PREDICTION_PROJECTED_LINEUP(models,test_data,awayId,homeId):
  test_dict = {}
  winner_list = []
  winnerB_vote_0 = 0
  winnerB_vote_1 = 0
  homeScore_list = []
  awayScore_list = []
  totalGoals_list = []
  goalDifferential_list = []
  for j in range(0,len(test_data)):
    lineup = test_data[j]
    line_dict = {}
    for i in MODEL_NAMES:
      model = models[f'model_{i}']
      prediction = model.predict([lineup])
      line_dict[f'test_prediction_{i}'] = prediction
      if i == 'winner':
        winner_list.append(prediction[0])
      elif i == 'winnerB':
        if prediction[0] == 0: winnerB_vote_0 += 1
        if prediction[0] == 1: winnerB_vote_1 += 1
      elif i == 'homeScore':
        homeScore_list.append(prediction[0])
      elif i == 'awayScore':
        awayScore_list.append(prediction[0])
      elif i == 'totalGoals':
        totalGoals_list.append(prediction[0])
      elif i == 'goalDifferential':
        goalDifferential_list.append(prediction[0])
    test_dict[j] = line_dict
  
  if winnerB_vote_0 > winnerB_vote_1:
    winnerB_vote_champ = 0
  elif winnerB_vote_0 < winnerB_vote_1:
    winnerB_vote_champ = 1
  else:
    winnerB_vote_champ = -1
  winner_average = 0 if len(winner_list) <= 0 else sum(winner_list) / len(winner_list)
  awayScore_average = 0 if len(awayScore_list) <= 0 else sum(awayScore_list) / len(awayScore_list)
  homeScore_average = 0 if len(homeScore_list) <= 0 else sum(homeScore_list) / len(homeScore_list)
  totalGoals_average = 0 if len(totalGoals_list) <= 0 else sum(totalGoals_list) / len(totalGoals_list)
  goalDifferential_average = 0 if len(goalDifferential_list) <= 0 else sum(goalDifferential_list) / len(goalDifferential_list)
  winnerA, offsetA = winnerOffset(winner_average,homeId,awayId)

  return {
    'test_prediction_winner': [winnerA],
    'test_prediction_winnerB': [winnerB_vote_champ],
    'test_prediction_awayScore': [awayScore_average],
    'test_prediction_homeScore': [homeScore_average],
    'test_prediction_totalGoals': [totalGoals_average],
    'test_prediction_goalDifferential': [goalDifferential_average],
  }

def TEST_COMPARE_PROJECTED_LINEUP():
  pass

def TEST_DATA_PROJECTED_LINEUP(test_data,awayId,homeId):
  test_data = list(test_data['result'].values())[0]
  print(test_data)
  return {
    'test_winner': round(test_data['winner']),
    'test_winnerB': round(test_data['winnerB']),
    'test_awayScore': round(test_data['awayScore']),
    'test_homeScore': round(test_data['homeScore']),
    'test_totalGoals': round(test_data['totalGoals']),
    'test_goalDifferential': round(test_data['goalDifferential']),
  }

def winnersAgree(predicted_winner,predicted_winnerB,test_winnerB,homeId,awayId):
  if predicted_winnerB == 0 and predicted_winner == homeId:
    if test_winnerB == 0:
      return 1
    else:
      return 0
  elif predicted_winnerB == 1 and predicted_winner == awayId:
    if test_winnerB == 1:
      return 1
    else:
      return 0
  else:
    return