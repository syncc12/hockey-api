import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util.helpers import adjusted_winner
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON

# RANDOM_STATE = 12
# FILE_VERSION = 7

MODEL_NAMES = [
  'winner',
  'homeScore',
  'awayScore',
  'totalGoals',
  'goalDifferential',
  'finalPeriod',
  'pastRegulation',
  'awayShots',
  'homeShots',
  'awayShotsPeriod1',
  'homeShotsPeriod1',
  'awayShotsPeriod2',
  'homeShotsPeriod2',
  'awayShotsPeriod3',
  'homeShotsPeriod3',
  'awayShotsPeriod4',
  'homeShotsPeriod4',
  'awayShotsPeriod5',
  'homeShotsPeriod5',
  'awayScorePeriod1',
  'homeScorePeriod1',
  'awayScorePeriod2',
  'homeScorePeriod2',
  'awayScorePeriod3',
  'homeScorePeriod3',
  'awayScorePeriod4',
  'homeScorePeriod4',
  'awayScorePeriod5',
  'homeScorePeriod5',
  'period1PuckLine',
  'period2PuckLine',
  'period3PuckLine',
]

MODELS = dict([(f'model_{i}',load(f'models/nhl_ai_v{FILE_VERSION}_{i}.joblib')) for i in MODEL_NAMES])
# MODELS = dict([(f'model_{i}',f'load(models/nhl_ai_v{FILE_VERSION}_{i}.joblib)') for i in MODEL_NAMES])

def MODEL_X_INPUTS(x): dict([(f'x_{i}',x) for i in MODEL_NAMES])
def MODEL_Y_OUTPUTS(data): dict([(f'y_{i}',data [[i]].values.ravel()) for i in MODEL_NAMES])

def MODEL_TRAIN_TEST_SPLIT(modelX,modelY): dict([(f'tts_{i}',train_test_split(modelX[f'x_{i}'], modelY[f'y_{i}'], test_size=0.2, random_state=RANDOM_STATE)) for i in MODEL_NAMES])

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

TEST_ALL_INIT = dict([(f'all_{i}_total',0) for i in MODEL_NAMES])
TEST_LINE_INIT = dict([(f'{i}_total',0) for i in MODEL_NAMES])

def TEST_LINE_UPDATE(testLine,r): dict([(f'{i}_total', testLine[f'{i}_total'] + r[i]) for i in MODEL_NAMES])
def TEST_ALL_UPDATE(testAll,testLines): dict([(f'all_{i}_total', testAll[f'all_{i}_total'] + testLines[f'{i}_total']) for i in MODEL_NAMES])

def TEST_PREDICTION(models,test_data):
  out_dict = {}
  for i in MODEL_NAMES:
    model = models[f'model_{i}']
    out_dict[f'test_prediction_{i}'] = model.predict(test_data['data'])
  return out_dict

def TEST_CONFIDENCE(models,test_data):
  out_dict = {}
  for i in MODEL_NAMES:
    model = models[f'model_{i}']
    out_dict[f'test_confidence_{i}'] = model.predict_proba(test_data['data'])
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
    if i == 'winner':
      test_prediction_data = adjusted_winner(awayId,homeId,test_data['result'][i])
    else:
      test_prediction_data = test_data['result'][i]
      out_dict[f'test_{i}'] = test_prediction_data
  return out_dict

def TEST_RESULTS(id,prediction,test):
  out_dict = {'id':id}
  for i in MODEL_NAMES:
    out_dict[i] = 1 if prediction[f'predicted_{i}']==test[f'test_{i}'] else 0
  return out_dict

def TEST_CONFIDENCE_RESULTS(id,test_confidence):
  out_dict = {'id':id}
  for i in MODEL_NAMES:
    out_dict[i] = int((np.max(test_confidence[f'test_confidence_{i}'], axis=1) * 100)[0]),
  return out_dict
