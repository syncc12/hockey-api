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
from constants.constants import VERSION, FILE_VERSION, H2O_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
# import h2o

# RANDOM_STATE = 12
# FILE_VERSION = 7

MODEL_NAMES = Y_OUTPUTS

MODELS = dict([(f'model_{i}',load(f'models/nhl_ai_v{FILE_VERSION}_{i}.joblib')) for i in MODEL_NAMES])
# MODELS = dict([(f'model_{i}',load(f'models/nhl_ai_v{FILE_VERSION}_gbc_{i}.joblib')) for i in MODEL_NAMES])
# MODELS = {}
# for i in MODEL_NAMES:
#   if i == 'winnerB':
#     # f'C:/Users/syncc/code/Hockey API/hockey_api/models/nhl_ai_v{H2O_FILE_VERSION}_h2o_{i}/StackedEnsemble_BestOfFamily_1_AutoML_1_20240121_02310'
#     MODELS[f'model_{i}'] = h2o.load_model(f'models/nhl_ai_v{H2O_FILE_VERSION}_h2o_winner/StackedEnsemble_BestOfFamily_1_AutoML_1_20240121_02310')
#   else:
#     MODELS[f'model_{i}'] = load(f'models/nhl_ai_v{FILE_VERSION}_{i}.joblib')


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
    # if i == 'winnerB':
    #   # print(data['data']['input_data'])
    #   selected_data = {key: data['data']['input_data'][key] for key in X_INPUTS}
    #   # print('selected_data',selected_data)
    #   hf = h2o.H2OFrame([selected_data])
    #   # print('hf',hf)
    #   out_dict[f'prediction_{i}'] = models[f'model_{i}'].predict(hf)
    # else:
    out_dict[f'prediction_{i}'] = models[f'model_{i}'].predict(data['data']['data'])
  return out_dict

def MODEL_CONFIDENCE(models,data):
  out_dict = {}
  for i in MODEL_NAMES:
    # if i == 'winnerB':
    #   pass
    # else:
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
