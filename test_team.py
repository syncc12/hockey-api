import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
import os
from constants.inputConstants import X_INPUTS, Y_OUTPUTS, X_INPUTS_T
from constants.constants import VERSION, FILE_VERSION, XGB_TEAM_VERSION, XGB_TEAM_FILE_VERSION, TORCH_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import xgboost as xgb
import warnings
from util.team_models import PREDICT, PREDICT_H2H, PREDICT_SCORE_H2H, PREDICT_CALIBRATED_H2H, PREDICT_CALIBRATED_SCORE_H2H, W_MODELS, L_MODELS, W_MODELS_C, L_MODELS_C
from util.team_helpers import away_rename, home_rename
from util.helpers import team_lookup
from training_input import test_input
from sklearn.metrics import accuracy_score


# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]

OUTPUT = 'winnerB'
TEST_DATA = test_input(X_INPUTS_T,[OUTPUT],no_format=True)
test_df = pd.DataFrame(TEST_DATA)
test_data1 = pd.DataFrame(TEST_DATA)
test_data2 = pd.DataFrame(TEST_DATA)
test_data1.rename(columns=home_rename, inplace=True)
test_data1['winB'] = 1 - test_data1['winB']
test_data1['lossB'] = 1 - test_data1['winB']
test_data2.rename(columns=away_rename, inplace=True)
test_data2['lossB'] = 1 - test_data2['winB']
test_data = pd.concat([test_data1, test_data2], axis=0)
test_data.reset_index(drop=True, inplace=True)
test_teams = test_data.groupby('team')

teamLookup = team_lookup(db)

def team_by_team_accuracies():
  accuracies = {}
  for team, team_data in test_teams:
    team_name = teamLookup[team]['abbrev']
    x_test = team_data [X_INPUTS_T]
    y_test_winB = team_data [['winB']].values.ravel()
    y_test_lossB = team_data [['lossB']].values.ravel()
    dtests_winB = xgb.DMatrix(x_test, label=y_test_winB)
    dtests_lossB = xgb.DMatrix(x_test, label=y_test_lossB)
    preds_w = W_MODELS[team]['model'].predict(dtests_winB)
    preds_l = L_MODELS[team]['model'].predict(dtests_lossB)
    predictions_w = [1 if i < 0.5 else 0 for i in preds_w] if W_MODELS[team]['inverse'] else [1 if i > 0.5 else 0 for i in preds_w]
    predictions_l = [1 if i < 0.5 else 0 for i in preds_l] if L_MODELS[team]['inverse'] else [1 if i > 0.5 else 0 for i in preds_l]
    accuracy_w = accuracy_score(y_test_winB, predictions_w)
    accuracy_l = accuracy_score(y_test_lossB, predictions_l)
    wl = (accuracy_w + accuracy_l) / 2
    accuracies[team] = {'team':team_name,'winB':accuracy_w,'lossB':accuracy_l,'score':wl,'id':team}

  print(accuracies)

def team_by_team_calibrated_accuracies():
  accuracies = {}
  for team, team_data in test_teams:
    team_name = teamLookup[team]['abbrev']
    x_test = team_data [X_INPUTS_T]
    y_test_winB = team_data [['winB']].values.ravel()
    y_test_lossB = team_data [['lossB']].values.ravel()
    preds_w = W_MODELS_C[team]['model'].predict_proba(x_test)
    preds_l = L_MODELS_C[team]['model'].predict_proba(x_test)
    predictions_w = [1 if max(i) < 0.5 else 0 for i in preds_w] if W_MODELS_C[team]['inverse'] else [1 if max(i) > 0.5 else 0 for i in preds_w]
    predictions_l = [1 if max(i) < 0.5 else 0 for i in preds_l] if L_MODELS_C[team]['inverse'] else [1 if max(i) > 0.5 else 0 for i in preds_l]
    accuracy_w = accuracy_score(y_test_winB, predictions_w)
    accuracy_l = accuracy_score(y_test_lossB, predictions_l)
    wl = (accuracy_w + accuracy_l) / 2
    accuracies[team] = {'team':team_name,'winB':accuracy_w,'lossB':accuracy_l,'score':wl,'id':team}

  print(accuracies)
  for i in accuracies.items():
    print(i)

def overall_accuracies():

  # prediction,confidences, *other = PREDICT_H2H(TEST_DATA, W_MODELS, L_MODELS, test=True)
  # score_prediction,score_confidences, *other = PREDICT_SCORE_H2H(TEST_DATA, W_MODELS, L_MODELS, test=True)
  calibrated_prediction,calibrated_confidences, *other = PREDICT_CALIBRATED_H2H(TEST_DATA, W_MODELS_C, L_MODELS_C, test=True)
  # calibrated_score_prediction,calibrated_score_confidences, *other = PREDICT_CALIBRATED_SCORE_H2H(TEST_DATA, W_MODELS_C, L_MODELS_C, test=True)
  y_test = test_df [[OUTPUT]].values.ravel()

  # accuracy = accuracy_score(y_test, prediction)
  # score_accuracy = accuracy_score(y_test, score_prediction)
  calibrated_accuracy = accuracy_score(y_test, calibrated_prediction)
  # calibrated_score_accuracy = accuracy_score(y_test, calibrated_score_prediction)
  # print(f'Accuracy: {(accuracy*100):.2f}% - {len(y_test)} games')
  # print(f'Score Accuracy: {(score_accuracy*100):.2f}% - {len(y_test)} games')
  print(f'Calibrated Accuracy: {(calibrated_accuracy*100):.2f}% - {len(y_test)} games')
  # print(f'Calibrated Score Accuracy: {(calibrated_score_accuracy*100):.2f}% - {len(y_test)} games')

def debug():
  # print(TEST_DATA)
  # print(TEST_DATA[0])
  print(test_df [[OUTPUT]].iloc[0].values.ravel())


if __name__ == '__main__':
  # team_by_team_accuracies()
  # team_by_team_calibrated_accuracies()
  overall_accuracies()
  # debug()