import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

import matplotlib.pyplot as plt
import pandas as pd
from inputs.inputs import master_inputs
from sklearn.metrics import accuracy_score, roc_curve, auc
from training_input import test_input
from constants.inputConstants import X_INPUTS_T
from constants.constants import VERSION, FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, TEST_DATA_FILE_VERSION, START_SEASON, END_SEASON
from util.team_models import PREDICT, PREDICT_H2H, PREDICT_SCORE_H2H, PREDICT_CALIBRATED_H2H, PREDICT_CALIBRATED_SCORE_H2H, W_MODELS, L_MODELS, W_MODELS_C, L_MODELS_C
from util.helpers import team_lookup
from util.team_helpers import away_rename, home_rename
from pymongo import MongoClient
import warnings
import xgboost as xgb
from collections import Counter
from joblib import dump, load
import shap

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")


OUTPUT = 'winnerB'
TEST_DATA = test_input(X_INPUTS_T,[OUTPUT],no_format=True)
test_df = pd.DataFrame(TEST_DATA)
# test_data1 = pd.DataFrame(TEST_DATA)
# test_data2 = pd.DataFrame(TEST_DATA)
# test_data1.rename(columns=home_rename, inplace=True)
# test_data1['winB'] = 1 - test_data1['winB']
# test_data1['lossB'] = 1 - test_data1['winB']
# test_data2.rename(columns=away_rename, inplace=True)
# test_data2['lossB'] = 1 - test_data2['winB']
# test_data = pd.concat([test_data1, test_data2], axis=0)
# test_data.reset_index(drop=True, inplace=True)

y_test = test_df [['winnerB']].values.ravel()

def plot_confidences():
  predictions,confidences,away_predictions,home_predictions,away_probability,home_probability,w_predictions_away,l_predictions_away,w_predictions_home,l_predictions_home,w_probability_away,l_probability_away,w_probability_home,l_probability_home = PREDICT_CALIBRATED_SCORE_H2H(TEST_DATA,W_MODELS_C,L_MODELS_C,test=True)
  correct_confidences = []
  incorrect_confidences = []
  for i in range(0,len(predictions)):
    if predictions[i] == y_test[i]:
      # correct_confidences.append(1 - abs(0.5 - confidences[i]))
      correct_confidences.append(confidences[i])
    else:
      # incorrect_confidences.append(1 - abs(0.5 - confidences[i]))
      incorrect_confidences.append( confidences[i])
  total_correct = len(correct_confidences)
  total_incorrect = len(incorrect_confidences)
  normal = total_incorrect / total_correct
  step = 10
  for j in range(0,100,step):
    g = [i for i in confidences if i >= float(f'0.{str(j).ljust(2,"0")}') and i < float(f'0.{str(j+step).ljust(2,"0")}')]
    c_g = [i for i in correct_confidences if i >= float(f'0.{str(j).ljust(2,"0")}') and i < float(f'0.{str(j+step).ljust(2,"0")}')]
    i_g = [i for i in incorrect_confidences if i >= float(f'0.{str(j).ljust(2,"0")}') and i < float(f'0.{str(j+step).ljust(2,"0")}')]
    print(f'{j}% - {j + (step-1)}%: {len(c_g)} ({(len(c_g)/len(g) if len(g) > 0 else 0)*100:.2f}%) {len(i_g)} ({(len(i_g)/len(g) if len(g) > 0 else 0)*100:.2f}%) {len(g)}')
  
  print(f'Total Correct: {total_correct}')
  print(f'Total Incorrect: {total_incorrect}')
  print(f'Normal: {normal}')
  print(f'Total: {len(confidences)}')
  plt.hist(correct_confidences, bins='auto', alpha=0.5, label='Correct', color='blue')
  plt.hist(incorrect_confidences, bins='auto', alpha=0.5, label='Incorrect', color='red')
  # plt.hist(confidences, bins='auto', alpha=0.5, label='All', color='green')
  plt.xlabel('Confidence')
  plt.ylabel('Count')
  plt.title('Confidence Count')
  plt.legend(loc="upper left")
  plt.show()


if __name__ == '__main__':
  plot_confidences()