import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

import matplotlib.pyplot as plt
import pandas as pd
from inputs.inputs import master_inputs
from sklearn.metrics import accuracy_score, roc_curve, auc
from training_input import training_input, test_input
from constants.inputConstants import X_INPUTS_T
from constants.constants import VERSION, FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, TEST_DATA_FILE_VERSION, START_SEASON, END_SEASON
from util.team_models import PREDICT, PREDICT_H2H, PREDICT_SCORE_H2H, PREDICT_CALIBRATED_H2H, PREDICT_CALIBRATED_SCORE_H2H, W_MODELS, L_MODELS
from util.helpers import team_lookup
from util.team_helpers import away_rename, home_rename
from pymongo import MongoClient
import warnings
import xgboost as xgb
from collections import Counter
from joblib import dump, load
import shap

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['hockey']

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")

SEASONS = [
  # 20052006,
  # 20062007,
  # 20072008,
  # 20082009,
  # 20092010,
  # 20102011,
  # 20112012,
  # 20122013,
  # 20132014,
  # 20142015,
  # 20152016,
  # 20162017,
  20172018,
  20182019,
  20192020,
  20202021,
  20212022,
  20222023,
]
TRAINING_DATA = training_input(SEASONS)
training_data1 = pd.DataFrame(TRAINING_DATA)
training_data2 = pd.DataFrame(TRAINING_DATA)

training_data1.rename(columns=home_rename, inplace=True)
training_data1['winB'] = 1 - training_data1['winB']
training_data1['lossB'] = 1 - training_data1['winB']
training_data2.rename(columns=away_rename, inplace=True)
training_data2['lossB'] = 1 - training_data2['winB']
training_data = pd.concat([training_data1, training_data2], axis=0)
training_data.reset_index(drop=True, inplace=True)
teams = training_data.groupby('team')

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
  predictions,confidences,away_predictions,home_predictions,away_probability,home_probability,w_predictions_away,l_predictions_away,w_predictions_home,l_predictions_home,w_probability_away,l_probability_away,w_probability_home,l_probability_home = PREDICT_SCORE_H2H(TEST_DATA,W_MODELS,L_MODELS,test=True)
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

def team_by_team_feature_importance(models, max_num_features=10):
  teamLookup  = team_lookup(db)
  for i in models:
    model = models[i]['model']
    plt.figure(figsize=(10, 8))
    ax = plt.gca()  # Get current axis
    xgb.plot_importance(model, max_num_features=max_num_features, ax=ax)
    plt.title(teamLookup[i]['abbrev'])
    plt.show()

def team_by_team_class_count(class_label='winB'):
  teamLookup  = team_lookup(db)
  for team, team_data in teams:
    print(f'{teamLookup[team]["abbrev"]}:')
    class_counts = team_data.value_counts(class_label)
    print(class_counts[0])

if __name__ == '__main__':
  # team_by_team_feature_importance(W_MODELS,100)
  team_by_team_class_count('winB')