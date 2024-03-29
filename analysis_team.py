import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from inputs.inputs import master_inputs
from sklearn.metrics import accuracy_score, roc_curve, auc
from training_input import training_input, test_input
from constants.inputConstants import X_INPUTS_T
from constants.constants import VERSION, FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, TEST_DATA_FILE_VERSION, START_SEASON, END_SEASON
from util.team_models import get_team_score, PREDICT, PREDICT_H2H, PREDICT_SCORE_H2H, PREDICT_SPREAD, PREDICT_SCORE_SPREAD, PREDICT_COVERS, PREDICT_SCORE_COVERS, PREDICT_LGBM_H2H, PREDICT_LGBM_SCORE_H2H, W_MODELS, L_MODELS, S_MODELS, C_MODELS, W_MODELS_LGBM
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

teamLookup  = team_lookup(db)

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")

TEAM = False

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
# TRAINING_DATA = training_input(SEASONS)
# training_data1 = pd.DataFrame(TRAINING_DATA)
# training_data2 = pd.DataFrame(TRAINING_DATA)

# training_data1.rename(columns=home_rename, inplace=True)
# training_data1['winB'] = 1 - training_data1['winB']
# training_data1['lossB'] = 1 - training_data1['winB']
# training_data2.rename(columns=away_rename, inplace=True)
# training_data2['lossB'] = 1 - training_data2['winB']
# training_data = pd.concat([training_data1, training_data2], axis=0)
# training_data.reset_index(drop=True, inplace=True)
# teams = training_data.groupby('team')
# teams_group = training_data.groupby('team')

# if TEAM:
#   teams = [(TEAM, teams.get_group(TEAM))]

OUTPUT = 'winnerB'
TEST_DATA = test_input(no_format=True)
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
test_teams_group = test_data.groupby('team')

x_test = test_data [X_INPUTS_T]
y_test = test_df [[OUTPUT]].values.ravel()

def accuracy(test_data, y_test, wModels, lModels):
  predictions, *other = PREDICT_SCORE_H2H(test_data, wModels, lModels)
  accuracy = accuracy_score(y_test, predictions)
  print(f'Accuracy: {accuracy}')
  return accuracy

def accuracy_lgbm(test_data, y_test, wModels):
  soft_predictions, soft_confidences = PREDICT_LGBM_H2H(test_data, wModels, simple_return=True)
  hard_predictions, hard_confidences = PREDICT_LGBM_SCORE_H2H(test_data, wModels, simple_return=True)
  soft_accuracy = accuracy_score(y_test, soft_predictions)
  hard_accuracy = accuracy_score(y_test, hard_predictions)
  print(f'Soft Accuracy: {soft_accuracy}')
  print(f'Hard Accuracy: {hard_accuracy}')
  return soft_accuracy, hard_accuracy

def spread_scores(x_test, y_test, sModels):
  scores = {}
  for team in sModels:
    predictions = sModels[team].predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    scores[team] = {
      'team': teamLookup[team]['abbrev'],
      'score': accuracy,
      'id': team
    }
  print(scores)

def spread_accuracies(test_data, y_test, sModels):
  predictions, *other = PREDICT_SCORE_SPREAD(test_data, sModels)
  accuracy = accuracy_score(y_test, predictions)
  print(f'Accuracy: {accuracy}')
  return accuracy

def covers_accuracies(test_data, y_test, cModels):
  class_counts = test_df.value_counts('covers')
  print(class_counts)
  predictions, *other = PREDICT_SCORE_COVERS(test_data, cModels)
  prediction_counts = Counter(predictions)
  print(prediction_counts)
  accuracy = accuracy_score(y_test, predictions)
  print(f'Accuracy: {accuracy}')
  return accuracy

def plot_confidences():
  # predictions,confidences = PREDICT_SCORE_H2H(TEST_DATA,W_MODELS,L_MODELS,test=True,simple_return=True)
  # predictions,confidences = PREDICT_SCORE_SPREAD(TEST_DATA,S_MODELS,simple_return=True)
  # predictions,confidences = PREDICT_LGBM_SCORE_H2H(TEST_DATA,W_MODELS_LGBM,simple_return=True)
  predictions,confidences = PREDICT_LGBM_H2H(TEST_DATA,W_MODELS_LGBM,simple_return=True)
  # predictions,confidences = PREDICT_SCORE_COVERS(TEST_DATA,C_MODELS,simple_return=True)
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
  bin = 10
  for j in range(0,101,bin):
    g = [i for i in confidences if i >= float(f'0.{str(j).ljust(2,"0")}') and i < float(f'0.{str(j+bin).ljust(2,"0")}')]
    c_g = [i for i in correct_confidences if i >= float(f'0.{str(j).ljust(2,"0")}') and i < float(f'0.{str(j+bin).ljust(2,"0")}')]
    i_g = [i for i in incorrect_confidences if i >= float(f'0.{str(j).ljust(2,"0")}') and i < float(f'0.{str(j+bin).ljust(2,"0")}')]
    print(f'{j}% - {j + (bin-1)}%: {len(c_g)} ({(len(c_g)/len(g) if len(g) > 0 else 0)*100:.2f}%) {len(i_g)} ({(len(i_g)/len(g) if len(g) > 0 else 0)*100:.2f}%) {len(g)}')
  
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

# def team_by_team_feature_importance(models, max_num_features=10):
#   for i in models:
#     model = models[i]['model']
#     plt.figure(figsize=(10, 8))
#     ax = plt.gca()  # Get current axis
#     xgb.plot_importance(model, max_num_features=max_num_features, ax=ax)
#     plt.title(teamLookup[i]['abbrev'])
#     plt.show()

def team_by_team_class_count(class_label='winB'):
  for team, team_data in teams:
    print(f'{teamLookup[team]["abbrev"]}:')
    class_counts = team_data.value_counts(class_label)
    class_values = list(class_counts.values)
    class_list = list(zip(class_counts.index,class_counts.values))
    class_max = max(class_values)
    class_list = [(i[0],i[1],round((1/(i[1]/class_max)))) for i in class_list]
    print(class_counts)
    print(class_list)

def team_by_team_spread_prediction_breakdown():
  for team, team_data in test_teams:
    team_name = teamLookup[team]['abbrev']
    x_test = team_data [X_INPUTS_T]
    y_test = team_data [['spread']].values.ravel()
    class_counts = Counter(y_test)
    predictions = S_MODELS[team].predict(x_test)
    prediction_counts = Counter(predictions)
    print(f'{team_name}: {prediction_counts} | {len(predictions)} || {class_counts} | {len(y_test)}')

def team_by_team_plot_confidences(team=1,output='winB'):
  team_data = test_teams_group.get_group(team)
  team_name = teamLookup[team]['abbrev']
  if output == 'winB':
    model = W_MODELS[team]
  else:
    model = L_MODELS[team]
  x_test = team_data [X_INPUTS_T]
  y_test = team_data [[output]].values.ravel()
  dtest = xgb.DMatrix(x_test, label=y_test)
  confidences = model['model'].predict(dtest)
  predictions = [1 if i < 0.5 else 0 for i in confidences] if model['inverse'] else [1 if i > 0.5 else 0 for i in confidences]
  correct_confidences = []
  incorrect_confidences = []
  for i in range(0,len(predictions)):
    if predictions[i] == y_test[i]:
      # correct_confidences.append(1 - abs(0.5 - confidences[i]))
      correct_confidences.append(confidences[i])
    else:
      # incorrect_confidences.append(1 - abs(0.5 - confidences[i]))
      incorrect_confidences.append(confidences[i])
  total_correct = len(correct_confidences)
  total_incorrect = len(incorrect_confidences)
  bin = 10
  for j in range(0,101,bin):
    g = [i for i in confidences if i >= float(f'0.{str(j).ljust(2,"0")}') and i < float(f'0.{str(j+bin).ljust(2,"0")}')]
    c_g = [i for i in correct_confidences if i >= float(f'0.{str(j).ljust(2,"0")}') and i < float(f'0.{str(j+bin).ljust(2,"0")}')]
    i_g = [i for i in incorrect_confidences if i >= float(f'0.{str(j).ljust(2,"0")}') and i < float(f'0.{str(j+bin).ljust(2,"0")}')]
    print(f'{j}% - {j + (bin-1)}%: {len(c_g)} ({(len(c_g)/len(g) if len(g) > 0 else 0)*100:.2f}%) {len(i_g)} ({(len(i_g)/len(g) if len(g) > 0 else 0)*100:.2f}%) {len(g)}')
  
  print(f'Breakdown: {Counter(predictions)}')
  print(f'Total Correct: {total_correct} ({(total_correct/len(confidences))*100:.2f}%)')
  print(f'Total Incorrect: {total_incorrect} ({(total_incorrect/len(confidences))*100:.2f}%)')
  print(f'Total: {len(confidences)}')
  plt.hist(correct_confidences, bins=bin, alpha=0.5, label='Correct', color='blue')
  plt.hist(incorrect_confidences, bins=bin, alpha=0.5, label='Incorrect', color='red')
  # plt.hist(confidences, bins='auto', alpha=0.5, label='All', color='green')
  plt.xlabel('Confidence')
  plt.ylabel('Count')
  plt.title(f'{team_name} {output[:-1].capitalize()}B{" F" if model["inverse"] else ""} Confidence Count')
  plt.legend(loc="upper left")
  plt.show()

def team_by_team_spread_plot_confidences(team=1):
  team_data = test_teams_group.get_group(team)
  team_name = teamLookup[team]['abbrev']
  model = S_MODELS[team]
  x_test = team_data [X_INPUTS_T]
  y_test = team_data [['spread']].values.ravel()
  predictions = model.predict(x_test)
  confidences = model.predict_proba(x_test)
  correct_confidences = []
  incorrect_confidences = []
  for i in range(0,len(predictions)):
    if predictions[i] == y_test[i]:
      correct_confidences.append(np.max(confidences[i]))
    else:
      incorrect_confidences.append(np.max(confidences[i]))
  total_correct = len(correct_confidences)
  total_incorrect = len(incorrect_confidences)
  bin = 10
  for j in range(0,101,bin):
    g = [np.max(i) for i in confidences if np.max(i) >= float(f'0.{str(j).ljust(2,"0")}') and np.max(i) < float(f'0.{str(j+bin).ljust(2,"0")}')]
    c_g = [np.max(i) for i in correct_confidences if np.max(i) >= float(f'0.{str(j).ljust(2,"0")}') and np.max(i) < float(f'0.{str(j+bin).ljust(2,"0")}')]
    i_g = [np.max(i) for i in incorrect_confidences if np.max(i) >= float(f'0.{str(j).ljust(2,"0")}') and np.max(i) < float(f'0.{str(j+bin).ljust(2,"0")}')]
    print(f'{j}% - {j + (bin-1)}%: {len(c_g)} ({(len(c_g)/len(g) if len(g) > 0 else 0)*100:.2f}%) {len(i_g)} ({(len(i_g)/len(g) if len(g) > 0 else 0)*100:.2f}%) {len(g)}')
  
  print(f'Breakdown: {Counter(predictions)}')
  print(f'Total Correct: {total_correct}')
  print(f'Total Incorrect: {total_incorrect}')
  print(f'Total: {len(confidences)}')
  plt.hist(correct_confidences, bins='auto', alpha=0.5, label='Correct', color='blue')
  plt.hist(incorrect_confidences, bins='auto', alpha=0.5, label='Incorrect', color='red')
  # plt.hist(confidences, bins='auto', alpha=0.5, label='All', color='green')
  plt.xlabel('Confidence')
  plt.ylabel('Count')
  plt.title(f'{team_name} Spread Confidence Count')
  plt.legend(loc="upper left")
  plt.show()

def false_positives_negatives_lgbm(test_data, y_test, wModels):
  soft_predictions, *other = PREDICT_LGBM_H2H(test_data, wModels)
  hard_predictions, *other = PREDICT_LGBM_SCORE_H2H(test_data, wModels)
  # Generate the confusion matrix
  soft_conf_matrix = confusion_matrix(y_test, soft_predictions)
  hard_conf_matrix = confusion_matrix(y_test, hard_predictions)

  # The confusion matrix is structured as follows:
  # [[true negatives, false positives],
  #  [false negatives, true positives]]

  # Extracting false positives and false negatives
  soft_false_positives = soft_conf_matrix[0][1]
  soft_false_negatives = soft_conf_matrix[1][0]
  hard_false_positives = hard_conf_matrix[0][1]
  hard_false_negatives = hard_conf_matrix[1][0]
  soft_true_positives = soft_conf_matrix[1][1]
  soft_true_negatives = soft_conf_matrix[0][0]
  hard_true_positives = hard_conf_matrix[1][1]
  hard_true_negatives = hard_conf_matrix[0][0]

  print("Soft Voting:")
  print(f"Positives: T:{soft_true_positives} F:{soft_false_positives}")
  print(f"Negatives: T:{soft_true_negatives} F:{soft_false_negatives}")
  print("Hard Voting:")
  print(f"Positives: T:{hard_true_positives} F:{hard_false_positives}")
  print(f"Negatives: T:{hard_true_negatives} F:{hard_false_negatives}")

if __name__ == '__main__':
  # spread_scores(x_test, y_test, S_MODELS)
  # spread_accuracies(TEST_DATA, y_test, S_MODELS)
  # covers_accuracies(TEST_DATA, y_test, C_MODELS)
  # team_by_team_spread_prediction_breakdown()
  # team_by_team_plot_confidences(team=5,output='winB')
  # team_by_team_spread_plot_confidences(team=8)
  # team_spread_score = get_team_score(test_teams=test_teams, teamLookup=teamLookup, models=(S_MODELS), score_type='spread')
  # # print(team_spread_score)
  # for k,v in team_spread_score.items():
  #   print(str(k) + ': ' + str(v) + ',')
  # team_covers_score = get_team_score(test_teams=test_teams, teamLookup=teamLookup, models=(C_MODELS), score_type='covers')
  # # print(team_covers_score)
  # for k,v in team_covers_score.items():
  #   print(str(k) + ': ' + str(v) + ',')
  # team_score = get_team_score(test_teams=test_teams, teamLookup=teamLookup, models=(W_MODELS_LGBM), model_type='lgbm', score_type='moneyline')
  # # print(team_score)
  # for k,v in team_score.items():
  #   print(str(k) + ': ' + str(v) + ',')
  # accuracy(TEST_DATA, y_test, W_MODELS, L_MODELS)
  accuracy_lgbm(TEST_DATA, y_test, W_MODELS_LGBM)
  # false_positives_negatives_lgbm(TEST_DATA, y_test, W_MODELS_LGBM)
  plot_confidences()
  # team_by_team_feature_importance(W_MODELS,100)
  # team_by_team_class_count('covers')
  pass