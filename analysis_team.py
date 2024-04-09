import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from inputs.inputs import master_inputs
from sklearn.metrics import accuracy_score, roc_curve, auc, brier_score_loss
from training_input import training_input, test_input
from constants.inputConstants import X_INPUTS_T
from constants.constants import VERSION, FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, TEST_DATA_FILE_VERSION, START_SEASON, END_SEASON
from util.team_models import get_team_score, calibrate, PREDICT, PREDICT_H2H, PREDICT_SCORE_H2H, PREDICT_SPREAD, PREDICT_SCORE_SPREAD, PREDICT_COVERS, PREDICT_SCORE_COVERS, PREDICT_LGBM_H2H, PREDICT_LGBM_SCORE_H2H, W_MODELS, L_MODELS, S_MODELS, C_MODELS, W_MODELS_LGBM
from util.helpers import team_lookup
from util.team_helpers import away_rename, home_rename, franchise_map
from util.util import CalibrationClassifier
from pymongo import MongoClient
import warnings
import xgboost as xgb
import lightgbm as lgb
from collections import Counter
from joblib import dump, load
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from inputs.projectedLineup import projected_shift

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
  20052006,
  20062007,
  20072008,
  20082009,
  20092010,
  20102011,
  20112012,
  20122013,
  20132014,
  20142015,
  20152016,
  20162017,
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
for franchise in franchise_map:
  training_data1.loc[training_data1['team'] == franchise, 'team'] = franchise_map[franchise]
  training_data2.loc[training_data2['team'] == franchise, 'team'] = franchise_map[franchise]
training_data = pd.concat([training_data1, training_data2], axis=0)
training_data.reset_index(drop=True, inplace=True)
teams = training_data.groupby('team')
training_teams = training_data.groupby('team')
teams_group = training_data.groupby('team')

if TEAM:
  teams = [(TEAM, teams.get_group(TEAM))]

OUTPUT = 'winnerB'
TEST_DATA = test_input(no_format=True)
PROJECTED_TEST_DATA, TEST_DATA = projected_shift(TEST_DATA)
# TEST_DATA, CAL_DATA = TEST_DATA[:len(TEST_DATA)//2], TEST_DATA[len(TEST_DATA)//2:]
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

projected_test_df = pd.DataFrame(PROJECTED_TEST_DATA)
projected_test_data1 = pd.DataFrame(PROJECTED_TEST_DATA)
projected_test_data2 = pd.DataFrame(PROJECTED_TEST_DATA)
projected_test_data1.rename(columns=home_rename, inplace=True)
projected_test_data1['winB'] = 1 - projected_test_data1['winB']
projected_test_data1['lossB'] = 1 - projected_test_data1['winB']
projected_test_data2.rename(columns=away_rename, inplace=True)
projected_test_data2['lossB'] = 1 - projected_test_data2['winB']
projected_test_data = pd.concat([projected_test_data1, projected_test_data2], axis=0)
projected_test_data.reset_index(drop=True, inplace=True)
projected_test_teams = projected_test_data.groupby('team')
projected_test_teams_group = projected_test_data.groupby('team')

projected_x_test = projected_test_data [X_INPUTS_T]
projected_y_test = projected_test_df [[OUTPUT]].values.ravel()

def accuracy(test_data, y_test, wModels, lModels):
  predictions, *other = PREDICT_SCORE_H2H(test_data, wModels, lModels)
  accuracy = accuracy_score(y_test, predictions)
  print(f'Accuracy: {accuracy}')
  return accuracy

def accuracy_lgbm(test_data, y_test, wModels, prefix='', only_return=False):
  soft_predictions, soft_confidences = PREDICT_LGBM_H2H(test_data, wModels, simple_return=True)
  hard_predictions, hard_confidences = PREDICT_LGBM_SCORE_H2H(test_data, wModels, simple_return=True)
  soft_accuracy = accuracy_score(y_test, soft_predictions)
  hard_accuracy = accuracy_score(y_test, hard_predictions)
  if not only_return:
    print(f'{prefix}Soft Accuracy: {soft_accuracy}')
    print(f'{prefix}Hard Accuracy: {hard_accuracy}')
  return soft_accuracy, hard_accuracy

def accuracy_over_time_lgbm(test_data, y_test, wModels):
  soft_predictions, soft_confidences = PREDICT_LGBM_H2H(test_data, wModels, simple_return=True)
  hard_predictions, hard_confidences = PREDICT_LGBM_SCORE_H2H(test_data, wModels, simple_return=True)
  # step = 10
  fig, axs = plt.subplots(3,3)
  steps = [25,50,75,100,250,500,750,1000]
  grids = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1)]
  for step,grid in zip(steps,grids):
    soft_accuracies = []
    hard_accuracies = []
    x_axis = []
    for i in range(1,len(test_data),step):
      soft_accuracy = accuracy_score(y_test[i:i+step], soft_predictions[i:i+step])
      hard_accuracy = accuracy_score(y_test[i:i+step], hard_predictions[i:i+step])
      soft_accuracies.append(soft_accuracy)
      hard_accuracies.append(hard_accuracy)
      x_axis.append(i)
    axs[grid[0],grid[1]].plot(x_axis, soft_accuracies, label=f'Soft Accuracies step {step}')
    axs[grid[0],grid[1]].plot(x_axis, hard_accuracies, label=f'Hard Accuracies step {step}')
    axs[grid[0],grid[1]].set_title(f'{step} Accuracies')
    axs[grid[0],grid[1]].set_xlabel('Games')
    axs[grid[0],grid[1]].set_ylabel('Accuracy')
  fig.legend(loc='upper right')
  fig.tight_layout()
  plt.show()
  
def team_by_team_accuracy_over_time_lgbm(test_data, wModels):
  RANGE_START = 1
  STEP = 5
  # accuracies = {}
  x_axis = range(RANGE_START,85,STEP)
  fig, axs = plt.subplots(4,8)
  grids = [
    (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),
    (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
    (2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),
    (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),
  ]
  for grid, (team, team_data) in zip(grids,test_data):
    team_name = teamLookup[team]['abbrev']
    accuracies = []
    x_test = team_data [X_INPUTS_T]
    y_test = team_data [['winB']].values.ravel()
    model = wModels[team]
    preds = model.predict(x_test, num_iteration=model.best_iteration)
    predictions = [1 if i > 0.5 else 0 for i in preds]
    for i in x_axis:
      accuracy = accuracy_score(y_test[i:i+STEP], predictions[i:i+STEP])
      accuracies.append(accuracy)
    axs[grid[0],grid[1]].plot(x_axis, accuracies, label=f'{team_name} Accuracies step {STEP}')
    axs[grid[0],grid[1]].set_title(f'{team_name} Accuracies {STEP}')
    axs[grid[0],grid[1]].set_xlabel('Games')
    axs[grid[0],grid[1]].set_ylabel('Accuracy')
    # plt.plot(x_axis, accuracies, label=f'{team_name} Accuracies')
    # plt.title(f'{team_name} Accuracies')
    # plt.xlabel('Games')
    # plt.ylabel('Accuracy')
    fig.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

def single_accuracy_lgbm(test_data, y_test, wModels):
  # soft_predictions, soft_confidences = PREDICT_LGBM_H2H(test_data, wModels, simple_return=True)
  hard_predictions, hard_confidences, detail = PREDICT_LGBM_SCORE_H2H(test_data, wModels, detailed_return=True)
  # soft_accuracy = accuracy_score(y_test, soft_predictions)
  hard_accuracy = accuracy_score(y_test, hard_predictions)
  # print(f'Soft Accuracy: {soft_accuracy}')
  print(f'Hard Accuracy: {hard_accuracy}')

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
  predictions,confidences,detail = PREDICT_LGBM_SCORE_H2H(TEST_DATA,W_MODELS_LGBM,detailed_return=True)
  # predictions,confidences = PREDICT_LGBM_H2H(TEST_DATA,W_MODELS_LGBM,simple_return=True)
  # predictions,confidences = PREDICT_SCORE_COVERS(TEST_DATA,C_MODELS,simple_return=True)
  correct_confidences = []
  away_correct_confidences = []
  home_correct_confidences = []
  incorrect_confidences = []
  away_incorrect_confidences = []
  home_incorrect_confidences = []

  for i in range(0,len(predictions)):
    if predictions[i] == y_test[i]:
      # correct_confidences.append(1 - abs(0.5 - confidences[i]))
      correct_confidences.append(confidences[i])
      if detail[i] == 'away':
        away_correct_confidences.append(confidences[i])
      elif detail[i] == 'home':
        home_correct_confidences.append(confidences[i])
    else:
      # incorrect_confidences.append(1 - abs(0.5 - confidences[i]))
      incorrect_confidences.append(confidences[i])
      if detail[i] == 'away':
        away_incorrect_confidences.append(confidences[i])
      elif detail[i] == 'home':
        home_incorrect_confidences.append(confidences[i])

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
  fig, axs = plt.subplots(1, 3, figsize=(15, 10))
  sns.kdeplot(data=correct_confidences, fill=True, ax=axs[0], color="blue", alpha=0.5, label='Correct')
  sns.kdeplot(data=incorrect_confidences, fill=True, ax=axs[0], color="red", alpha=0.5, label='Incorrect')

  sns.kdeplot(data=away_correct_confidences, fill=True, ax=axs[1], color="blue", alpha=0.5, label='Away Correct')
  sns.kdeplot(data=away_incorrect_confidences, fill=True, ax=axs[1], color="red", alpha=0.5, label='Away Incorrect')

  sns.kdeplot(data=home_correct_confidences, fill=True, ax=axs[2], color="blue", alpha=0.5, label='Home Correct')
  sns.kdeplot(data=home_incorrect_confidences, fill=True, ax=axs[2], color="red", alpha=0.5, label='Home Incorrect')
  # bins = 1000
  # axs[0].hist(correct_confidences, bins=bins, alpha=0.5, label='Correct', color='blue')
  # axs[0].hist(incorrect_confidences, bins=bins, alpha=0.5, label='Incorrect', color='red')
  # axs[1].hist(away_correct_confidences, bins=bins, alpha=0.5, label='Away Correct', color='blue')
  # axs[1].hist(away_incorrect_confidences, bins=bins, alpha=0.5, label='Away Incorrect', color='red')
  # axs[2].hist(home_correct_confidences, bins=bins, alpha=0.5, label='Home Correct', color='blue')
  # axs[2].hist(home_incorrect_confidences, bins=bins, alpha=0.5, label='Home Incorrect', color='red')
  # plt.hist(confidences, bins='auto', alpha=0.5, label='All', color='green')
  for ax in axs:
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.legend()
  axs[0].set_title('Confidence Count')
  axs[1].set_title('Away Confidence Count')
  axs[2].set_title('Home Confidence Count')
  # axs[0].legend(loc="upper left")
  # axs[1].legend(loc="upper left")
  # axs[2].legend(loc="upper left")
  plt.tight_layout()
  plt.show()

# def team_by_team_feature_importance(models, max_num_features=10):
#   for i in models:
#     model = models[i]['model']
#     plt.figure(figsize=(10, 8))
#     ax = plt.gca()  # Get current axis
#     xgb.plot_importance(model, max_num_features=max_num_features, ax=ax)
#     plt.title(teamLookup[i]['abbrev'])
#     plt.show()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def team_by_team_brier_score(wModels):
  for team, team_data in test_teams:
    confidences = wModels[team].predict(team_data[X_INPUTS_T])
    score = brier_score_loss(team_data[['winB']], confidences)
    print(f'{teamLookup[team]["abbrev"]}: {score}')

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

def chunk_list(input_list, chunk_size):
  return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def dechunk_list(input_list):
  return [item for sublist in input_list for item in sublist]

def chunk_dataframe(input_df, num_chunks):
  return np.array_split(input_df, num_chunks)

def dechunk_dataframe(input_list):
  data = pd.concat(input_list, axis=0)
  data.reset_index(drop=True, inplace=True)
  return data

def online_accuracy_lgbm(test_data, wModels):
  chunks = [10,5,2]
  num_chunks = 10
  team_datasets = {}
  for team, team_data in test_data:
    team_chunks = chunk_dataframe(team_data, num_chunks)
    train_data = training_teams.get_group(team)
    team_datasets[team] = []
    for i in range(1,len(team_chunks)-1):
      train_chunk = dechunk_dataframe(team_chunks[0:i])
      train_chunk = dechunk_dataframe([train_data, train_chunk])
      test_chunk = dechunk_dataframe(team_chunks[i:len(team_chunks)])
      x_train_lgbm = train_chunk [X_INPUTS_T]
      y_train_lgbm = train_chunk [['winB']].values.ravel()
      x_test_lgbm = test_chunk [X_INPUTS_T]
      y_test_lgbm = test_chunk [['winB']].values.ravel()
      dtrain = lgb.Dataset(x_train_lgbm, label=y_train_lgbm)
      team_datasets[team].append({'dtrain': dtrain, 'x_test_lgbm': x_test_lgbm, 'y_test_lgbm': y_test_lgbm})

  for team in team_datasets:
    team_name = teamLookup[team]['abbrev']
    accuracies = []
    datasets = team_datasets[team]
    for dataset in datasets:
      dtrain = dataset['dtrain']
      x_test_lgbm = dataset['x_test_lgbm']
      y_test_lgbm = dataset['y_test_lgbm']
      model = wModels[team]
      params = model.params
      params.pop('num_iterations', None)
      num_boost_round = model.num_trees()
      model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
      preds = model.predict(x_test_lgbm, num_iteration=model.best_iteration)
      predictions = np.where(preds > 0.5, 1, 0)
      accuracy = accuracy_score(y_test_lgbm, predictions)
      accuracies.append(accuracy)
    print(f'{team_name}: {accuracies}')

def projected_flips(test_data,projected_test_data,wModels):
  soft_predictions, soft_confidences = PREDICT_LGBM_H2H(test_data, wModels, simple_return=True)
  hard_predictions, hard_confidences = PREDICT_LGBM_SCORE_H2H(test_data, wModels, simple_return=True)
  projected_soft_predictions, projected_soft_confidences = PREDICT_LGBM_H2H(projected_test_data, wModels, simple_return=True)
  projected_hard_predictions, projected_hard_confidences = PREDICT_LGBM_SCORE_H2H(projected_test_data, wModels, simple_return=True)
  soft_flips = 0
  hard_flips = 0
  for i in range(0,len(soft_predictions)):
    # print(soft_predictions[i],projected_soft_predictions[i])
    if soft_predictions[i] != projected_soft_predictions[i]:
      soft_flips += 1
  for i in range(0,len(hard_predictions)):
    if hard_predictions[i] != projected_hard_predictions[i]:
      hard_flips += 1
  print(f'Soft Flips: {soft_flips}/{len(soft_predictions)} ({(soft_flips/len(soft_predictions))*100:.2f}%) | {len(soft_predictions)} - {len(projected_soft_predictions)}')
  print(f'Hard Flips: {hard_flips}/{len(hard_predictions)} ({(hard_flips/len(hard_predictions))*100:.2f}%) | {len(hard_predictions)} - {len(projected_hard_predictions)}')

def team_by_team_projected_flips(test_data, wModels):
  for team, team_data in test_data:
    team_name = teamLookup[team]['abbrev']
    projected_team_data = projected_test_teams.get_group(team)
    x_test = team_data [X_INPUTS_T]
    y_test = team_data [['winB']].values.ravel()
    projected_x_test = projected_team_data [X_INPUTS_T]
    projected_y_test = projected_team_data [['winB']].values.ravel()
    model = wModels[team]
    preds = model.predict(x_test, num_iteration=model.best_iteration)
    # predictions = np.where(preds > 0.5, 1, 0)
    predictions = [1 if i > 0.5 else 0 for i in preds]
    projected_preds = model.predict(projected_x_test, num_iteration=model.best_iteration)
    # projected_predictions = np.where(projected_preds > 0.5, 1, 0)
    projected_predictions = [1 if i > 0.5 else 0 for i in projected_preds]
    flips = 0
    for i in range(0,len(predictions)):
      if predictions[i] != projected_predictions[i]:
        flips += 1
    print(f'{team_name}: {flips}/{len(predictions)} ({(flips/len(predictions))*100:.2f}%) | {len(predictions)} - {len(projected_predictions)}')

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
  # accuracy_lgbm(TEST_DATA, y_test, W_MODELS_LGBM)
  # team_by_team_accuracy_over_time_lgbm(test_teams, W_MODELS_LGBM)
  # accuracy_over_time_lgbm(TEST_DATA, y_test, W_MODELS_LGBM)
  team_by_team_projected_flips(test_teams, W_MODELS_LGBM)
  # projected_flips(TEST_DATA,PROJECTED_TEST_DATA,W_MODELS_LGBM)
  # online_accuracy_lgbm(test_teams, W_MODELS_LGBM)
  # team_by_team_brier_score(W_MODELS_LGBM)
  # single_accuracy_lgbm(TEST_DATA, y_test, W_MODELS_LGBM)
  # false_positives_negatives_lgbm(TEST_DATA, y_test, W_MODELS_LGBM)
  # plot_confidences()
  # projected_flips(TEST_DATA, y_test, W_MODELS_LGBM)
  # team_by_team_feature_importance(W_MODELS,100)
  # team_by_team_class_count('covers')
  pass