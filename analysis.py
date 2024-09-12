import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

import matplotlib.pyplot as plt
import pandas as pd
from inputs.inputs import master_inputs
from sklearn.metrics import accuracy_score, roc_curve, auc
from constants.inputConstants import X_INPUTS
from constants.constants import VERSION, FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, TEST_DATA_FILE_VERSION, START_SEASON, END_SEASON
from util.models import MODELS
from pymongo import MongoClient
import warnings
import xgboost as xgb
from collections import Counter
from joblib import dump, load
import shap

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")


winnerB_correct_confidences = [
  74,38,16,18,25,93,41,77,35,66,66,48,42,40,38,17,30,47,76,12,53,39,45,82,
  31,72,32,61,21,28,95,34,24,62,80,57,90,78,24,84,16,74,62,62,39,30,63,86,
  80,63,66,70,46,64,74,24,15,16,49,55,30,15,39,11,11,75,72,50,23,31,49,62,
  32,31,23,65,59,69,14,41,21,70,56,57,13,19,44,77,20,24,51,73,50,68,32,58,
  75,62,54,10,57,23,91,82,44,49,37,90,23,84,13,42,53,71,19,91,71,18,73,19,
  84,77,41,37,54,36,63,10,39,85,50,79,21,73,64,22,76,85,35,80,63,73,31,86,
  67,7,29,64,76,42,44,91,38,89,42,39,12,34,33,7,67,81,28,90,17,46,62,94,7,
  12,75,81,31,47,58,85,23,7,92,32,78,41,65,49,55,89,17,74,92,74,11,24,44,
  60,14,52,35,45,52,28,85,56,67,10,87,92,26,73,49,39,15,25,77,56,55,30,37,
  48,16,65,46,27,66,53,12,90,63,7,52,86,49,62,51,62,67,20,19,56,37,7,44,6,
  42,91,22,88,65,73,71,16,31,4,6,89,38,42,55,41,27,12,51,26,11,35,84,90,
  32,72,47,29,73,42,26,12,65,8,24,22,46,65,61,76,60,20,23,78,19,59,35,42,
  66,38,28,32,80,72,75,52,91,17,69,46,52,65,80,64,84,20,73,76,16,7,10,13,
  86,82,55,37,48,88,36,11,26,67,10,70,60,47,33,32,23,79,80,66,35,34,26,57,
  39,10,78,83,73,60,43,90,28,18,82,63,79,43,27,44,51,8,83,74,40,78,36,69,
  85,79,24,38,25,75,66,93,53,81,89,7,32,67,27,84,49,14,11,34,42,35,30,66,
  58,11,53,47,42,35,53,21,82,66,44,84,47,47,79,76,22,72,54,65,80,52,23,86,
  35,38,64,41,43,68,21,39,4,16,88,73,9,49,74,86,62,84,41,87,39,50,58,76,
  80,36,58,31,85,64,50,47,62,19,19,20,21,49,65,12,28,29,59,10,22,72,79,41,
  83,8,59,61,40,62,14,32,9,74,67,39,7,47,69,81,78,27,33,73,24,80,64,78,67,
  48,74,17,82,31,21,58,72,50,26,54,35,29,55,87,83,57,37,57,52,55,56,60,61,
  76,11,83,28,44,52,80,83,12,8,33,48
]

winnerB_incorrect_confidences = [
  50,78,20,66,56,53,15,83,90,59,24,25,78,41,36,34,69,68,87,87,46,88,79,26,
  29,58,73,13,88,81,23,76,17,30,61,31,14,82,16,39,39,6,20,35,85,58,42,50,
  81,22,12,43,60,34,73,78,47,96,27,72,75,53,32,70,30,73,66,30,68,84,79,51,
  47,8,56,73,42,9,22,85,80,83,41,66,17,42,72,56,28,88,75,71,15,72,59,24,
  18,51,54,73,78,25,46,71,92,75,76,16,32,58,31,13,47,44,12,15,26,86,50,83,
  21,78,4,68,59,68,44,82,56,72,89,87,52,58,54,52,48,55,24,75,21,65,86,36,
  41,42,28,15,57,48,89,83,93,78,91,89,27,15,71,92,26,66,77,11,31,64,59,72,
  40,31,35,31,67,70,10,33,79,53,50,67,87,51,17,75,3,68,20,19,73,62,58,92,
  27,34,78,43,16,25,72,53,39,62,76,65,77,61,59,13,80,62,51,71,82,86,27,48,
  89,78,69,30,94,81,44,78,77,82,75,24,45,15,84,86,29,56,72,21,32,41,26,43,
  26,38,44,39,42,44,54,32,42,64,67,9,36,64,40,35,63,11,85,43,80,88,88,10,
  77,86,44,79,57,71,56,69,16,53,64,61,79,76,49,67,48,60,69,57,47,51,67,75,
  55,14,32,53,82,42,20,25,86,72,47,21,75,71,31,63,44,51,61,38,21,25,65,61,
  44,52,57,59,41,57,58,66,19,36,78,75,55,72,70,39,63,78,26,66,42,71,35,64,
  28,11,11,58,61,84,47,56,38,57,14,44,43,68,54,38,29,24,13,28,78,15,73,17,
  88
]


# # First plot
# fig1, ax1 = plt.subplots() # Creates a new figure and a subplot
# ax1.hist(winnerB_correct_confidences, bins='auto', alpha=0.7, label='Correct', edgecolor='black') # Plots on the first subplot
# ax1.set_title('WinnerB Correct Confidences') # Sets title for the first subplot

# # Second plot
# fig2, ax2 = plt.subplots() # Creates another new figure and a subplot
# ax2.hist(winnerB_incorrect_confidences, bins='auto', alpha=0.7, label='Incorrect', edgecolor='black') # Plots on the second subplot
# ax2.set_title('WinnerB Incorrect Confidences') # Sets title for the second subplot

# plt.show()

def test_calculation(db,model,model_name='model'):
  Boxscores = db['dev_boxscores']
  boxscore_list = list(Boxscores.find(
    {'id': {'$gte':2023000000}}
  ))

  results = []
  correct_confidences = []
  incorrect_confidences = []
  daily_percents = []

  test_results = {}

  for boxscore in boxscore_list:
    test_results[boxscore['gameDate']] = {}
    test_results[boxscore['gameDate']] = {
      'games': [],
      'percent': 0,
      'line_results': [],
      'analysis': {
        'correct_vs_day_size': [],
      },
    }
  # print('boxscore_list:',boxscore_list)
  for boxscore in boxscore_list:

    gameId = boxscore['id']
    inputs = master_inputs(db,boxscore)
    inputs = inputs['data']
    # print(inputs)
    df = pd.DataFrame([inputs])
    data = df [X_INPUTS]

    probability = model.predict(xgb.DMatrix(data))
    prediction = [1 if i > 0.5 else 0 for i in probability]

    actual = inputs['winnerB']
    calculation = 1 if prediction[0] == actual else 0
    results.append(calculation)
    if calculation == 1:
      correct_confidences.append(round(probability[0] * 100))
    else:
      incorrect_confidences.append(round(probability[0] * 100))

    test_results[boxscore['gameDate']]['line_results'].append(calculation)
    test_results[boxscore['gameDate']]['games'].append({
      'id': gameId,
      'home': inputs['homeTeam'],
      'away': inputs['awayTeam'],
      'homeTeam': boxscore['homeTeam']['abbrev'],
      'awayTeam': boxscore['awayTeam']['abbrev'],
      'homeScore': inputs['homeScore'],
      'awayScore': inputs['awayScore'],
      model_name: {
        'prediction': prediction[0],
        'actual': actual,
        'calculation': calculation,
        'confidence': round(probability[0] * 100),
      },
    })

  correct_vs_day_size = []
  correct_vs_team = []
  incorrect_vs_team = []
  correct_vs_matchup = []
  incorrect_vs_matchup = []
  # print('test_results:',test_results)
  for date in test_results:
    # print(date)
    percent = (sum(test_results[date]['line_results']) / len(test_results[date]['line_results'])) * 100
    test_results[date]['percent'] = percent
    daily_percents.append((percent,len(test_results[date]['line_results'])))
    correct_vs_day_size.append((percent,len(test_results[date]['line_results'])))
    test_results[date]['analysis']['correct_vs_day_size'].append((percent,len(test_results[date]['line_results'])))
    for game in test_results[date]['games']:
      predicted_team = game['homeTeam'] if game[model_name]['prediction'] == 1 else game['awayTeam']
      if game[model_name]['calculation'] == 1:
        correct_vs_team.append(predicted_team)
        correct_vs_matchup.append(f"{game['awayTeam']} v {game['homeTeam']}")
      else:
        incorrect_vs_team.append(predicted_team)
        incorrect_vs_matchup.append(f"{game['awayTeam']} v {game['homeTeam']}")

  test_results['analysis'] = {
    'correct_vs_day_size': correct_vs_day_size,
    'correct_vs_team': list(Counter(correct_vs_team).items()),
    'incorrect_vs_team': list(Counter(incorrect_vs_team).items()),
    'correct_vs_matchup': list(Counter(correct_vs_matchup).items()),
    'incorrect_vs_matchup': list(Counter(incorrect_vs_matchup).items()),
  }
  # test_results['Results'] = results
  # test_results['Daily Percents'] = daily_percents
  test_results['Correct Confidences'] = correct_confidences
  test_results['Incorrect Confidences'] = incorrect_confidences
  test_results['allPercent'] = (sum(results) / len(results)) * 100

  return test_results


def correct_vs_incorrect_confidences_winnerB(correct_confidences, incorrect_confidences):
  fig, ax = plt.subplots()
  ax.hist(correct_confidences, bins='auto', alpha=0.7, label='Correct', edgecolor='black')
  ax.hist(incorrect_confidences, bins='auto', alpha=0.7, label='Incorrect', edgecolor='black')
  ax.set_title('Correct vs. Incorrect Confidences')
  ax.legend(loc='upper right')
  plt.show()

def correct_vs_day_size(db,model):
  data = test_calculation(db,model)
  c_v_ds = data['analysis']['correct_vs_day_size']
  y,x = zip(*c_v_ds)
  plt.scatter(x, y)
  plt.title('Correct vs. Day Size')
  plt.xlabel('Day Size')
  plt.ylabel('Correct')
  plt.show()

def correct_vs_incorrect_team(data):
  fig, ax = plt.subplots()
  c_v_t = data['analysis']['correct_vs_team']
  i_v_t = data['analysis']['incorrect_vs_team']
  xc,yc = zip(*c_v_t)
  xi,yi = zip(*i_v_t)
  ax.scatter(xc, yc, label='Correct')
  ax.scatter(xi, yi, label='Incorrect')
  ax.set_title('Correct Team vs. Incorrect Team')
  plt.xlabel('Team')
  plt.ylabel('Count')
  ax.legend(loc='upper right')
  plt.show()

def correct_vs_incorrect_matchup(data):
  fig, ax = plt.subplots()
  c_v_t = data['analysis']['correct_vs_matchup']
  i_v_t = data['analysis']['incorrect_vs_matchup']
  yc,xc = zip(*c_v_t)
  yi,xi = zip(*i_v_t)
  ax.scatter(xc, yc, label='Correct')
  ax.scatter(xi, yi, label='Incorrect')
  ax.set_title('Correct Matchup vs. Incorrect Matchup')
  plt.xlabel('Matchup')
  plt.ylabel('Count')
  ax.legend(loc='upper right')
  plt.show()

def sampling(data):
  df = pd.DataFrame(data)
  class_counts = df['awayTeam'].value_counts()
  print(class_counts)

  # Calculate percentages
  class_percentages = class_counts / len(df) * 100
  print(class_percentages)

def predictions(model,dtest):
  preds = model.predict(dtest)
  return preds

def model_accuracy(model,dtest):
  preds = model.predict(dtest)
  predictions = [1 if i > 0.5 else 0 for i in preds]
  accuracy = accuracy_score(y_test, predictions)
  # print(f"Accuracy:", accuracy)
  return accuracy

def model_confidences(model,dtest):
  confidences = model.predict(dtest)
  # print(f"Confidences:", confidences)
  return confidences

def roc_auc_curve(y_test, y_pred_prob):
  fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
  roc_auc = auc(fpr, tpr)

  # Plot ROC curve
  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")
  plt.show()

def feature_importance(model, max_num_features=10):
  xgb.plot_importance(model, max_num_features=max_num_features) # Plot the top 10 features
  plt.show()

def shap_values(model, x_test):
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(x_test)
  shap.summary_plot(shap_values, x_test)

def visualize_one_tree(model):
  # xgb.to_graphviz(model, num_trees=0) # Visualize the first tree
  xgb.plot_tree(model, num_trees=0)
  plt.show()

if __name__ == '__main__':
  OUTPUT = 'winnerB'

  db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net"
  # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
  client = MongoClient(db_url)
  db = client['hockey']
  TEST_DATA = load(f'test_data/test_data_v{TEST_DATA_FILE_VERSION}.joblib')
  test_data = pd.DataFrame(TEST_DATA)
  test_data = test_data.sort_values(by='id')
  x_test = test_data [X_INPUTS]
  y_test = test_data [[OUTPUT]].values.ravel()
  dtest = xgb.DMatrix(x_test, label=y_test)
  # model = MODELS['model_winnerB']
  model = load(f'models/nhl_ai_v{XGB_FILE_VERSION}_xgboost_{OUTPUT}_1.joblib')
  accuracy = model_accuracy(model,dtest)
  print("Accuracy:", accuracy)
  # confidence = model_confidences(model,dtest)
  # roc_auc_curve(y_test, confidence)
  # feature_importance(model)
  # shap_values(model, x_test)
  # visualize_one_tree(model)