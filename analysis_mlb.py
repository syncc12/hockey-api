import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

import matplotlib.pyplot as plt
import pandas as pd
from inputs.inputs import master_inputs
from sklearn.metrics import accuracy_score, roc_curve, auc
from constants.constants import MLB_VERSION, MLB_FILE_VERSION, TEST_DATA_FILE_VERSION
from pages.mlb.inputs import X_INPUTS_MLB, ENCODE_COLUMNS, mlb_training_input
# from pages.mlb.models import W_MODELS
from pymongo import MongoClient
import warnings
import xgboost as xgb
from collections import Counter
from joblib import dump, load
import shap

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")


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
    data = df [X_INPUTS_MLB]

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
  for feature in plt.gca().get_yticklabels():
    print(f"'{feature.get_text()}',")
  print(f'Max Features: {max_num_features}')
  print(f'Plotted Feature Count: {len(plt.gca().get_yticklabels())}')
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
  OUTPUT = 'winner'

  # db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net"
  # # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
  # client = MongoClient(db_url)
  # db = client['hockey']
  # TEST_DATA = load(f'test_data/test_data_v{TEST_DATA_FILE_VERSION}.joblib')
  test_data = load(f'pages/mlb/data/test_data.joblib')
  for column in ENCODE_COLUMNS:
    encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
    test_data = test_data[test_data[column] != -1]
    test_data[column] = encoder.transform(test_data[column])
  test_data = test_data.sort_values(by='id')
  x_test = test_data [X_INPUTS_MLB]
  y_test = test_data [[OUTPUT]].values.ravel()
  # dtest = xgb.DMatrix(x_test, label=y_test)
  # model = MODELS['model_winnerB']
  model = load(f'models/mlb_ai_v{MLB_FILE_VERSION}_xgboost_{OUTPUT}.joblib')
  # accuracy = model_accuracy(model,dtest)
  # print("Accuracy:", accuracy)
  # confidence = model_confidences(model,dtest)
  # roc_auc_curve(y_test, confidence)
  # feature_importance(model, max_num_features=100)
  # print(f'Feature Count: {len(X_INPUTS_MLB)}')
  shap_values(model, x_test)
  # visualize_one_tree(model)
  pass