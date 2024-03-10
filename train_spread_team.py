import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import X_INPUTS_T, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, TEST_DATA_VERSION,TEST_DATA_FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, TEAM_VERSION, TEAM_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from util.helpers import team_lookup
from training_input import training_input, test_input
from util.team_helpers import away_rename, home_rename, franchise_map

# scikit-learn==1.3.2

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
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

OUTPUT = 'spread'

MAX_ITER = 1000
C = 10.0
TOL = 1e-4
SOLVER = 'liblinear'

TRIAL = True
DRY_RUN = False
TEAM = False

def train(db,dtrain,dtest,team_name='',params={},trial=False,output=OUTPUT):

  # class_counts = dtrain['data'].value_counts(output)
  # class_max = max(list(class_counts))
  # class_weights = {k: round((1/(v/class_max))) for k, v in class_counts.items()}

  model = LogisticRegression(
    max_iter=MAX_ITER,
    C=C,
    tol=TOL,
    solver=SOLVER,
    class_weight=params,
    random_state=RANDOM_STATE
  )
  # model = LinearRegression()
  model.fit(dtrain['x'], dtrain['y'])

  preds = model.predict(dtest['x'])
  y_test = dtest['y']
  predictions = [round(i) for i in preds]
  accuracy = accuracy_score(y_test, predictions)
  if not trial:
    # cm = confusion_matrix(y_test, predictions)
    model_data = f"{team_name} {output} Accuracy: {'%.2f%%' % (accuracy * 100.0)} | {dtest['len']}"
    # cm = confusion_matrix(y_test, preds)
    # print('Confusion Matrix:')
    # print(cm)
    print(model_data)

  if not DRY_RUN and not trial:
    TrainingRecords = db['dev_training_records']

    timestamp = time.time()
    TrainingRecords.insert_one({
      'savedAt': timestamp,
      'version': VERSION,
      'XGBVersion': TEAM_VERSION,
      'testDataVersion': TEST_DATA_VERSION,
      'inputs': X_INPUTS_T,
      'outputs': OUTPUT,
      'randomState': RANDOM_STATE,
      'startingSeason': START_SEASON,
      'finalSeason': END_SEASON,
      'seasons': SEASONS,
      'team': team_name,
      'teamId': team,
      'model': 'Logistic Regression',
      'file': 'train_spread_team.py',
      'max_iter': MAX_ITER,
      'C': C,
      'tol': TOL,
      'solver': SOLVER,
      'params': {str(k):v for k,v in params.items()},
      'accuracies': {
        output: {
          'accuracy': accuracy,
        },
      },
    })

    # path_accuracy = ('%.2f%%' % (accuracy * 100.0)).replace('.','_')
    # save_path = f'models/nhl_ai_v{XGB_FILE_VERSION}_xgboost_{team_name}_{OUTPUT}_{path_accuracy}.joblib'
    save_path = f'models/nhl_ai_v{TEAM_FILE_VERSION}_team{team}_{output}.joblib'
    dump(model, save_path)
  return accuracy

if __name__ == '__main__':
  teamLookup = team_lookup(db)
  TRAINING_DATA = training_input(SEASONS)
  data1 = pd.DataFrame(TRAINING_DATA)
  data2 = pd.DataFrame(TRAINING_DATA)

  data1.rename(columns=home_rename, inplace=True)
  data2.rename(columns=away_rename, inplace=True)
  data = pd.concat([data1, data2], axis=0)
  data.reset_index(drop=True, inplace=True)
  teams = data.groupby('team')

  if TEAM:
    teams = [(TEAM, teams.get_group(TEAM))]

  dtrains = {}
  dtests = {}
  for team, team_data in teams:
    x_train = team_data [X_INPUTS_T]
    y_train = team_data [[OUTPUT]].values.ravel()
    dtrains[team] = {'x':x_train,'y':y_train,'len':len(x_train),'data':team_data}

  TEST_DATA = test_input(X_INPUTS_T,[OUTPUT],no_format=True)
  test_data1 = pd.DataFrame(TEST_DATA)
  test_data2 = pd.DataFrame(TEST_DATA)
  test_data1.rename(columns=home_rename, inplace=True)
  test_data2.rename(columns=away_rename, inplace=True)
  test_data = pd.concat([test_data1, test_data2], axis=0)
  test_data.reset_index(drop=True, inplace=True)
  test_teams = test_data.groupby('team')

  if TEAM:
    test_teams = [(TEAM, test_teams.get_group(TEAM))]

  for team, team_data in test_teams:
    x_test = team_data [X_INPUTS_T]
    y_test = team_data [[OUTPUT]].values.ravel()
    dtests[team] = {'x':x_test,'y':y_test,'len':len(x_test),'data':team_data}


  accuracies = []
  if TRIAL:
    best = {}
    for i, (team, dtrain) in enumerate(dtrains.items()):
      best[team] = {'accuracy': 0.0}
      if team in franchise_map:
        team = franchise_map[team]
      team_name = teamLookup[team]['abbrev']
      for w1 in range(1, 51, 1):
        for w2 in range(1, 51, 1):
          for w3 in range(1, 51, 1):
            params = {
              1: w1,
              2: w2,
              3: w3,
            }
            accuracy = train(db,dtrain,dtests[team],params=params,trial=True)
            if accuracy > best[team]['accuracy']:
              best[team] = {
                'accuracy': accuracy,
                'params': params,
              }
            p_w1 = str(w1).ljust(3)
            p_w2 = str(w2).ljust(3)
            p_w3 = str(w3).ljust(3)
            p_w1_best = str(best[team]['params'][1]).ljust(3)
            p_w2_best = str(best[team]['params'][2]).ljust(3)
            p_w3_best = str(best[team]['params'][3]).ljust(3)
            p_accuracy = f'{team_name} {OUTPUT} Accuracy:{((accuracy)*100):.2f}%|1:{p_w1}|2:{p_w2}|3:{p_w3}'
            p_best = f'Best: Accuracy:{(best[team]["accuracy"]*100):.2f}%|1:{p_w1_best}|2:{p_w2_best}|3:{p_w3_best}'
            print(f'[{i+1}/{len(dtrains)}] {p_accuracy}||{p_best}')
      accuracies.append(best[team]['accuracy'])
      train(db,dtrain,dtests[team],team_name=team_name,params=best[team]['params'],trial=False)
    print('Average Accuracy:', f'{round((np.mean(accuracies)*100),2)}%')
  else:
    # print('Inputs:', X_INPUTS_T)
    print('Output:', OUTPUT)
    for team, dtrain in dtrains.items():
      if team in franchise_map:
        team = franchise_map[team]
      dtest = dtests[team]
      team_name = teamLookup[team]['abbrev']
      accuracy = train(db,dtrain,dtest,team_name=team_name)
      accuracies.append(accuracy)
    print('Average Accuracy:', f'{round((np.mean(accuracies)*100),2)}%')