import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

import json
from pymongo import MongoClient
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump, load
import pandas as pd
from multiprocessing import Pool
from util.training_data import season_training_data_projectedLineup
from constants.inputConstants import X_INPUTS_P, Y_OUTPUTS_P
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import time
from constants.constants import PROJECTED_LINEUP_VERSION, PROJECTED_LINEUP_FILE_VERSION, PROJECTED_LINEUP_TEST_DATA_VERSION, PROJECTED_LINEUP_TEST_DATA_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
from scipy.stats import mode


RE_PULL = False
N_ESTIMATORS = 100


def train_batch(reg, X, Y, n_splits=5):
  kf = KFold(n_splits=n_splits)
  for train_index, _ in kf.split(X):
    reg.fit(X[train_index], Y[train_index])
  return reg


def train(db, inData, inTestData):
  print('Inputs:', X_INPUTS_P)
  print('Outputs:', Y_OUTPUTS_P)

  data = pd.DataFrame(inData)
  test_data = pd.DataFrame(inTestData)
  x_train = data [X_INPUTS_P]
  y_train = data [Y_OUTPUTS_P]
  x_test = test_data [X_INPUTS_P]
  y_test = test_data [Y_OUTPUTS_P]

  # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

  base_rf = RandomForestRegressor(random_state=RANDOM_STATE)
  reg = MultiOutputRegressor(base_rf, n_jobs=1)
  
  # reg.fit(x_train,y_train)
  reg = train_batch(reg, x_train.values, y_train.values, n_splits=4)

  predictions = reg.predict(x_test)

  accuracies = {}
  for i in range(y_test.shape[1]):  # Iterate through each of the target variables
    target_name = y_test.columns[i]
    
    # Calculate metrics
    mse = mean_squared_error(y_test.iloc[:, i], predictions[:, i])
    mae = mean_absolute_error(y_test.iloc[:, i], predictions[:, i])
    r2 = r2_score(y_test.iloc[:, i], predictions[:, i])

    print(f"{target_name} Mean Squared Error: {mse}")
    print(f"{target_name} Mean Absolute Error: {mae}")
    print(f"{target_name} R-squared: {r2}")

    accuracies[target_name] = {
      'mse': mse,
      'mae': mae,
      'r2': r2,
    }


  TrainingRecords = db['dev_training_records']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'savedAt': timestamp,
    'lastTrainedId': inData[len(inData)-1]['id'],
    'version': PROJECTED_LINEUP_VERSION,
    'inputs': X_INPUTS_P,
    'outputs': Y_OUTPUTS_P,
    'randomState': RANDOM_STATE,
    # 'n_estimators': N_ESTIMATORS,
    'startingSeason': START_SEASON,
    'finalSeason': END_SEASON,
    'projectedLineup': True,
    'model': 'Multi-Class Random Forest Regressor',
    'file': 'train_mor_projectedLineup.py',
    'accuracies': {
      'projectedLineup': accuracies,
    },
  })

  # dump(reg, f'models/nhl_ai_v{FILE_VERSION}.joblib')
  dump(reg, f'models/nhl_ai_v{PROJECTED_LINEUP_FILE_VERSION}_mor_projectedLineup.joblib')


dir_path = f'training_data/v{PROJECTED_LINEUP_VERSION}'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

dir_path = f'training_data/v{PROJECTED_LINEUP_VERSION}/projected_lineup'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

tdList = os.listdir(f'training_data/v{PROJECTED_LINEUP_VERSION}/projected_lineup')

SKIP_SEASONS = [int(td.replace(f'training_data_v{PROJECTED_LINEUP_FILE_VERSION}_','').replace('_projectedLineup','').replace('.joblib','')) for td in tdList] if len(tdList) > 0 and not f'training_data_v{PROJECTED_LINEUP_FILE_VERSION}_projectedLineup.joblib' in os.listdir('training_data') else []

LATEST_SEASON = 20232024

if __name__ == '__main__':
  # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
  # client = MongoClient(db_url)
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
  db = client["hockey"]
  if RE_PULL:
    seasons = list(db["dev_seasons"].find(
      {
        'seasonId': {
          '$gte': START_SEASON,
          '$lte': END_SEASON,
        }
      },
      {'_id':0,'seasonId': 1}
    ))
    seasons = [int(season['seasonId']) for season in seasons]
    print(seasons)
    if (len(SKIP_SEASONS) > 0):
      for season in SKIP_SEASONS:
        seasons.remove(season)
      print(seasons)

    pool = Pool(processes=4)
    result = pool.map(season_training_data_projectedLineup,seasons)
    if len(SKIP_SEASONS) > 0:
      for skip_season in SKIP_SEASONS:
        season_data = load(f'training_data/v{PROJECTED_LINEUP_VERSION}/training_data_v{PROJECTED_LINEUP_FILE_VERSION}_{skip_season}_projectedLineup.joblib')
        result.append(season_data)
    result = np.concatenate(result).tolist()
    pool.close()
    dump(result,f'training_data/training_data_v{PROJECTED_LINEUP_FILE_VERSION}_projectedLineup.joblib')
    f = open('training_data/training_data_text.txt', 'w')
    f.write(json.dumps(result[len(result)-200:len(result)]))
  else:
    training_data_path = f'training_data/training_data_v{PROJECTED_LINEUP_FILE_VERSION}_projectedLineup.joblib'
    print(training_data_path)
    result = load(training_data_path)
    f = open('training_data/training_data_text.txt', 'w')
    f.write(json.dumps(result[len(result)-200:len(result)]))
  print('Games Collected')
  test_data = load(f'test_data/test_data_v{PROJECTED_LINEUP_TEST_DATA_FILE_VERSION}_projectedLineup.joblib')
  train(db, result, test_data)
