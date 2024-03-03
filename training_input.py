import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from joblib import load
import numpy as np
from constants.constants import VERSION, FILE_VERSION, TEST_DATA_FILE_VERSION
import pandas as pd

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
  # 20172018,
  20182019,
  20192020,
  20202021,
  20212022,
  20222023,
]

def training_input(seasons, use_default_seasons=False):
  if use_default_seasons:
    seasons = SEASONS
  training_data = np.concatenate([load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib') for season in seasons]).tolist()
  print('Seasons Loaded')
  return training_data

def test_input(inputs,outputs,season=False,no_format=False):
  if season:
    if isinstance(season, list):
      test_data = np.concatenate([load(f'test_data/test_data_v{TEST_DATA_FILE_VERSION}.joblib') if s == 20232024 else load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{s}.joblib') for s in season]).tolist()
      if no_format:
        return test_data
    else:
      test_data = load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{season}.joblib')
      if no_format:
        return test_data
  else:
    test_data = load(f'test_data/test_data_v{TEST_DATA_FILE_VERSION}.joblib')
    if no_format:
      return test_data
  test_data = pd.DataFrame(test_data)
  x_test = test_data [inputs].to_numpy()
  y_test = test_data [outputs].to_numpy()
  return x_test, y_test