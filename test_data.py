import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
import os
from joblib import dump
from util.training_data import season_test_data, season_training_data_projectedLineup
from constants.constants import VERSION, FILE_VERSION, PROJECTED_LINEUP_VERSION, PROJECTED_LINEUP_FILE_VERSION

PROJECTED_LINEUP = True

USE_VERSION = PROJECTED_LINEUP_VERSION if PROJECTED_LINEUP else VERSION
USE_FILE_VERSION = PROJECTED_LINEUP_FILE_VERSION if PROJECTED_LINEUP else FILE_VERSION

dir_path = f'training_data/v{USE_VERSION}'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

tdList = os.listdir(f'training_data/v{USE_VERSION}')

LATEST_SEASON = 20232024

if __name__ == '__main__':

  if PROJECTED_LINEUP:
    result = season_training_data_projectedLineup(LATEST_SEASON)
    dump(result,f'test_data/test_data_v{USE_FILE_VERSION}_projectedLineup.joblib')
  else:
    result = season_test_data(LATEST_SEASON)
    dump(result,f'test_data/test_data_v{USE_FILE_VERSION}.joblib')

  print('Test Data Collected')