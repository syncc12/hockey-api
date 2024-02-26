import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, TEST_DATA_VERSION,TEST_DATA_FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
import xgboost as xgb
from util.helpers import all_combinations
from itertools import combinations
import json
import optuna


SEASONS = [20162017, 20172018, 20182019, 20192020, 20202021, 20212022, 20222023]
ALL_SEASONS = all_combinations(SEASONS)

# print(ALL_SEASONS)
used_seasons = [','.join([str(season)[-2:] for season in seasons]) for seasons in ALL_SEASONS]
print(used_seasons)