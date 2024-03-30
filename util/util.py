import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from joblib import dump, load
import pandas as pd
import numpy as np
import os
from constants.inputConstants import X_INPUTS, Y_OUTPUTS, X_INPUTS_T
from constants.constants import XGB_TEAM_FILE_VERSION, TEAM_FILE_VERSION, LGBM_TEAM_FILE_VERSION
from util.team_helpers import away_rename, home_rename, team_score, team_score_lgbm, team_spread_score, team_covers_score, TEAM_IDS
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import warnings
from datetime import datetime, timedelta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

def now(offset=0):
  now_map = {
    'now': 0,
    'today': 0,
    'yesterday': -1,
    'tomorrow': 1,
  }
  if offset in now_map:
    offset = now_map[offset]
  current_date = datetime.now()
  return (current_date + timedelta(days=offset)).strftime('%Y-%m-%d')

# Step 3: Define a 'dummy' classifier that simply returns precomputed probabilities
class CalibrationClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, probs):
    self.probs = probs

  def fit(self, X, y):
    # No fitting necessary, as probabilities are precomputed
    self.classes_ = np.unique(y)
    return self

  def predict_proba(self, X):
    return self.probs

