import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

home_rename = {
  'homeTeam': 'team',
  'awayTeam': 'opponent',
  'homeScore': 'score',
  'awayScore': 'opponentScore',
  'homeHeadCoach': 'headCoach',
  'awayHeadCoach': 'opponentHeadCoach',
  'homeForwardAverage': 'forwardAverage',
  'homeDefenseAverage': 'defenseAverage',
  'homeGoalieAverage': 'goalieAverage',
  'awayForwardAverage': 'opponentForwardAverage',
  'awayDefenseAverage': 'opponentDefenseAverage',
  'awayGoalieAverage': 'opponentGoalieAverage',
  'homeForwardAverageAge': 'forwardAverageAge',
  'homeDefenseAverageAge': 'defenseAverageAge',
  'homeGoalieAverageAge': 'goalieAverageAge',
  'awayForwardAverageAge': 'opponentForwardAverageAge',
  'awayDefenseAverageAge': 'opponentDefenseAverageAge',
  'awayGoalieAverageAge': 'opponentGoalieAverageAge',
  'winner': 'win',
  'winnerB': 'winB',
}
away_rename = {
  'homeTeam': 'opponent',
  'awayTeam': 'team',
  'homeScore': 'opponentScore',
  'awayScore': 'score',
  'homeHeadCoach': 'opponentHeadCoach',
  'awayHeadCoach': 'headCoach',
  'homeForwardAverage': 'opponentForwardAverage',
  'homeDefenseAverage': 'opponentDefenseAverage',
  'homeGoalieAverage': 'opponentGoalieAverage',
  'awayForwardAverage': 'forwardAverage',
  'awayDefenseAverage': 'defenseAverage',
  'awayGoalieAverage': 'goalieAverage',
  'homeForwardAverageAge': 'opponentForwardAverageAge',
  'homeDefenseAverageAge': 'opponentDefenseAverageAge',
  'homeGoalieAverageAge': 'opponentGoalieAverageAge',
  'awayForwardAverageAge': 'forwardAverageAge',
  'awayDefenseAverageAge': 'defenseAverageAge',
  'awayGoalieAverageAge': 'goalieAverageAge',
  'winner': 'win',
  'winnerB': 'winB',
}

team_score = {
  1: {'team': 'NJD', 'winB': 0.6206896551724138, 'lossB': 0.6206896551724138, 'score': 0.6206896551724138, 'id': 1}, 
  2: {'team': 'NYI', 'winB': 0.6551724137931034, 'lossB': 0.6551724137931034, 'score': 0.6551724137931034, 'id': 2}, 
  3: {'team': 'NYR', 'winB': 0.6271186440677966, 'lossB': 0.6271186440677966, 'score': 0.6271186440677966, 'id': 3}, 
  4: {'team': 'PHI', 'winB': 0.6, 'lossB': 0.6, 'score': 0.6, 'id': 4}, 
  5: {'team': 'PIT', 'winB': 0.6, 'lossB': 0.6, 'score': 0.6, 'id': 5}, 
  6: {'team': 'BOS', 'winB': 0.6333333333333333, 'lossB': 0.6333333333333333, 'score': 0.6333333333333333, 'id': 6}, 
  7: {'team': 'BUF', 'winB': 0.6101694915254238, 'lossB': 0.6101694915254238, 'score': 0.6101694915254238, 'id': 7}, 
  8: {'team': 'MTL', 'winB': 0.6440677966101694, 'lossB': 0.6440677966101694, 'score': 0.6440677966101694, 'id': 8}, 
  9: {'team': 'OTT', 'winB': 0.6491228070175439, 'lossB': 0.6491228070175439, 'score': 0.6491228070175439, 'id': 9}, 
  10: {'team': 'TOR', 'winB': 0.6379310344827587, 'lossB': 0.6379310344827587, 'score': 0.6379310344827587, 'id': 10}, 
  12: {'team': 'CAR', 'winB': 0.6440677966101694, 'lossB': 0.6440677966101694, 'score': 0.6440677966101694, 'id': 12}, 
  13: {'team': 'FLA', 'winB': 0.7241379310344828, 'lossB': 0.7241379310344828, 'score': 0.7241379310344828, 'id': 13}, 
  14: {'team': 'TBL', 'winB': 0.6229508196721312, 'lossB': 0.6229508196721312, 'score': 0.6229508196721312, 'id': 14}, 
  15: {'team': 'WSH', 'winB': 0.603448275862069, 'lossB': 0.603448275862069, 'score': 0.603448275862069, 'id': 15}, 
  16: {'team': 'CHI', 'winB': 0.7457627118644068, 'lossB': 0.7457627118644068, 'score': 0.7457627118644068, 'id': 16}, 
  17: {'team': 'DET', 'winB': 0.576271186440678, 'lossB': 0.5423728813559322, 'score': 0.5593220338983051, 'id': 17}, 
  18: {'team': 'NSH', 'winB': 0.6166666666666667, 'lossB': 0.5833333333333334, 'score': 0.6000000000000001, 'id': 18}, 
  19: {'team': 'STL', 'winB': 0.5862068965517241, 'lossB': 0.5862068965517241, 'score': 0.5862068965517241, 'id': 19}, 
  20: {'team': 'CGY', 'winB': 0.603448275862069, 'lossB': 0.603448275862069, 'score': 0.603448275862069, 'id': 20}, 
  21: {'team': 'COL', 'winB': 0.6610169491525424, 'lossB': 0.6610169491525424, 'score': 0.6610169491525424, 'id': 21}, 
  22: {'team': 'EDM', 'winB': 0.6785714285714286, 'lossB': 0.6785714285714286, 'score': 0.6785714285714286, 'id': 22}, 
  23: {'team': 'VAN', 'winB': 0.65, 'lossB': 0.65, 'score': 0.65, 'id': 23}, 
  24: {'team': 'ANA', 'winB': 0.6896551724137931, 'lossB': 0.6896551724137931, 'score': 0.6896551724137931, 'id': 24}, 
  25: {'team': 'DAL', 'winB': 0.6666666666666666, 'lossB': 0.6666666666666666, 'score': 0.6666666666666666, 'id': 25}, 
  26: {'team': 'LAK', 'winB': 0.6140350877192983, 'lossB': 0.6140350877192983, 'score': 0.6140350877192983, 'id': 26}, 
  28: {'team': 'SJS', 'winB': 0.75, 'lossB': 0.75, 'score': 0.75, 'id': 28}, 
  29: {'team': 'CBJ', 'winB': 0.7017543859649122, 'lossB': 0.7017543859649122, 'score': 0.7017543859649122, 'id': 29}, 
  30: {'team': 'MIN', 'winB': 0.6101694915254238, 'lossB': 0.6101694915254238, 'score': 0.6101694915254238, 'id': 30}, 
  52: {'team': 'WPG', 'winB': 0.5964912280701754, 'lossB': 0.5964912280701754, 'score': 0.5964912280701754, 'id': 52}, 
  53: {'team': 'ARI', 'winB': 0.6724137931034483, 'lossB': 0.6724137931034483, 'score': 0.6724137931034483, 'id': 53}, 
  54: {'team': 'VGK', 'winB': 0.6206896551724138, 'lossB': 0.6206896551724138, 'score': 0.6206896551724138, 'id': 54}, 
  55: {'team': 'SEA', 'winB': 0.6379310344827587, 'lossB': 0.6379310344827587, 'score': 0.6379310344827587, 'id': 55},
}

def team_lookup(db):
  Teams = db['dev_teams']
  teams = list(Teams.find({},{'_id': 0}))
  team_lookup = {}
  for team in teams:
    team_lookup[team['id']] = team
  return team_lookup

# class XGBWrapper:
#   def __init__(self, bst):
#     self.bst = bst

#   def fit(self, X, y):
#     pass

#   # def predict_proba(self, X):
#   #   ddata = xgb.DMatrix(X)
#   #   # Return a 2-column array: one for each class
#   #   return np.column_stack((1 - self.bst.predict(ddata), self.bst.predict(ddata)))

#   def predict_proba(self, X):
#     ddata = xgb.DMatrix(X)
#     # Assuming binary classification and bst.predict returns probability of positive class
#     pred_proba = self.bst.predict(ddata)
#     # Return a 2-column array: one for negative class (1 - prob) and one for positive class (prob)
#     return np.vstack((1-pred_proba, pred_proba)).T

class XGBWrapper(BaseEstimator, ClassifierMixin):
  def __init__(self, params=None, num_round=100):
    self.params = params if params is not None else {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    self.num_round = num_round
    self.bst = None

  def fit(self, X, y):
      dtrain = xgb.DMatrix(X, label=y)
      self.bst = xgb.train(self.params, dtrain, self.num_round)
      self.classes_ = unique_labels(y)
      return self

  def predict_proba(self, X):
    ddata = xgb.DMatrix(X)
    # Assuming binary classification and bst.predict returns probability of positive class
    pred_proba = self.bst.predict(ddata)
    # Return a 2-column array: one for negative class (1 - prob) and one for positive class (prob)
    return np.vstack((1-pred_proba, pred_proba)).T

class XGBWrapperInverse:
  def __init__(self, bst):
    self.bst = bst

  def fit(self, X, y):
    pass

  def predict_proba(self, X):
    ddata = xgb.DMatrix(X)
    # Return a 2-column array: one for each class
    return np.column_stack((self.bst.predict(ddata), 1 - self.bst.predict(ddata)))