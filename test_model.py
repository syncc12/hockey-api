import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from joblib import load
import pandas as pd
from constants.inputConstants import X_INPUTS, X_INPUTS_P, Y_OUTPUTS_P
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, average_precision_score
from constants.constants import FILE_VERSION, PROJECTED_LINEUP_FILE_VERSION, RANDOM_STATE
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
from sklearn.inspection import permutation_importance
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

AVERAGE = None

PARAMS = {
  'max_depth': 23,  # the maximum depth of each tree
  'eta': 0.18,  # the training step for each iteration
  'objective': 'binary:logistic',  # binary classification
  'eval_metric': 'logloss'  # evaluation metric
}
NUM_BOOST_ROUND = 500

model = load(f'models/nhl_ai_v{FILE_VERSION}_xgboost_winnerB.joblib')
test_data = load(f'test_data/test_data_v{FILE_VERSION}.joblib')
# model = load(f'models/nhl_ai_v{PROJECTED_LINEUP_FILE_VERSION}_projectedLineup.joblib')
# test_data = load(f'test_data/test_data_v{PROJECTED_LINEUP_FILE_VERSION}_projectedLineup.joblib')

OUTPUT = 'winnerB'

data = pd.DataFrame(test_data)
# x = data [X_INPUTS_P]
# y = data [Y_OUTPUTS_P]
x = data [X_INPUTS]
y = data [[OUTPUT]].values.ravel()


# test_predictions = model.predict(x)

class XGBWrapper:
  def __init__(self, model):
    self.model = model

  def fit(self, X, y):
    # Ensure X is not already a DMatrix
    if not isinstance(X, xgb.DMatrix):
      X = xgb.DMatrix(X, label=y)
    self.model.fit(X, y)

  def predict(self, X):
    # Check if X is a DMatrix; if not, convert it
    if not isinstance(X, xgb.DMatrix):
      X = xgb.DMatrix(X)
    return self.model.predict(X)

  def score(self, X, y):
    # Generate predictions
    predictions = self.predict(X)
    # Calculate and return accuracy
    return accuracy_score(y, (predictions > 0.5).astype(int))

model_wrapper = XGBClassifier(use_label_encoder=False, eval_metric='logloss', params=PARAMS, num_boost_round=500, n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)
# model_wrapper = XGBWrapper(xgb_model)
model_wrapper.fit(x, y)

def accuracyScore(y_column, prediction, target_name=""):
  accuracy = accuracy_score(y_column, prediction)
  print(f"{target_name} Accuracy:", accuracy)

def balancedAccuracyScore(y_column, prediction, target_name=""):
  balanced_accuracy = balanced_accuracy_score(y_column, prediction)
  print(f"{target_name} Balanced Accuracy:", balanced_accuracy)

def precisionScore(y_column, prediction, target_name=""):
  precision = precision_score(y_column, prediction, average=AVERAGE)
  print(f"{target_name} Precision:", precision)

def averagePrecisionScore(y_column, prediction, target_name=""):
  average_precision = average_precision_score(y_column, prediction)
  print(f"{target_name} Average Precision:", average_precision)

def recallScore(y_column, prediction, target_name=""):
  recall = recall_score(y_column, prediction, average=AVERAGE)
  print(f"{target_name} Recall:", recall)

def f1Score(y_column, prediction, target_name=""):
  f1 = f1_score(y_column, prediction, average=AVERAGE)
  print(f"{target_name} F1:", f1)

def featureImportances(model):
  print("Feature Importances:")
  for i, estimator in enumerate(model.estimators_):
    print(f"Feature importances for output {X_INPUTS_P[i]}: {estimator.feature_importances_}")

def featureImportancesXGB(model):
  print("Feature Importances:")
  print(model.feature_importances_)

def permutationImportances(model, x, y):
  permutation_importances = permutation_importance(model, x, y, n_repeats=30, random_state=RANDOM_STATE)
  print("Permutation Importances:")
  for i in range(len(permutation_importances.importances_mean)):
    print(f"{X_INPUTS_P[i]}: {permutation_importances.importances_mean[i]}")

def permutationImportancesXGB(model, x, y):
  permutation_importances = permutation_importance(model, x, y, n_repeats=10, random_state=RANDOM_STATE)
  print("Permutation Importances:")
  for feature, importance in zip(x.columns, permutation_importances.importances_mean):
    print(f"{feature}: {importance}")

def crossValidationXGB(model, x, y):
  kfold = KFold(n_splits=10, random_state=RANDOM_STATE, shuffle=True)
  results = cross_val_score(model, x, y, cv=kfold)
  print("Cross Validation Results:", results)

def leaveOneOutCrossValidationXGB(model, x, y):
  loocv = LeaveOneOut()
  results = cross_val_score(model, x, y, cv=loocv, scoring='accuracy')
  print("Leave One Out Cross Validation Results:", results)

def featureImportancesXGB(model):
  print("Feature Importances:")
  feature_importances = model.feature_importances_
  for feature, importance in zip(x.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")
  
  # Plot feature importance
  plt.figure(figsize=(10, 6))
  plt.bar(x.columns, feature_importances)
  plt.xticks(rotation='vertical')
  plt.xlabel('Feature')
  plt.ylabel('Importance')
  plt.title('Feature Importance')
  plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
  plt.show()

def isList(inVariable):
  return isinstance(inVariable, list)

def test_model():
  # crossValidationXGB(model_wrapper, x, y)
  # leaveOneOutCrossValidationXGB(model_wrapper, x, y)
  permutationImportancesXGB(model_wrapper, x, y)
  featureImportancesXGB(model_wrapper)

if __name__ == "__main__":
  test_model()