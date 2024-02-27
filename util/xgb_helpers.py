import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef

def aggregate_statistics():
  pass

def learning_rate_schedule(round, num_boost_round, initial_lr=0.1, final_lr=0.01):
  """
  Calculates the learning rate at a given round using a linear schedule.
  
  Parameters:
  - round: int, current round number.
  - num_boost_round: int, total number of boosting rounds.
  - initial_lr: float, initial learning rate.
  - final_lr: float, final learning rate.
  
  Returns:
  - lr: float, the learning rate for the current round.
  """
  if round >= num_boost_round:
    return final_lr
  else:
    lr = initial_lr - ((initial_lr - final_lr) / num_boost_round) * round
    return lr

def update_learning_rate(round, num_boost_round, initial_lr, final_lr):
  """
  Callback function to update learning rate.
  """
  def callback(env):
    lr = learning_rate_schedule(round, num_boost_round, initial_lr, final_lr)
    env.model.set_param('learning_rate', lr)
  return callback

class LearningRateScheduler(xgb.callback.TrainingCallback):
  def __init__(self, num_boost_round, initial_lr=0.1, final_lr=0.01):
    super().__init__()
    self.num_boost_round = num_boost_round
    self.initial_lr = initial_lr
    self.final_lr = final_lr
  
  def before_iteration(self, model, epoch, dtrain, evals=(), obj=None):
    """Adjust learning rate before each iteration. Updated to include missing 'obj' parameter."""
    lr = self.learning_rate_schedule(epoch)
    model.set_param('learning_rate', lr)
    return False  # Return False to indicate training should not stop

  def learning_rate_schedule(self, round):
    """Calculate the learning rate for the current round."""
    if round >= self.num_boost_round:
      return self.final_lr
    else:
      lr = self.initial_lr - ((self.initial_lr - self.final_lr) / self.num_boost_round) * round
      return lr

def mcc_eval(y_pred, dtrain):
  y_true = dtrain.get_label()
  # XGBoost provides predictions as continuous values, need to convert to binary (e.g., using 0.5 threshold)
  y_pred_binary = np.where(y_pred > 0.5, 1, 0)
  mcc = matthews_corrcoef(y_true, y_pred_binary)
  # Return a tuple (name of the metric, value)
  return 'MCC', mcc