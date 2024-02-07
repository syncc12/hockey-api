import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
import time

def pipeline():
  pass
