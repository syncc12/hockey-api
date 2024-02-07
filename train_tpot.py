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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

GENERATIONS = 5
POPULATION_SIZE = 50
VERBOSITY = 2
N_JOBS = 4

OUTPUT = 'totalGoals'


def train(inData):
  data = pd.DataFrame(inData)
  x = data [X_INPUTS]
  y = data [[OUTPUT]].values.ravel()

  # Split the dataset into training and testing sets
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_STATE)

  tpot = TPOTClassifier(generations=GENERATIONS, population_size=POPULATION_SIZE, verbosity=VERBOSITY, n_jobs=N_JOBS, random_state=RANDOM_STATE)
  tpot.fit(x_train, y_train)

  y_pred = tpot.predict(x_test)
  print(f"{OUTPUT} Accuracy:", accuracy_score(y_test, y_pred))

  tpot.export(f'tpot_{OUTPUT}_pipeline.py')


training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
train(training_data)