import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import load
from pages.mlb.inputs import X_INPUTS_MLB, X_INPUTS_MLB_S, ENCODE_COLUMNS, mlb_training_input, mlb_test_input
from pages.mlb.mlb_helpers import team_lookup, away_rename, home_rename
from constants.constants import MLB_VERSION, FILE_MLB_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import pandas as pd
import numpy as np
from util.torch_helpers import HingeLoss, FocalLoss, binary_accuracy, errorAnalysis
from util.torch_layers import MemoryModule, NoiseInjection
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]
mlb_db = client["mlb"]

TEST_SEASONS = [
  2023,
  2022,
  2021,
]
SEASONS = [
  2023,
  # 2022,
  # 2021,
  # 2020,
  # 2019,
  # 2018,
  # 2017,
  # 2016,

  # 2015,
  # 2014,
  # 2013,
  # 2012,
  # 2011,
  # 2010,
  # 2009,
  # 2008,
  # 2007,
  # 2006,
  # 2005,
  # 2004,
  # 2003,
  # 2002,
  # 2001,
  # 2000,
]

USE_X_INPUTS_MLB = X_INPUTS_MLB

NUM_EPOCHS = 100
BATCH_SIZE = 1
MAX_SEQUENCE_LENGTH = 100
NUM_WORKERS = 4
LR=0.001
L2=1e-4
INPUT_DIM = len(USE_X_INPUTS_MLB)
OUTPUT_DIM = 1

N = 10

USE_TEST_SPLIT = True

OUTPUT = 'winner'



def fe(x):
  principalComponentAnalysis(x)

def principalComponentAnalysis(x):
  # Initialize PCA
  pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization purposes

  # Fit PCA on the dataset
  x_r = pca.fit_transform(x)
  x_r = np.array(x_r)
  # Plot the PCA-transformed version of the dataset
  plt.figure()

  plt.scatter(x_r[:,0], x_r[:,1], alpha=.8)
  plt.legend(loc='best', shadow=False, scatterpoints=1)
  plt.title('PCA of MLB dataset')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.show()



if __name__ == '__main__':
  x, y = mlb_training_input(SEASONS,encode=True,inputs=USE_X_INPUTS_MLB,outputs=[OUTPUT])
  fe(x)