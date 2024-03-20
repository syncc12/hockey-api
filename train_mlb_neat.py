import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

# from pymongo import MongoClient
# from sklearn.model_selection import train_test_split
from joblib import load
from pages.mlb.inputs import X_INPUTS_MLB, X_INPUTS_MLB_S, ENCODE_COLUMNS, mlb_training_input, mlb_test_input
# from pages.mlb.mlb_helpers import team_lookup, away_rename, home_rename
# from constants.constants import MLB_VERSION, FILE_MLB_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import pandas as pd
import numpy as np
# import time
import neat


# client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
# db = client["hockey"]
# mlb_db = client["mlb"]

TEST_SEASONS = [
  2023,
  2022,
  2021,
]
SEASONS = [
  2023,
  2022,
  2021,
  2020,
  2019,
  2018,
  2017,
  2016,
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

USE_X_INPUTS_MLB = X_INPUTS_MLB_S

NUM_EPOCHS = 50
BATCH_SIZE = 16
NUM_WORKERS = 4
LR=0.001
L2=1e-4
INPUT_DIM = len(USE_X_INPUTS_MLB)
OUTPUT_DIM = 1

USE_TEST_SPLIT = True

OUTPUT = 'winner'


def eval_genomes(genomes, config):
  for genome_id, genome in genomes:
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    correct_predictions = 0
    total_predictions = 0
    for xi, xo in zip(x_train, y_train):
      output = net.activate(xi)
      predicted_label = 1 if output[0] > 0.5 else 0
      correct_predictions += int(predicted_label == xo[0])
      total_predictions += 1
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    distance = abs(accuracy - 0.5)
    inverse = True if accuracy < 0.5 else False
    genome.fitness = distance
    print(f'Genome ID: {genome_id} Accuracy: {(accuracy*100):.2f}% Distance: {distance:.2f} {"Inverse" if inverse else ""}')

def train():
  # Load configuration.
  config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      'neat/config-feedforward.txt')

  # Create the population, which is the top-level object for a NEAT run.
  p = neat.Population(config)

  # Add a stdout reporter to show progress in the terminal.
  p.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  p.add_reporter(stats)

  # Run for up to 50 generations.
  winner = p.run(eval_genomes, 50)

  # Display the winning genome.
  print('\nBest genome:\n{!s}'.format(winner))






if __name__ == '__main__':
  # teamLookup = team_lookup(mlb_db,only_active_mlb=True)
  print(f'Input Length: {len(USE_X_INPUTS_MLB)}')
  TRAINING_DATA = mlb_training_input(SEASONS)
  data = pd.DataFrame(TRAINING_DATA)
  for column in ENCODE_COLUMNS:
    encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
    data = data[data[column] != -1]
    data[column] = encoder.transform(data[column])

  x_train = data [USE_X_INPUTS_MLB].to_numpy()
  y_train = data [[OUTPUT]].to_numpy()
  # if USE_TEST_SPLIT:
  #   x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
  
  # if not USE_TEST_SPLIT:
  #   TEST_DATA = mlb_test_input(TEST_SEASONS)
  #   test_data = pd.DataFrame(TEST_DATA)
  #   for column in ENCODE_COLUMNS:
  #     encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
  #     test_data = test_data[test_data[column] != -1]
  #     test_data[column] = encoder.transform(test_data[column])

  #   x_test = test_data [USE_X_INPUTS_MLB].to_numpy()
  #   y_test = test_data [[OUTPUT]].to_numpy()

  train()