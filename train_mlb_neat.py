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

GENERATIONS = 400

generation = -1
overall_best = {
  "accuracy": 0,
  "distance": 0,
  "distance_accuracy": 0,
  "generation": 0,
  "genome_id": 0,
  "inverse": False,
}

def eval_genomes(genomes, config):
  global generation
  generation += 1
  genomes_len = len(genomes)
  generation_best = {
    "accuracy": 0,
    "distance": 0,
    "distance_accuracy": 0,
    "genome_id": 0,
    "inverse": False,
    "i": 0,
  }
  for i, (genome_id, genome) in enumerate(genomes):
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
    distance_accuracy = 0.5 + distance
    inverse = True if accuracy < 0.5 else False
    genome.fitness = distance
    if distance > overall_best["distance"]:
      overall_best["distance"] = distance
      overall_best["accuracy"] = accuracy
      overall_best["distance_accuracy"] = distance_accuracy
      overall_best["generation"] = generation
      overall_best["genome_id"] = genome_id
      overall_best["inverse"] = inverse
    if distance > generation_best["distance"]:
      generation_best["distance"] = distance
      generation_best["accuracy"] = accuracy
      generation_best["distance_accuracy"] = distance_accuracy
      generation_best["genome_id"] = genome_id
      generation_best["inverse"] = inverse
      generation_best["i"] = i
    
    p_current = f'[{str(generation).rjust(len(str(GENERATIONS-1)))}/{GENERATIONS-1}][{str(i).rjust(len(str(genomes_len)))}/{genomes_len}]GID:{str(genome_id).ljust(4)}|A:{(accuracy*100):.2f}%({(distance_accuracy*100):.2f}%)|D:{distance:.4f}|{"I" if inverse else " "}'
    p_generation_best = f'[{str(generation).rjust(len(str(GENERATIONS-1)))}][{str(generation_best["i"]).rjust(len(str(genomes_len)))}]GID:{str(generation_best["genome_id"]).ljust(4)}|A:{(generation_best["accuracy"]*100):.2f}%({(generation_best["distance_accuracy"]*100):.2f}%)|D:{generation_best["distance"]:.4f}|{"I" if generation_best["inverse"] else " "}'
    p_overall_best = f'[{str(overall_best["generation"]).rjust(len(str(GENERATIONS-1)))}]GID:{str(overall_best["genome_id"]).ljust(4)}|A:{(overall_best["accuracy"]*100):.2f}%({(overall_best["distance_accuracy"]*100):.2f}%)|D:{overall_best["distance"]:.4f}|{"I" if overall_best["inverse"] else " "}'
    print(f'{p_current}||{p_generation_best}||{p_overall_best}')

def train():
  # Load configuration.
  config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      'neat/mlb-config-feedforward.txt')

  # Create the population, which is the top-level object for a NEAT run.
  p = neat.Population(config)

  # Add a stdout reporter to show progress in the terminal.
  p.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  p.add_reporter(stats)

  winner = p.run(eval_genomes, GENERATIONS)

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