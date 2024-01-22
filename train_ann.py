import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from joblib import load, dump
from constants.inputConstants import X_INPUTS_ANN, Y_OUTPUTS, AWAY_FORWARD_INPUTS, AWAY_DEFENSE_INPUTS, AWAY_GOALIE_INPUTS, HOME_FORWARD_INPUTS, HOME_DEFENSE_INPUTS, HOME_GOALIE_INPUTS, AWAY_FORWARD_AGE_INPUTS, AWAY_DEFENSE_AGE_INPUTS, AWAY_GOALIE_AGE_INPUTS, AWAY_GOALIE_CATCHES_INPUTS, HOME_FORWARD_AGE_INPUTS, HOME_DEFENSE_AGE_INPUTS, HOME_GOALIE_AGE_INPUTS, HOME_GOALIE_CATCHES_INPUTS
from constants.constants import VERSION, RANDOM_STATE, FILE_VERSION
from tensorflow.keras.preprocessing.sequence import pad_sequences

EMBEDDING_SIZE = 10
N_FORWARDS = 13
N_DEFENSE = 7
N_SKATERS = N_FORWARDS + N_DEFENSE
N_GOALIES = 2
N_PLAYERS = N_SKATERS + N_GOALIES
N_HEAD_COACHES = 1
N_REFS = 2
N_LINESMEN = 1

training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')


def formatData(data):
  # x_id = data.loc[:,'id'].to_numpy()
  x_season = data.loc[:,'season'].to_numpy()
  x_gameType = data.loc[:,'gameType'].to_numpy()
  x_venue = data.loc[:,'venueT'].to_numpy()
  x_neutralSite = data.loc[:,'neutralSite'].to_numpy()
  x_homeTeam = data.loc[:,'homeTeam'].to_numpy()
  x_awayTeam = data.loc[:,'awayTeam'].to_numpy()
  # x_startTime = data.loc[:,'startTime'].to_numpy()
  # x_date = data.loc[:,'date'].to_numpy()
  x_awayHeadCoach = data.loc[:,'awayHeadCoachT'].to_numpy()
  x_homeHeadCoach = data.loc[:,'homeHeadCoachT'].to_numpy()
  x_refs = data.loc[:,['ref1T','ref2T']].to_numpy()
  x_linesmen = data.loc[:,['linesman1T','linesman2T']].to_numpy()
  # x_linesmen = data.loc[:,'linesman1T'].to_numpy()
  x_awayForwards = data.loc[:,AWAY_FORWARD_INPUTS].to_numpy()
  x_awayDefense = data.loc[:,AWAY_DEFENSE_INPUTS].to_numpy()
  x_awayStartingGoalie = data.loc[:,'awayStartingGoalie'].to_numpy()
  x_awayStartingGoalieCatches = data.loc[:,'awayStartingGoalieCatchesT'].to_numpy()
  x_awayBackupGoalie = data.loc[:,'awayBackupGoalie'].to_numpy()
  x_awayBackupGoalieCatches = data.loc[:,'awayBackupGoalieCatchesT'].to_numpy()
  x_awayPlayerAges = data.loc[:,AWAY_FORWARD_AGE_INPUTS+AWAY_DEFENSE_AGE_INPUTS+AWAY_GOALIE_AGE_INPUTS].to_numpy()
  x_homeForwards = data.loc[:,HOME_FORWARD_INPUTS].to_numpy()
  x_homeDefense = data.loc[:,HOME_DEFENSE_INPUTS].to_numpy()
  x_homeStartingGoalie = data.loc[:,'homeStartingGoalie'].to_numpy()
  x_homeStartingGoalieCatches = data.loc[:,'homeStartingGoalieCatchesT'].to_numpy()
  x_homeBackupGoalie = data.loc[:,'homeBackupGoalie'].to_numpy()
  x_homeBackupGoalieCatches = data.loc[:,'homeBackupGoalieCatchesT'].to_numpy()
  x_homePlayerAges = data.loc[:,HOME_FORWARD_AGE_INPUTS+HOME_DEFENSE_INPUTS+HOME_GOALIE_AGE_INPUTS].to_numpy()

  out_array = [
    # x_id,
    x_season,
    x_gameType,
    x_venue,
    x_neutralSite,
    x_homeTeam,
    x_awayTeam,
    # x_startTime,
    # x_date,
    x_awayHeadCoach,
    x_homeHeadCoach,
    x_refs,
    x_linesmen,
    x_awayForwards,
    x_awayDefense,
    x_awayStartingGoalie,
    x_awayStartingGoalieCatches,
    x_awayBackupGoalie,
    x_awayBackupGoalieCatches,
    x_awayPlayerAges,
    x_homeForwards,
    x_homeDefense,
    x_homeStartingGoalie,
    x_homeStartingGoalieCatches,
    x_homeBackupGoalie,
    x_homeBackupGoalieCatches,
    x_homePlayerAges,
  ]

  print(out_array)

  return out_array

def train():
  data = pd.DataFrame(training_data)

  encoders = {}
  encoder_columns = [
    'venueT',
    'awayHeadCoachT',
    'homeHeadCoachT',
    'ref1T',
    'ref2T',
    'linesman1T',
    'linesman2T',
    'awayStartingGoalieCatchesT',
    'awayBackupGoalieCatchesT',
    'homeStartingGoalieCatchesT',
    'homeBackupGoalieCatchesT',
  ]
  for column in encoder_columns:
    encoder = LabelEncoder()
    # print(column,data[column])
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder
  
  season = Input(shape=(1,), name='season')
  gameType = Input(shape=(1,), name='gameType')
  neutralSite = Input(shape=(1,), name='neutralSite')

  # player_ages_layer = Dense(64, activation='relu')(player_ages)
  # goalie_catches_layer = Dense(64, activation='relu')(goalie_catches)
  season_layer = Dense(64, activation='relu')(season)
  gameType_layer = Dense(64, activation='relu')(gameType)
  neutralSite_layer = Dense(64, activation='relu')(neutralSite)

  # id_input = Input(shape=(1,), name='id')
  awayPlayerAges_input = Input(shape=(N_PLAYERS,), name='awayPlayerAges')
  homePlayerAges_input = Input(shape=(N_PLAYERS,), name='homePlayerAges')
  awayStartingGoalieCatches_input = Input(shape=(1,), name='awayStartingGoalieCatchesT')
  awayBackupGoalieCatches_input = Input(shape=(1,), name='awayBackupGoalieCatchesT')
  homeStartingGoalieCatches_input = Input(shape=(1,), name='homeStartingGoalieCatchesT')
  homeBackupGoalieCatches_input = Input(shape=(1,), name='homeBackupGoalieCatchesT')
  awayForwards = Input(shape=(N_FORWARDS,), name='awayForwards')
  awayDefense = Input(shape=(N_DEFENSE,), name='awayDefense')
  awayStartingGoalie_input = Input(shape=(1,), name='awayStartingGoalie')
  awayBackupGoalie_input = Input(shape=(1,), name='awayBackupGoalie')
  homeForwards = Input(shape=(N_FORWARDS,), name='homeForwards')
  homeDefense = Input(shape=(N_DEFENSE,), name='homeDefense')
  homeStartingGoalie_input = Input(shape=(1,), name='homeStartingGoalie')
  homeBackupGoalie_input = Input(shape=(1,), name='homeBackupGoalie')
  awayHeadCoach_input = Input(shape=(N_HEAD_COACHES,), name='awayHeadCoachT')
  homeHeadCoach_input = Input(shape=(N_HEAD_COACHES,), name='homeHeadCoachT')
  refs_input = Input(shape=(N_REFS,), name='refsT')
  linesmen_input = Input(shape=(N_LINESMEN,), name='linesmenT')
  venue_input = Input(shape=(1,), name='venueT')
  awayTeam_input = Input(shape=(1,), name='awayTeam')
  homeTeam_input = Input(shape=(1,), name='homeTeam')
  # date_input = Input(shape=(1,), name='date')
  # startTime_input = Input(shape=(1,), name='startTime')

  # id_embedding = Embedding(1, EMBEDDING_SIZE, input_length=1)(id_input)
  skater_embedding = Embedding(9000000, EMBEDDING_SIZE, input_length=1, name='skater_embedding')
  awayPlayerAges_embedding = Embedding(60, EMBEDDING_SIZE, input_length=1)(awayPlayerAges_input)
  homePlayerAges_embedding = Embedding(60, EMBEDDING_SIZE, input_length=1)(homePlayerAges_input)
  awayStartingGoalieCatches_embedding = Embedding(len(encoders['awayStartingGoalieCatchesT'].classes_)+1, EMBEDDING_SIZE, input_length=1)(awayStartingGoalieCatches_input)
  awayBackupGoalieCatches_embedding = Embedding(len(encoders['awayBackupGoalieCatchesT'].classes_)+1, EMBEDDING_SIZE, input_length=1)(awayBackupGoalieCatches_input)
  homeStartingGoalieCatches_embedding = Embedding(len(encoders['homeStartingGoalieCatchesT'].classes_)+1, EMBEDDING_SIZE, input_length=1)(homeStartingGoalieCatches_input)
  homeBackupGoalieCatches_embedding = Embedding(len(encoders['homeBackupGoalieCatchesT'].classes_)+1, EMBEDDING_SIZE, input_length=1)(homeBackupGoalieCatches_input)
  awayForwards_embedded = skater_embedding(awayForwards)
  awayDefense_embedded = skater_embedding(homeForwards)
  awayStartingGoalie_embedding = Embedding(9000000, EMBEDDING_SIZE, input_length=1)(awayStartingGoalie_input)
  awayBackupGoalie_embedding = Embedding(9000000, EMBEDDING_SIZE, input_length=1)(awayBackupGoalie_input)
  homeForwards_embedded = skater_embedding(awayDefense)
  homeDefense_embedded = skater_embedding(homeDefense)
  homeStartingGoalie_embedding = Embedding(9000000, EMBEDDING_SIZE, input_length=1)(homeStartingGoalie_input)
  homeBackupGoalie_embedding = Embedding(9000000, EMBEDDING_SIZE, input_length=1)(homeBackupGoalie_input)
  venue_embedding = Embedding(len(encoders['venueT'].classes_)+1, EMBEDDING_SIZE, input_length=1)(venue_input)
  homeTeam_embedding = Embedding(10000, EMBEDDING_SIZE, input_length=1)(homeTeam_input)
  awayTeam_embedding = Embedding(10000, EMBEDDING_SIZE, input_length=1)(awayTeam_input)
  # startTime_embedding = Embedding(1, EMBEDDING_SIZE, input_length=1)(startTime_input)
  # date_embedding = Embedding(30000000, EMBEDDING_SIZE, input_length=1)(date_input)
  awayHeadCoach_embedding = Embedding(len(encoders['awayHeadCoachT'].classes_)+1, EMBEDDING_SIZE, input_length=1)(awayHeadCoach_input)
  homeHeadCoach_embedding = Embedding(len(encoders['homeHeadCoachT'].classes_)+1, EMBEDDING_SIZE, input_length=1)(homeHeadCoach_input)
  refs_embedding = Embedding(len(encoders['ref1T'].classes_)+1, EMBEDDING_SIZE, input_length=1)(refs_input)
  linesmen_embedding = Embedding(len(encoders['linesman1T'].classes_)+1, EMBEDDING_SIZE, input_length=1)(linesmen_input)
  
  # id_flatten = Flatten()(id_embedding)
  venue_flatten = Flatten()(venue_embedding)
  homeTeam_flatten = Flatten()(homeTeam_embedding)
  awayTeam_flatten = Flatten()(awayTeam_embedding)
  # startTime_flatten = Flatten()(startTime_embedding)
  # date_flatten = Flatten()(date_embedding)
  awayForwards_flatten = Flatten()(awayForwards_embedded)
  awayDefense_flatten = Flatten()(awayDefense_embedded)
  awayStartingGoalie_flatten = Flatten()(awayStartingGoalie_embedding)
  awayBackupGoalie_flatten = Flatten()(awayBackupGoalie_embedding)
  awayStartingGoalieCatches_flatten = Flatten()(awayStartingGoalieCatches_embedding)
  awayBackupGoalieCatches_flatten = Flatten()(awayBackupGoalieCatches_embedding)
  awayPlayerAges_flatten = Flatten()(awayPlayerAges_embedding)
  homeDefense_flatten = Flatten()(homeDefense_embedded)
  homeForwards_flatten = Flatten()(homeForwards_embedded)
  homeStartingGoalie_flatten = Flatten()(homeStartingGoalie_embedding)
  homeBackupGoalie_flatten = Flatten()(homeBackupGoalie_embedding)
  homeStartingGoalieCatches_flatten = Flatten()(homeStartingGoalieCatches_embedding)
  homeBackupGoalieCatches_flatten = Flatten()(homeBackupGoalieCatches_embedding)
  homePlayerAges_flatten = Flatten()(homePlayerAges_embedding)
  awayHeadCoach_flatten = Flatten()(awayHeadCoach_embedding)
  homeHeadCoach_flatten = Flatten()(homeHeadCoach_embedding)
  refs_flatten = Flatten()(refs_embedding)
  linesmen_flatten = Flatten()(linesmen_embedding)


  concatenated = Concatenate()([
    # id_flatten,
    season_layer,
    gameType_layer,
    venue_flatten,
    neutralSite_layer,
    homeTeam_flatten,
    awayTeam_flatten,
    # startTime_flatten,
    # date_flatten,
    awayHeadCoach_flatten,
    homeHeadCoach_flatten,
    refs_flatten,
    linesmen_flatten,
    awayForwards_flatten,
    awayDefense_flatten,
    awayStartingGoalie_flatten,
    awayStartingGoalieCatches_flatten,
    awayBackupGoalie_flatten,
    awayBackupGoalieCatches_flatten,
    awayPlayerAges_flatten,
    homeForwards_flatten,
    homeDefense_flatten,
    homeStartingGoalie_flatten,
    homeStartingGoalieCatches_flatten,
    homeBackupGoalie_flatten,
    homeBackupGoalieCatches_flatten,
    homePlayerAges_flatten
  ])

  # Add some dense layers for learning
  dense1 = Dense(128, activation='relu')(concatenated)
  dropout1 = Dropout(0.5)(dense1)
  dense2 = Dense(64, activation='relu')(dropout1)
  output = Dense(10000, activation='softmax')(dense2)
  # output = Dense(1, activation='sigmoid')(dense2)

  model_inputs = [
    # id_input,
    season,
    gameType,
    venue_input,
    neutralSite,
    homeTeam_input,
    awayTeam_input,
    # startTime_input,
    # date_input,
    awayHeadCoach_input,
    homeHeadCoach_input,
    refs_input,
    linesmen_input,
    awayForwards,
    awayDefense,
    awayStartingGoalie_input,
    awayStartingGoalieCatches_input,
    awayBackupGoalie_input,
    awayBackupGoalieCatches_input,
    awayPlayerAges_input,
    homeForwards,
    homeDefense,
    homeStartingGoalie_input,
    homeStartingGoalieCatches_input,
    homeBackupGoalie_input,
    homeBackupGoalieCatches_input,
    homePlayerAges_input,
  ]
  
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model = Model(inputs=model_inputs, outputs=output)
  model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])
  
  # model.summary()

  x = data [X_INPUTS_ANN].values
  y = data ['winner'].values

  # print(data.loc[:,["awayForward1","awayForward2"]].to_numpy())
  x[x == -1] = 0
  x_train_nd, x_test_nd, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)


  x_train = formatData(pd.DataFrame(data=x_train_nd, columns=X_INPUTS_ANN))
  # x_test = formatData(pd.DataFrame(data=x_test_nd, columns=X_INPUTS_ANN))

  # print("Item at position 8:", x_train[8])
  # print("Shape of item at position 8:", np.shape(x_train[8]))

  for i in range(8,len(x_train)):
    x_train[i] = x_train[i].flatten()

  # print('len(training_data)',len(training_data))
  x_train_padded = pad_sequences(x_train, padding='post', maxlen=len(training_data))
  x_train = np.array(x_train_padded, dtype=np.float32)
  y_train = np.array(y_train, dtype=np.float32)

  # print('y',y)
  # print('x',x)
  # print('y_train', y_train)
  # print('x_train', x_train)
  # f = open('training_data/training_data_text.txt', 'w')
  # f.write(json.dumps({'data':x_train}))
  # f.close()
  print(x_train)
  model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

  # loss, accuracy = model.evaluate(x_test, y_test)
  # print(f"Test Loss: {loss}")
  # print(f"Test Accuracy: {accuracy}")

  dump(model,f'models/nhl_ai_v{VERSION}_ann.joblib')

train()