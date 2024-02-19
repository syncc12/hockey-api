import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

import requests
import json
from pymongo import MongoClient
import math
from datetime import datetime
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from joblib import dump, load
import pandas as pd
import time
import flatdict
import featuretools as ft
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE


# training_data_path = f'training_data/training_data_v{FILE_VERSION}.joblib'
# print(training_data_path)
# result = load(training_data_path)

def flatten_dict(d, parent_key='', sep='_'):
  items = []
  for k, v in d.items():
    new_key = f"{parent_key}{sep}{k}" if parent_key else k
    if isinstance(v, dict):
      items.extend(flatten_dict(v, new_key, sep=sep).items())
    elif isinstance(v, list):
      for i, item in enumerate(v):
        if isinstance(item, dict):
          items.extend(flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
        else:
          items.append((f"{new_key}{sep}{i}", item))
    else:
      items.append((new_key, v))
  return dict(items)

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
Boxscores = db["dev_boxscores"]
Players = db['dev_players']


drop_columns = {
  '_id': 0,'gameState':0,'gameScheduleState':0,'period':0,'periodDescriptor':0,'awayTeam.sog':0,'awayTeam.faceoffWinningPctg':0,'awayTeam.powerPlayConversion':0,'awayTeam.pim':0,
  'awayTeam.hits':0,'awayTeam.blocks':0,'awayTeam.logo':0,'homeTeam.sog':0,'homeTeam.faceoffWinningPctg':0,'homeTeam.powerPlayConversion':0,'homeTeam.pim':0,'homeTeam.hits':0,
  'homeTeam.blocks':0,'homeTeam.logo':0,'clock':0,'boxscore.shotsByPeriod':0,'boxscore.gameReports':0,'gameOutcome':0,'gameVideo':0,'boxscore.linescore':0,
  'boxscore.playerByGameStats.awayTeam.forwards._id':0,'boxscore.playerByGameStats.awayTeam.defense._id':0,'boxscore.playerByGameStats.awayTeam.goalies._id':0,
  'boxscore.playerByGameStats.homeTeam.forwards._id':0,'boxscore.playerByGameStats.homeTeam.defense._id':0,'boxscore.playerByGameStats.homeTeam.goalies._id':0,
  'boxscore.playerByGameStats.awayTeam.forwards.name':0,'boxscore.playerByGameStats.awayTeam.defense.name':0,'boxscore.playerByGameStats.awayTeam.goalies.name':0,
  'boxscore.playerByGameStats.homeTeam.forwards.name':0,'boxscore.playerByGameStats.homeTeam.defense.name':0,'boxscore.playerByGameStats.homeTeam.goalies.name':0,
  'boxscore.playerByGameStats.awayTeam.forwards.goals':0,
  'boxscore.playerByGameStats.awayTeam.forwards.assists': 0,'boxscore.playerByGameStats.awayTeam.forwards.points': 0,'boxscore.playerByGameStats.awayTeam.forwards.plusMinus': 0,
  'boxscore.playerByGameStats.awayTeam.forwards.pim': 0,'boxscore.playerByGameStats.awayTeam.forwards.hits': 0,'boxscore.playerByGameStats.awayTeam.forwards.blockedShots': 0,
  'boxscore.playerByGameStats.awayTeam.forwards.powerPlayGoals': 0,'boxscore.playerByGameStats.awayTeam.forwards.powerPlayPoints': 0,
  'boxscore.playerByGameStats.awayTeam.forwards.shorthandedGoals': 0,'boxscore.playerByGameStats.awayTeam.forwards.shPoints': 0,'boxscore.playerByGameStats.awayTeam.forwards.shots': 0,
  'boxscore.playerByGameStats.awayTeam.forwards.faceoffs': 0,'boxscore.playerByGameStats.awayTeam.forwards.faceoffWinningPctg': 0,'boxscore.playerByGameStats.awayTeam.forwards.toi': 0,
  'boxscore.playerByGameStats.awayTeam.forwards.powerPlayToi': 0,'boxscore.playerByGameStats.awayTeam.forwards.shorthandedToi': 0,'boxscore.playerByGameStats.awayTeam.defense.goals':0,
  'boxscore.playerByGameStats.awayTeam.defense.assists': 0,'boxscore.playerByGameStats.awayTeam.defense.points': 0,'boxscore.playerByGameStats.awayTeam.defense.plusMinus': 0,
  'boxscore.playerByGameStats.awayTeam.defense.pim': 0,'boxscore.playerByGameStats.awayTeam.defense.hits': 0,'boxscore.playerByGameStats.awayTeam.defense.blockedShots': 0,
  'boxscore.playerByGameStats.awayTeam.defense.powerPlayGoals': 0,'boxscore.playerByGameStats.awayTeam.defense.powerPlayPoints': 0,
  'boxscore.playerByGameStats.awayTeam.defense.shorthandedGoals': 0,'boxscore.playerByGameStats.awayTeam.defense.shPoints': 0,'boxscore.playerByGameStats.awayTeam.defense.shots': 0,
  'boxscore.playerByGameStats.awayTeam.defense.faceoffs': 0,'boxscore.playerByGameStats.awayTeam.defense.faceoffWinningPctg': 0,'boxscore.playerByGameStats.awayTeam.defense.toi': 0,
  'boxscore.playerByGameStats.awayTeam.defense.powerPlayToi': 0,'boxscore.playerByGameStats.awayTeam.defense.shorthandedToi': 0,
  'boxscore.playerByGameStats.awayTeam.goalies.evenStrengthShotsAgainst':0,'boxscore.playerByGameStats.awayTeam.goalies.powerPlayShotsAgainst':0,
  'boxscore.playerByGameStats.awayTeam.goalies.shorthandedShotsAgainst':0,'boxscore.playerByGameStats.awayTeam.goalies.saveShotsAgainst':0,
  'boxscore.playerByGameStats.awayTeam.goalies.savePctg':0,'boxscore.playerByGameStats.awayTeam.goalies.evenStrengthGoalsAgainst':0,
  'boxscore.playerByGameStats.awayTeam.goalies.powerPlayGoalsAgainst':0,'boxscore.playerByGameStats.awayTeam.goalies.shorthandedGoalsAgainst':0,
  'boxscore.playerByGameStats.awayTeam.goalies.pim':0,'boxscore.playerByGameStats.awayTeam.goalies.goalsAgainst':0,'boxscore.playerByGameStats.awayTeam.goalies.toi':0,
  'boxscore.playerByGameStats.homeTeam.forwards.goals':0,'boxscore.playerByGameStats.homeTeam.forwards.assists': 0,'boxscore.playerByGameStats.homeTeam.forwards.points': 0,
  'boxscore.playerByGameStats.homeTeam.forwards.plusMinus': 0,'boxscore.playerByGameStats.homeTeam.forwards.pim': 0,'boxscore.playerByGameStats.homeTeam.forwards.hits': 0,
  'boxscore.playerByGameStats.homeTeam.forwards.blockedShots': 0,'boxscore.playerByGameStats.homeTeam.forwards.powerPlayGoals': 0,
  'boxscore.playerByGameStats.homeTeam.forwards.powerPlayPoints': 0,'boxscore.playerByGameStats.homeTeam.forwards.shorthandedGoals': 0,'boxscore.playerByGameStats.homeTeam.forwards.shPoints':0,
  'boxscore.playerByGameStats.homeTeam.forwards.shots': 0,'boxscore.playerByGameStats.homeTeam.forwards.faceoffs': 0,'boxscore.playerByGameStats.homeTeam.forwards.faceoffWinningPctg': 0,
  'boxscore.playerByGameStats.homeTeam.forwards.toi': 0,'boxscore.playerByGameStats.homeTeam.forwards.powerPlayToi': 0,'boxscore.playerByGameStats.homeTeam.forwards.shorthandedToi': 0,
  'boxscore.playerByGameStats.homeTeam.defense.goals':0,'boxscore.playerByGameStats.homeTeam.defense.assists': 0,'boxscore.playerByGameStats.homeTeam.defense.points': 0,
  'boxscore.playerByGameStats.homeTeam.defense.plusMinus': 0,'boxscore.playerByGameStats.homeTeam.defense.pim': 0,'boxscore.playerByGameStats.homeTeam.defense.hits': 0,
  'boxscore.playerByGameStats.homeTeam.defense.blockedShots': 0,'boxscore.playerByGameStats.homeTeam.defense.powerPlayGoals': 0,'boxscore.playerByGameStats.homeTeam.defense.powerPlayPoints':0,
  'boxscore.playerByGameStats.homeTeam.defense.shorthandedGoals': 0,'boxscore.playerByGameStats.homeTeam.defense.shPoints': 0,'boxscore.playerByGameStats.homeTeam.defense.shots': 0,
  'boxscore.playerByGameStats.homeTeam.defense.faceoffs': 0,'boxscore.playerByGameStats.homeTeam.defense.faceoffWinningPctg': 0,'boxscore.playerByGameStats.homeTeam.defense.toi': 0,
  'boxscore.playerByGameStats.homeTeam.defense.powerPlayToi': 0,'boxscore.playerByGameStats.homeTeam.defense.shorthandedToi': 0,
  'boxscore.playerByGameStats.homeTeam.goalies.evenStrengthShotsAgainst':0,'boxscore.playerByGameStats.homeTeam.goalies.powerPlayShotsAgainst':0,
  'boxscore.playerByGameStats.homeTeam.goalies.shorthandedShotsAgainst':0,'boxscore.playerByGameStats.homeTeam.goalies.saveShotsAgainst':0,
  'boxscore.playerByGameStats.homeTeam.goalies.savePctg':0,'boxscore.playerByGameStats.homeTeam.goalies.evenStrengthGoalsAgainst':0,
  'boxscore.playerByGameStats.homeTeam.goalies.powerPlayGoalsAgainst':0,'boxscore.playerByGameStats.homeTeam.goalies.shorthandedGoalsAgainst':0,
  'boxscore.playerByGameStats.homeTeam.goalies.pim':0,'boxscore.playerByGameStats.homeTeam.goalies.goalsAgainst':0,'boxscore.playerByGameStats.homeTeam.goalies.toi':0,'venue._id':0,
  'awayTeam.name._id':0,'awayTeam._id':0,'homeTeam.name._id':0,'homeTeam._id':0,'boxscore.playerByGameStats.awayTeam._id':0,'boxscore.playerByGameStats.homeTeam._id':0,
  'boxscore.playerByGameStats._id':0,'boxscore.gameInfo.awayTeam.headCoach._id':0,'boxscore.gameInfo.awayTeam._id':0,'boxscore.gameInfo.homeTeam.headCoach._id':0,
  'boxscore.gameInfo.homeTeam._id':0,'boxscore.gameInfo._id':0,'boxscore._id':0,'__v':0,
}

keep_columns = [
  'id',
  'season',
  'gameType',
  'gameDate',
  'startTimeUTC',
  'easternUTCOffset',
  'venueUTCOffset',
  'winnerB',
  'winner',
  'venue',
  # 'venue.en',
  # 'venue.default',
  'awayTeam',
  # 'awayTeam.id',
  # 'awayTeam.name.en',
  # 'awayTeam.name.default',
  # 'awayTeam.abbrev',
  # 'awayTeam.score',
  'homeTeam',
  # 'homeTeam.id',
  # 'homeTeam.name.en',
  # 'homeTeam.name.default',
  # 'homeTeam.abbrev',
  # 'homeTeam.score',
  # 'boxscore',
  # 'boxscore.gameInfo.awayTeam.headCoach.en',
  # 'boxscore.gameInfo.awayTeam.headCoach.default',
  # 'boxscore.gameInfo.homeTeam.headCoach.en',
  # 'boxscore.gameInfo.homeTeam.headCoach.default',
]

# [
#   'id',
#   'season',
#   'gameType',
#   'gameDate',
#   'startTimeUTC',
#   'easternUTCOffset',
#   'venueUTCOffset',
#     'tvBroadcasts',
#   'winnerB',
#   'winner',
#   'venue.en',
#   'venue.default',
#   'awayTeam.id',
#   'awayTeam.name.en',
#   'awayTeam.name.default',
#   'awayTeam.abbrev',
#   'awayTeam.score',
#   'homeTeam.id',
#   'homeTeam.name.en',
#   'homeTeam.name.default',
#   'homeTeam.abbrev',
#   'homeTeam.score',
#     'boxscore.playerByGameStats.awayTeam.forwards',
#     'boxscore.playerByGameStats.awayTeam.defense',
#     'boxscore.playerByGameStats.awayTeam.goalies',
#     'boxscore.playerByGameStats.homeTeam.forwards',
#     'boxscore.playerByGameStats.homeTeam.defense',
#     'boxscore.playerByGameStats.homeTeam.goalies',
#     'boxscore.gameInfo.referees',
#     'boxscore.gameInfo.linesmen',
#   'boxscore.gameInfo.awayTeam.headCoach.en',
#   'boxscore.gameInfo.awayTeam.headCoach.default',
#     'boxscore.gameInfo.awayTeam.scratches',
#   'boxscore.gameInfo.homeTeam.headCoach.en',
#   'boxscore.gameInfo.homeTeam.headCoach.default',
#     'boxscore.gameInfo.homeTeam.scratches',
# ]


boxscores = list(Boxscores.find(
  {'season': 20232024},
  drop_columns
))
players = list(Players.find(
  {},
  {'_id':0,'playerSlug':0,'__v':0}
))

allAwayForwards = []
allAwayDefense = []
allAwayGoalies = []
allHomeForwards = []
allHomeDefense = []
allHomeGoalies = []
allAwayPlayers = []
allHomePlayers = []
allPlayers = []
for i in range(0,len(boxscores)):
  winnerB = 1 if boxscores[i]['awayTeam']['score'] > boxscores[i]['homeTeam']['score'] else 0
  winner = boxscores[i]['awayTeam']['id'] if boxscores[i]['awayTeam']['score'] > boxscores[i]['homeTeam']['score'] else boxscores[i]['homeTeam']['id']
  boxscores[i]['winnerB'] = winnerB
  boxscores[i]['winner'] = winner
  awayForwards = [player['playerId'] for player in boxscores[i]['boxscore']['playerByGameStats']['awayTeam']['forwards']]
  awayDefense = [player['playerId'] for player in boxscores[i]['boxscore']['playerByGameStats']['awayTeam']['defense']]
  awayGoalies = [player['playerId'] for player in boxscores[i]['boxscore']['playerByGameStats']['awayTeam']['goalies']]
  homeForwards = [player['playerId'] for player in boxscores[i]['boxscore']['playerByGameStats']['homeTeam']['forwards']]
  homeDefense = [player['playerId'] for player in boxscores[i]['boxscore']['playerByGameStats']['homeTeam']['defense']]
  homeGoalies = [player['playerId'] for player in boxscores[i]['boxscore']['playerByGameStats']['homeTeam']['goalies']]
  awayPlayers = [*awayForwards,*awayDefense,*awayGoalies]
  homePlayers = [*homeForwards,*homeDefense,*homeGoalies]
  gamePlayers = [*awayPlayers,*homePlayers]
  allAwayForwards.append(awayForwards)
  allAwayDefense.append(awayDefense)
  allAwayGoalies.append(awayGoalies)
  allHomeForwards.append(homeForwards)
  allHomeDefense.append(homeDefense)
  allHomeGoalies.append(homeGoalies)
  allAwayPlayers.append(awayPlayers)
  allHomePlayers.append(homePlayers)
  allPlayers.append(gamePlayers)
  # boxscores[i]['boxscore']['playerByGameStats']['awayTeam']['forwards'] = awayForwards
  # boxscores[i]['boxscore']['playerByGameStats']['awayTeam']['defense'] = awayDefense
  # boxscores[i]['boxscore']['playerByGameStats']['awayTeam']['goalies'] = awayGoalies
  # boxscores[i]['boxscore']['playerByGameStats']['homeTeam']['forwards'] = homeForwards
  # boxscores[i]['boxscore']['playerByGameStats']['homeTeam']['defense'] = homeDefense
  # boxscores[i]['boxscore']['playerByGameStats']['homeTeam']['goalies'] = homeGoalies
  # boxscores[i]['boxscore']['playerByGameStats']['awayTeam']['players'] = awayPlayers
  # boxscores[i]['boxscore']['playerByGameStats']['homeTeam']['players'] = homePlayers
  # boxscores[i]['boxscore']['playerByGameStats']['allPlayers'] = gamePlayers
# print(boxscores[0])
  
# flatboxscores = [dict(flatdict.FlatDict(game, delimiter='.')) for game in boxscores]
# print(boxscores[0])
boxscores = [flatten_dict(game) for game in boxscores]
for i in range(0,len(boxscores)):
  boxscores[i]['awayForwards'] = tuple(allAwayForwards[i])
  boxscores[i]['awayDefense'] = tuple(allAwayDefense[i])
  boxscores[i]['awayGoalies'] = tuple(allAwayGoalies[i])
  boxscores[i]['homeForwards'] = tuple(allHomeForwards[i])
  boxscores[i]['homeDefense'] = tuple(allHomeDefense[i])
  boxscores[i]['homeGoalies'] = tuple(allHomeGoalies[i])
  boxscores[i]['awayPlayers'] = tuple(allAwayPlayers[i])
  boxscores[i]['homePlayers'] = tuple(allHomePlayers[i])
  boxscores[i]['allPlayers'] = tuple(allPlayers[i])
# print(boxscores[0])
# print(players[0])

boxscores = pd.DataFrame(boxscores)
boxscores['gameDate'] = pd.to_datetime(boxscores['gameDate'], format='%Y-%m-%d')
boxscores['startTimeUTC'] = pd.to_datetime(boxscores['startTimeUTC'], utc=True)
# boxscores = boxscores.drop("winnerB", axis='columns')
boxscores = boxscores.drop("winner", axis='columns')
players = pd.DataFrame(players)
players.columns = players.columns.map(str)
players['playerId_fk'] = players.index

es = ft.EntitySet(id="customer_data")

# Adding the customer dataframe as an entity
es = es.add_dataframe(dataframe_name="boxscores", dataframe=boxscores, index="allPlayers")

# Adding the sessions dataframe as an entity
es = es.add_dataframe(dataframe_name="players", dataframe=players, index="playerId")

# Defining the relationship between customers and sessions
relationship = es.add_relationship(
  parent_dataframe_name='boxscores',
  parent_column_name='allPlayers',
  child_dataframe_name='players',
  child_column_name='playerId_fk'
)

# Adding the relationship to the entity set
# es = es.add_relationship(relationship)

# Automatically generating features
features, feature_names = ft.dfs(
  entityset=es,
  target_dataframe_name="boxscores",
  max_depth=2
)

# Viewing the generated features
print('features',features)
print('feature_names',feature_names)