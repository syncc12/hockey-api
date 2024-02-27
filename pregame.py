import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
import pandas as pd
import numpy as np
import time
# from util.helpers import safe_chain

def safe_chain(obj, *keys, default=-1):
  for key in keys:
    if key == default:
      return default
    else:
      if type(key) == int:
        if len(obj) > key:
          obj = obj[key]
        else:
          return default
      else:
        try:
          obj = getattr(obj, key, default) if hasattr(obj, key) else obj[key]
        except (KeyError, TypeError, AttributeError):
          return default
  if obj == None:
    return default
  return obj

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
Boxscores = db['dev_boxscores']

def get_season_ids(seasonId):
  games = list(Boxscores.find({'season': seasonId}))
  return [game['id'] for game in games]

def get_season_team_games(startingGameId, seasonId, teamId, get_ids=False):
  games = list(Boxscores.find({'id': {'$lt': startingGameId}, 'season': seasonId, '$or': [{'homeTeam.id': teamId}, {'awayTeam.id': teamId}]}))
  if get_ids:
    return  [game['id'] for game in games]
  else:
    return games

def get_season_matchups(startingGameId, seasonId, homeTeamId, awayTeamId, get_ids=False):
  games = list(Boxscores.find({'id': {'$lt': startingGameId}, 'season': seasonId, '$or': [{'homeTeam.id': homeTeamId, 'awayTeam.id': awayTeamId}, {'homeTeam.id': awayTeamId, 'awayTeam.id': homeTeamId}]}))
  if get_ids:
    return [game['id'] for game in games]
  else:
    return games

def pad_list_right(input_list, desired_length=5, pad_value=-1):
  return input_list + [pad_value] * (desired_length - len(input_list))

# print(get_season_matchups(2022030326, 20222023, 54, 25, get_ids=True))

# test_list = [1,2,3,4,5,6,7,8,9]
# print(pad_list_right(test_list[:5]))

def matchup_stats(matchups, homeId, awayId):
  home_stats = {
    'scores': [],
    'shots': [],
    'hits': [],
    'pim': [],
    'powerplays': [],
    'powerplayGoals': [],
    'faceoffWinPercent': [],
    'blocks': [],
    'winLoss': [],
  }
  away_stats = {
    'scores': [],
    'shots': [],
    'hits': [],
    'pim': [],
    'powerplays': [],
    'powerplayGoals': [],
    'faceoffWinPercent': [],
    'blocks': [],
    'winLoss': [],
  }
  for matchup in matchups:
    if safe_chain(matchup,'homeTeam','id') == homeId:
      home_stats['scores'].append(safe_chain(matchup,'homeTeam','score',default=0))
      away_stats['scores'].append(safe_chain(matchup,'awayTeam','score',default=0))
      home_stats['shots'].append(safe_chain(matchup,'homeTeam','sog',default=0))
      away_stats['shots'].append(safe_chain(matchup,'awayTeam','sog',default=0))
      home_stats['hits'].append(safe_chain(matchup,'homeTeam','hits',default=0))
      away_stats['hits'].append(safe_chain(matchup,'awayTeam','hits',default=0))
      home_stats['pim'].append(safe_chain(matchup,'homeTeam','pim',default=0))
      away_stats['pim'].append(safe_chain(matchup,'awayTeam','pim',default=0))
      home_stats['powerplays'].append(int(safe_chain(matchup,'homeTeam','powerPlayConversion',default='0/0').split('/')[0]))
      away_stats['powerplays'].append(int(safe_chain(matchup,'awayTeam','powerPlayConversion',default='0/0').split('/')[0]))
      home_stats['powerplayGoals'].append(int(safe_chain(matchup,'homeTeam','powerPlayConversion',default='0/0').split('/')[1]))
      away_stats['powerplayGoals'].append(int(safe_chain(matchup,'awayTeam','powerPlayConversion',default='0/0').split('/')[1]))
      home_stats['faceoffWinPercent'].append(safe_chain(matchup,'homeTeam','faceoffWinningPctg',default=0))
      away_stats['faceoffWinPercent'].append(safe_chain(matchup,'awayTeam','faceoffWinningPctg',default=0))
      home_stats['blocks'].append(safe_chain(matchup,'homeTeam','blocks',default=0))
      away_stats['blocks'].append(safe_chain(matchup,'awayTeam','blocks',default=0))
      home_stats['winLoss'].append(1 if safe_chain(matchup,'homeTeam','score',default=0) > safe_chain(matchup,'awayTeam','score',default=0) else 0)
      away_stats['winLoss'].append(1 if safe_chain(matchup,'homeTeam','score',default=0) < safe_chain(matchup,'awayTeam','score',default=0) else 0)
    elif safe_chain(matchup,'homeTeam','id') == awayId:
      home_stats['scores'].append(safe_chain(matchup,'awayTeam','score',default=0))
      away_stats['scores'].append(safe_chain(matchup,'homeTeam','score',default=0))
      home_stats['shots'].append(safe_chain(matchup,'awayTeam','sog',default=0))
      away_stats['shots'].append(safe_chain(matchup,'homeTeam','sog',default=0))
      home_stats['hits'].append(safe_chain(matchup,'awayTeam','hits',default=0))
      away_stats['hits'].append(safe_chain(matchup,'homeTeam','hits',default=0))
      home_stats['pim'].append(safe_chain(matchup,'awayTeam','pim',default=0))
      away_stats['pim'].append(safe_chain(matchup,'homeTeam','pim',default=0))
      home_stats['powerplays'].append(int(safe_chain(matchup,'awayTeam','powerPlayConversion',default='0/0').split('/')[0]))
      away_stats['powerplays'].append(int(safe_chain(matchup,'homeTeam','powerPlayConversion',default='0/0').split('/')[0]))
      home_stats['powerplayGoals'].append(int(safe_chain(matchup,'awayTeam','powerPlayConversion',default='0/0').split('/')[1]))
      away_stats['powerplayGoals'].append(int(safe_chain(matchup,'homeTeam','powerPlayConversion',default='0/0').split('/')[1]))
      home_stats['faceoffWinPercent'].append(safe_chain(matchup,'awayTeam','faceoffWinningPctg',default=0))
      away_stats['faceoffWinPercent'].append(safe_chain(matchup,'homeTeam','faceoffWinningPctg',default=0))
      home_stats['blocks'].append(safe_chain(matchup,'awayTeam','blocks',default=0))
      away_stats['blocks'].append(safe_chain(matchup,'homeTeam','blocks',default=0))
      home_stats['winLoss'].append(1 if safe_chain(matchup,'awayTeam','score',default=0) > safe_chain(matchup,'homeTeam','score',default=0) else 0)
      away_stats['winLoss'].append(1 if safe_chain(matchup,'awayTeam','score',default=0) < safe_chain(matchup,'homeTeam','score',default=0) else 0)
  home_stats = {key: float(sum(value)/len(value)) if len(value) > 0 else 0 for key, value in home_stats.items()}
  away_stats = {key: float(sum(value)/len(value)) if len(value) > 0 else 0 for key, value in away_stats.items()}
  return home_stats, away_stats

# matchups = get_season_matchups(2022030326, 20222023, 54, 25)
# home_stats, away_stats = matchup_stats(matchups, 54, 25)
# print(home_stats)
# print(away_stats)
# addition = {
#   'pregame': {
#     'matchup': {
#       f'{0}': home_stats,
#       f'{1}': away_stats,
#     }
#   }
# }
# print(addition)

# print(get_season_ids(20222023))

def update_boxscores(seasonId):
  games = list(Boxscores.find({'season': seasonId}))
  for i, game in enumerate(games):
    homeTeamId = game['homeTeam']['id']
    awayTeamId = game['awayTeam']['id']
    matchups = get_season_matchups(game['id'], seasonId, homeTeamId, awayTeamId)
    home_stats, away_stats = matchup_stats(matchups, homeTeamId, awayTeamId)
    addition = {
      'pregame': {
        'matchup': {
          f'{homeTeamId}': {
            'scores': float(home_stats['scores']),
            'shots': float(home_stats['shots']),
            'hits': float(home_stats['hits']),
            'pim': float(home_stats['pim']),
            'powerplays': float(home_stats['powerplays']),
            'powerplayGoals': float(home_stats['powerplayGoals']),
            'faceoffWinPercent': float(home_stats['faceoffWinPercent']),
            'blocks': float(home_stats['blocks']),
            'winLoss': float(home_stats['winLoss']),
          },
          f'{awayTeamId}': {
            'scores': float(away_stats['scores']),
            'shots': float(away_stats['shots']),
            'hits': float(away_stats['hits']),
            'pim': float(away_stats['pim']),
            'powerplays': float(away_stats['powerplays']),
            'powerplayGoals': float(away_stats['powerplayGoals']),
            'faceoffWinPercent': float(away_stats['faceoffWinPercent']),
            'blocks': float(away_stats['blocks']),
            'winLoss': float(away_stats['winLoss']),
          }
        }
      }
    }
    # print(addition)
    Boxscores.update_one({'id': game['id']}, {'$set': addition})
    print(f'{seasonId} Updated game: {game["id"]} | {i+1}/{len(games)}')
  print(f'{seasonId} DONE')


SEASONS = [
  20052006,
  20062007,
  20072008,
  20082009,
  20092010,
  20102011,
  20112012,
  20122013,
  20132014,
  20142015,
  20152016,
  20162017,
  20172018,
  20182019,
  20192020,
  20202021,
  20212022,
  # 20222023,
]
for season in SEASONS:
  print(f'Updating season: {season}')
  update_boxscores(season)