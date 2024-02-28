import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
import pandas as pd
import numpy as np
import time
from datetime import datetime
from util.helpers import parse_utc_offset, parse_start_time

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

def filter_lookup(inLookup, startingGameId, teamId, include_starting_game=False):
  if include_starting_game:
    return [game for game in inLookup[teamId] if game['id'] <= startingGameId]
  else:
    return [game for game in inLookup[teamId] if game['id'] < startingGameId]
  
def filter_lookup_matchup(inLookup, homeId, awayId, startingGameId, include_starting_game=False):
  if include_starting_game:
    return [game for game in inLookup if game['id'] <= startingGameId and (game['homeTeam']['id'] == homeId and game['awayTeam']['id'] == awayId) or (game['homeTeam']['id'] == awayId and game['awayTeam']['id'] == homeId)]
  else:
    return [game for game in inLookup if game['id'] < startingGameId and (game['homeTeam']['id'] == homeId and game['awayTeam']['id'] == awayId) or (game['homeTeam']['id'] == awayId and game['awayTeam']['id'] == homeId)]

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
Boxscores = db['dev_boxscores']

def get_season_ids(seasonId):
  games = list(Boxscores.find({'season': seasonId}))
  return [game['id'] for game in games]

def get_all_season_team_games(seasonId, teamId, get_ids=False, for_lookup=False):
  games = list(Boxscores.find({'season': seasonId, '$or': [{'homeTeam.id': teamId}, {'awayTeam.id': teamId}]}))
  if for_lookup:
    # return {game['id']: game for game in games}
    return games
  if get_ids:
    return [game['id'] for game in games]
  else:
    return games

def get_season_team_games(startingGameId, seasonId, teamId, get_ids=False, include_starting_game=False, lookup=False):
  if lookup:
    games = filter_lookup(lookup, startingGameId, teamId, include_starting_game)
  elif include_starting_game:
    games = list(Boxscores.find({'id': {'$lte': startingGameId}, 'season': seasonId, '$or': [{'homeTeam.id': teamId}, {'awayTeam.id': teamId}]}))
  else:
    games = list(Boxscores.find({'id': {'$lt': startingGameId}, 'season': seasonId, '$or': [{'homeTeam.id': teamId}, {'awayTeam.id': teamId}]}))
  if get_ids:
    return  [game['id'] for game in games]
  else:
    return games

def get_season_matchups(startingGameId, seasonId, homeTeamId, awayTeamId, get_ids=False, lookup=False):
  if lookup:
    games = filter_lookup_matchup(lookup, homeTeamId, awayTeamId, startingGameId)
  else:
    games = list(Boxscores.find({'id': {'$lt': startingGameId}, 'season': seasonId, '$or': [{'homeTeam.id': homeTeamId, 'awayTeam.id': awayTeamId}, {'homeTeam.id': awayTeamId, 'awayTeam.id': homeTeamId}]}))
  if get_ids:
    return [game['id'] for game in games]
  else:
    return games
  
def get_season_ids_shell(games):
  return [game['id'] for game in games]

def get_season_team_games_shell(games, get_ids=False):
  if get_ids:
    return  [game['id'] for game in games]
  else:
    return games

def get_season_matchups_shell(games, get_ids=False):
  if get_ids:
    return [game['id'] for game in games]
  else:
    return games

def pad_list_right(input_list, desired_length=5, pad_value=-1):
  return input_list + [pad_value] * (desired_length - len(input_list))

# print(get_season_matchups(2022030326, 20222023, 54, 25, get_ids=True))

# test_list = [1,2,3,4,5,6,7,8,9]
# print(pad_list_right(test_list[:5]))

def time_between_games(homeTeamGames, awayTeamGames, games_back=None):
  homeGameDates = [datetime.strptime(game['gameDate'], '%Y-%m-%d') for game in homeTeamGames]
  awayGameDates = [datetime.strptime(game['gameDate'], '%Y-%m-%d') for game in awayTeamGames]
  homeGameDates.sort(reverse=True)
  awayGameDates.sort(reverse=True)

  if games_back:
    homeGameDates = homeGameDates[:games_back]
    awayGameDates = awayGameDates[:games_back]

  homeRest = [abs(homeGameDates[i+1] - homeGameDates[i]).days for i in range(len(homeGameDates) - 1)]
  awayRest = [abs(awayGameDates[i+1] - awayGameDates[i]).days for i in range(len(awayGameDates) - 1)]

  homeLastRest = homeRest[0] if len(homeRest) > 0 else 0
  awayLastRest = awayRest[0] if len(awayRest) > 0 else 0

  homeRest = round(sum(homeRest)/len(homeRest) if len(homeRest) > 0 else 0, 2)
  awayRest = round(sum(awayRest)/len(awayRest)  if len(awayRest) > 0 else 0, 2)

  return homeRest, awayRest, homeLastRest, awayLastRest


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

def previous_game_stats(homeTeamId, awayTeamId, home_season_team_games, away_season_team_games):
  home_stats = {
    'season': {'scores':[],'shots':[],'hits':[],'pim':[],'powerplays':[],'powerplayGoals':[],'faceoffWinPercent':[],'blocks':[],'winLoss':[],'startUTCTimes':[],'startUTCDates':[],'easternOffsets':[],'venueOffsets':[]},
    'last5': {'scores':[],'shots':[],'hits':[],'pim':[],'powerplays':[],'powerplayGoals':[],'faceoffWinPercent':[],'blocks':[],'winLoss':[],'startUTCTimes':[],'startUTCDates':[],'easternOffsets':[],'venueOffsets':[]},
    'last10': {'scores':[],'shots':[],'hits':[],'pim':[],'powerplays':[],'powerplayGoals':[],'faceoffWinPercent':[],'blocks':[],'winLoss':[],'startUTCTimes':[],'startUTCDates':[],'easternOffsets':[],'venueOffsets':[]},
  }
  away_stats = {
    'season': {'scores':[],'shots':[],'hits':[],'pim':[],'powerplays':[],'powerplayGoals':[],'faceoffWinPercent':[],'blocks':[],'winLoss':[],'startUTCTimes':[],'startUTCDates':[],'easternOffsets':[],'venueOffsets':[]},
    'last5': {'scores':[],'shots':[],'hits':[],'pim':[],'powerplays':[],'powerplayGoals':[],'faceoffWinPercent':[],'blocks':[],'winLoss':[],'startUTCTimes':[],'startUTCDates':[],'easternOffsets':[],'venueOffsets':[]},
    'last10': {'scores':[],'shots':[],'hits':[],'pim':[],'powerplays':[],'powerplayGoals':[],'faceoffWinPercent':[],'blocks':[],'winLoss':[],'startUTCTimes':[],'startUTCDates':[],'easternOffsets':[],'venueOffsets':[]},
  }
  homeLast5 = pad_list_right(home_season_team_games[:5],pad_value=0)
  homeLast10 = pad_list_right(home_season_team_games[:10],pad_value=0)
  awayLast5 = pad_list_right(away_season_team_games[:5],pad_value=0)
  awayLast10 = pad_list_right(away_season_team_games[:10],pad_value=0)
  for game in home_season_team_games:
    start_date, start_time = parse_start_time(safe_chain(game,'startTimeUTC',default=0))
    home_stats['season']['scores'].append(safe_chain(game,'homeTeam','score',default=0))
    home_stats['season']['shots'].append(safe_chain(game,'homeTeam','sog',default=0))
    home_stats['season']['hits'].append(safe_chain(game,'homeTeam','hits',default=0))
    home_stats['season']['pim'].append(safe_chain(game,'homeTeam','pim',default=0))
    home_stats['season']['powerplays'].append(int(safe_chain(game,'homeTeam','powerPlayConversion',default='0/0').split('/')[0]))
    home_stats['season']['powerplayGoals'].append(int(safe_chain(game,'homeTeam','powerPlayConversion',default='0/0').split('/')[1]))
    home_stats['season']['faceoffWinPercent'].append(safe_chain(game,'homeTeam','faceoffWinningPctg',default=0))
    home_stats['season']['blocks'].append(safe_chain(game,'homeTeam','blocks',default=0))
    home_stats['season']['startUTCDates'].append(start_date)
    home_stats['season']['startUTCTimes'].append(start_time)
    home_stats['season']['easternOffsets'].append(parse_utc_offset(safe_chain(game,'easternUTCOffset',default=0)))
    home_stats['season']['venueOffsets'].append(parse_utc_offset(safe_chain(game,'venueUTCOffset',default=0)))
    if safe_chain(game,'homeTeam','id') == homeTeamId:
      home_stats['season']['winLoss'].append(1 if safe_chain(game,'homeTeam','score',default=0) > safe_chain(game,'awayTeam','score',default=0) else 0)
    elif safe_chain(game,'homeTeam','id') == awayTeamId:
      home_stats['season']['winLoss'].append(1 if safe_chain(game,'homeTeam','score',default=0) < safe_chain(game,'awayTeam','score',default=0) else 0)

  for game in homeLast5:
    start_date, start_time = parse_start_time(safe_chain(game,'startTimeUTC',default=0))
    home_stats['last5']['scores'].append(safe_chain(game,'homeTeam','score',default=0))
    home_stats['last5']['shots'].append(safe_chain(game,'homeTeam','sog',default=0))
    home_stats['last5']['hits'].append(safe_chain(game,'homeTeam','hits',default=0))
    home_stats['last5']['pim'].append(safe_chain(game,'homeTeam','pim',default=0))
    home_stats['last5']['powerplays'].append(int(safe_chain(game,'homeTeam','powerPlayConversion',default='0/0').split('/')[0]))
    home_stats['last5']['powerplayGoals'].append(int(safe_chain(game,'homeTeam','powerPlayConversion',default='0/0').split('/')[1]))
    home_stats['last5']['faceoffWinPercent'].append(safe_chain(game,'homeTeam','faceoffWinningPctg',default=0))
    home_stats['last5']['blocks'].append(safe_chain(game,'homeTeam','blocks',default=0))
    home_stats['last5']['startUTCDates'].append(start_date)
    home_stats['last5']['startUTCTimes'].append(start_time)
    home_stats['last5']['easternOffsets'].append(parse_utc_offset(safe_chain(game,'easternUTCOffset',default=0)))
    home_stats['last5']['venueOffsets'].append(parse_utc_offset(safe_chain(game,'venueUTCOffset',default=0)))
    if safe_chain(game,'homeTeam','id') == homeTeamId:
      home_stats['last5']['winLoss'].append(1 if safe_chain(game,'homeTeam','score',default=0) > safe_chain(game,'awayTeam','score',default=0) else 0)
    elif safe_chain(game,'homeTeam','id') == awayTeamId:
      home_stats['last5']['winLoss'].append(1 if safe_chain(game,'homeTeam','score',default=0) < safe_chain(game,'awayTeam','score',default=0) else 0)

  for game in homeLast10:
    start_date, start_time = parse_start_time(safe_chain(game,'startTimeUTC',default=0))
    home_stats['last10']['scores'].append(safe_chain(game,'homeTeam','score',default=0))
    home_stats['last10']['shots'].append(safe_chain(game,'homeTeam','sog',default=0))
    home_stats['last10']['hits'].append(safe_chain(game,'homeTeam','hits',default=0))
    home_stats['last10']['pim'].append(safe_chain(game,'homeTeam','pim',default=0))
    home_stats['last10']['powerplays'].append(int(safe_chain(game,'homeTeam','powerPlayConversion',default='0/0').split('/')[0]))
    home_stats['last10']['powerplayGoals'].append(int(safe_chain(game,'homeTeam','powerPlayConversion',default='0/0').split('/')[1]))
    home_stats['last10']['faceoffWinPercent'].append(safe_chain(game,'homeTeam','faceoffWinningPctg',default=0))
    home_stats['last10']['blocks'].append(safe_chain(game,'homeTeam','blocks',default=0))
    home_stats['last10']['startUTCDates'].append(start_date)
    home_stats['last10']['startUTCTimes'].append(start_time)
    home_stats['last10']['easternOffsets'].append(parse_utc_offset(safe_chain(game,'easternUTCOffset',default=0)))
    home_stats['last10']['venueOffsets'].append(parse_utc_offset(safe_chain(game,'venueUTCOffset',default=0)))
    if safe_chain(game,'homeTeam','id') == homeTeamId:
      home_stats['last10']['winLoss'].append(1 if safe_chain(game,'homeTeam','score',default=0) > safe_chain(game,'awayTeam','score',default=0) else 0)
    elif safe_chain(game,'homeTeam','id') == awayTeamId:
      home_stats['last10']['winLoss'].append(1 if safe_chain(game,'homeTeam','score',default=0) < safe_chain(game,'awayTeam','score',default=0) else 0)

  
  for game in away_season_team_games:
    start_date, start_time = parse_start_time(safe_chain(game,'startTimeUTC',default=0))
    away_stats['season']['scores'].append(safe_chain(game,'awayTeam','score',default=0))
    away_stats['season']['shots'].append(safe_chain(game,'awayTeam','sog',default=0))
    away_stats['season']['hits'].append(safe_chain(game,'awayTeam','hits',default=0))
    away_stats['season']['pim'].append(safe_chain(game,'awayTeam','pim',default=0))
    away_stats['season']['powerplays'].append(int(safe_chain(game,'awayTeam','powerPlayConversion',default='0/0').split('/')[0]))
    away_stats['season']['powerplayGoals'].append(int(safe_chain(game,'awayTeam','powerPlayConversion',default='0/0').split('/')[1]))
    away_stats['season']['faceoffWinPercent'].append(safe_chain(game,'awayTeam','faceoffWinningPctg',default=0))
    away_stats['season']['blocks'].append(safe_chain(game,'awayTeam','blocks',default=0))
    away_stats['season']['startUTCDates'].append(start_date)
    away_stats['season']['startUTCTimes'].append(start_time)
    away_stats['season']['easternOffsets'].append(parse_utc_offset(safe_chain(game,'easternUTCOffset',default=0)))
    away_stats['season']['venueOffsets'].append(parse_utc_offset(safe_chain(game,'venueUTCOffset',default=0)))
    if safe_chain(game,'awayTeam','id') == awayTeamId:
      home_stats['season']['winLoss'].append(1 if safe_chain(game,'awayTeam','score',default=0) > safe_chain(game,'homeTeam','score',default=0) else 0)
    elif safe_chain(game,'awayTeam','id') == homeTeamId:
      home_stats['season']['winLoss'].append(1 if safe_chain(game,'awayTeam','score',default=0) < safe_chain(game,'homeTeam','score',default=0) else 0)
  
  for game in awayLast5:
    start_date, start_time = parse_start_time(safe_chain(game,'startTimeUTC',default=0))
    away_stats['last5']['scores'].append(safe_chain(game,'awayTeam','score',default=0))
    away_stats['last5']['shots'].append(safe_chain(game,'awayTeam','sog',default=0))
    away_stats['last5']['hits'].append(safe_chain(game,'awayTeam','hits',default=0))
    away_stats['last5']['pim'].append(safe_chain(game,'awayTeam','pim',default=0))
    away_stats['last5']['powerplays'].append(int(safe_chain(game,'awayTeam','powerPlayConversion',default='0/0').split('/')[0]))
    away_stats['last5']['powerplayGoals'].append(int(safe_chain(game,'awayTeam','powerPlayConversion',default='0/0').split('/')[1]))
    away_stats['last5']['faceoffWinPercent'].append(safe_chain(game,'awayTeam','faceoffWinningPctg',default=0))
    away_stats['last5']['blocks'].append(safe_chain(game,'awayTeam','blocks',default=0))
    away_stats['last5']['startUTCDates'].append(start_date)
    away_stats['last5']['startUTCTimes'].append(start_time)
    away_stats['last5']['easternOffsets'].append(parse_utc_offset(safe_chain(game,'easternUTCOffset',default=0)))
    away_stats['last5']['venueOffsets'].append(parse_utc_offset(safe_chain(game,'venueUTCOffset',default=0)))
    if safe_chain(game,'awayTeam','id') == awayTeamId:
      home_stats['last5']['winLoss'].append(1 if safe_chain(game,'awayTeam','score',default=0) > safe_chain(game,'homeTeam','score',default=0) else 0)
    elif safe_chain(game,'awayTeam','id') == homeTeamId:
      home_stats['last5']['winLoss'].append(1 if safe_chain(game,'awayTeam','score',default=0) < safe_chain(game,'homeTeam','score',default=0) else 0)
  
  for game in awayLast10:
    start_date, start_time = parse_start_time(safe_chain(game,'startTimeUTC',default=0))
    away_stats['last10']['scores'].append(safe_chain(game,'awayTeam','score',default=0))
    away_stats['last10']['shots'].append(safe_chain(game,'awayTeam','sog',default=0))
    away_stats['last10']['hits'].append(safe_chain(game,'awayTeam','hits',default=0))
    away_stats['last10']['pim'].append(safe_chain(game,'awayTeam','pim',default=0))
    away_stats['last10']['powerplays'].append(int(safe_chain(game,'awayTeam','powerPlayConversion',default='0/0').split('/')[0]))
    away_stats['last10']['powerplayGoals'].append(int(safe_chain(game,'awayTeam','powerPlayConversion',default='0/0').split('/')[1]))
    away_stats['last10']['faceoffWinPercent'].append(safe_chain(game,'awayTeam','faceoffWinningPctg',default=0))
    away_stats['last10']['blocks'].append(safe_chain(game,'awayTeam','blocks',default=0))
    away_stats['last10']['startUTCDates'].append(start_date)
    away_stats['last10']['startUTCTimes'].append(start_time)
    away_stats['last10']['easternOffsets'].append(parse_utc_offset(safe_chain(game,'easternUTCOffset',default=0)))
    away_stats['last10']['venueOffsets'].append(parse_utc_offset(safe_chain(game,'venueUTCOffset',default=0)))
    if safe_chain(game,'awayTeam','id') == awayTeamId:
      home_stats['last10']['winLoss'].append(1 if safe_chain(game,'awayTeam','score',default=0) > safe_chain(game,'homeTeam','score',default=0) else 0)
    elif safe_chain(game,'awayTeam','id') == homeTeamId:
      home_stats['last10']['winLoss'].append(1 if safe_chain(game,'awayTeam','score',default=0) < safe_chain(game,'homeTeam','score',default=0) else 0)
  
  home_stats['season'] = {key: float(sum(value)/len(value)) if len(value) > 0 else 0 for key, value in home_stats['season'].items()}
  home_stats['last5'] = {key: float(sum(value)/len(value)) if len(value) > 0 else 0 for key, value in home_stats['last5'].items()}
  home_stats['last10'] = {key: float(sum(value)/len(value)) if len(value) > 0 else 0 for key, value in home_stats['last10'].items()}
  away_stats['season'] = {key: float(sum(value)/len(value)) if len(value) > 0 else 0 for key, value in away_stats['season'].items()}
  away_stats['last5'] = {key: float(sum(value)/len(value)) if len(value) > 0 else 0 for key, value in away_stats['last5'].items()}
  away_stats['last10'] = {key: float(sum(value)/len(value)) if len(value) > 0 else 0 for key, value in away_stats['last10'].items()}

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
  team_lookups = {}
  for i, game in enumerate(games):
    homeTeamId = game['homeTeam']['id']
    awayTeamId = game['awayTeam']['id']
    gameId = game['id']
    if homeTeamId not in team_lookups:
      team_lookups[homeTeamId] = get_all_season_team_games(seasonId, homeTeamId, for_lookup=True)
    if awayTeamId not in team_lookups:
      team_lookups[awayTeamId] = get_all_season_team_games(seasonId, awayTeamId, for_lookup=True)
    homeTeamGames = get_season_team_games(gameId, seasonId, homeTeamId, include_starting_game=True, lookup=team_lookups)
    awayTeamGames = get_season_team_games(gameId, seasonId, awayTeamId, include_starting_game=True, lookup=team_lookups)
    matchups = get_season_matchups(gameId, seasonId, homeTeamId, awayTeamId)
    home_stats, away_stats = matchup_stats(matchups, homeTeamId, awayTeamId)
    homeRest, awayRest, homeLastRest, awayLastRest = time_between_games(homeTeamGames, awayTeamGames, games_back=5)
    homeTeamGames = [d for d in homeTeamGames if d.get("id") != gameId]
    previous_home_stats, previous_away_stats = previous_game_stats(homeTeamId, awayTeamId, homeTeamGames, awayTeamGames)
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
        },
        'last5': {
          f'{homeTeamId}': {
            'scores': float(previous_home_stats['last5']['scores']),
            'shots': float(previous_home_stats['last5']['shots']),
            'hits': float(previous_home_stats['last5']['hits']),
            'pim': float(previous_home_stats['last5']['pim']),
            'powerplays': float(previous_home_stats['last5']['powerplays']),
            'powerplayGoals': float(previous_home_stats['last5']['powerplayGoals']),
            'faceoffWinPercent': float(previous_home_stats['last5']['faceoffWinPercent']),
            'blocks': float(previous_home_stats['last5']['blocks']),
            'winLoss': float(previous_home_stats['last5']['winLoss']),
            'startUTCTimes': previous_home_stats['last5']['startUTCTimes'],
            'startUTCDates': previous_home_stats['last5']['startUTCDates'],
            'easternOffsets': previous_home_stats['last5']['easternOffsets'],
            'venueOffsets': previous_home_stats['last5']['venueOffsets'],
          },
          f'{awayTeamId}': {
            'scores': float(previous_away_stats['last5']['scores']),
            'shots': float(previous_away_stats['last5']['shots']),
            'hits': float(previous_away_stats['last5']['hits']),
            'pim': float(previous_away_stats['last5']['pim']),
            'powerplays': float(previous_away_stats['last5']['powerplays']),
            'powerplayGoals': float(previous_away_stats['last5']['powerplayGoals']),
            'faceoffWinPercent': float(previous_away_stats['last5']['faceoffWinPercent']),
            'blocks': float(previous_away_stats['last5']['blocks']),
            'winLoss': float(previous_away_stats['last5']['winLoss']),
            'startUTCTimes': previous_away_stats['last5']['startUTCTimes'],
            'startUTCDates': previous_away_stats['last5']['startUTCDates'],
            'easternOffsets': previous_away_stats['last5']['easternOffsets'],
            'venueOffsets': previous_away_stats['last5']['venueOffsets'],
          }
        },
        'last10': {
          f'{homeTeamId}': {
            'scores': float(previous_home_stats['last10']['scores']),
            'shots': float(previous_home_stats['last10']['shots']),
            'hits': float(previous_home_stats['last10']['hits']),
            'pim': float(previous_home_stats['last10']['pim']),
            'powerplays': float(previous_home_stats['last10']['powerplays']),
            'powerplayGoals': float(previous_home_stats['last10']['powerplayGoals']),
            'faceoffWinPercent': float(previous_home_stats['last10']['faceoffWinPercent']),
            'blocks': float(previous_home_stats['last10']['blocks']),
            'winLoss': float(previous_home_stats['last10']['winLoss']),
            'startUTCTimes': previous_home_stats['last10']['startUTCTimes'],
            'startUTCDates': previous_home_stats['last10']['startUTCDates'],
            'easternOffsets': previous_home_stats['last10']['easternOffsets'],
            'venueOffsets': previous_home_stats['last10']['venueOffsets'],
          },
          f'{awayTeamId}': {
            'scores': float(previous_away_stats['last10']['scores']),
            'shots': float(previous_away_stats['last10']['shots']),
            'hits': float(previous_away_stats['last10']['hits']),
            'pim': float(previous_away_stats['last10']['pim']),
            'powerplays': float(previous_away_stats['last10']['powerplays']),
            'powerplayGoals': float(previous_away_stats['last10']['powerplayGoals']),
            'faceoffWinPercent': float(previous_away_stats['last10']['faceoffWinPercent']),
            'blocks': float(previous_away_stats['last10']['blocks']),
            'winLoss': float(previous_away_stats['last10']['winLoss']),
            'startUTCTimes': previous_away_stats['last10']['startUTCTimes'],
            'startUTCDates': previous_away_stats['last10']['startUTCDates'],
            'easternOffsets': previous_away_stats['last10']['easternOffsets'],
            'venueOffsets': previous_away_stats['last10']['venueOffsets'],
          }
        },
        'season': {
          f'{homeTeamId}': {
            'scores': float(previous_home_stats['season']['scores']),
            'shots': float(previous_home_stats['season']['shots']),
            'hits': float(previous_home_stats['season']['hits']),
            'pim': float(previous_home_stats['season']['pim']),
            'powerplays': float(previous_home_stats['season']['powerplays']),
            'powerplayGoals': float(previous_home_stats['season']['powerplayGoals']),
            'faceoffWinPercent': float(previous_home_stats['season']['faceoffWinPercent']),
            'blocks': float(previous_home_stats['season']['blocks']),
            'winLoss': float(previous_home_stats['season']['winLoss']),
            'startUTCTimes': previous_home_stats['season']['startUTCTimes'],
            'startUTCDates': previous_home_stats['season']['startUTCDates'],
            'easternOffsets': previous_home_stats['season']['easternOffsets'],
            'venueOffsets': previous_home_stats['season']['venueOffsets'],
          },
          f'{awayTeamId}': {
            'scores': float(previous_away_stats['season']['scores']),
            'shots': float(previous_away_stats['season']['shots']),
            'hits': float(previous_away_stats['season']['hits']),
            'pim': float(previous_away_stats['season']['pim']),
            'powerplays': float(previous_away_stats['season']['powerplays']),
            'powerplayGoals': float(previous_away_stats['season']['powerplayGoals']),
            'faceoffWinPercent': float(previous_away_stats['season']['faceoffWinPercent']),
            'blocks': float(previous_away_stats['season']['blocks']),
            'winLoss': float(previous_away_stats['season']['winLoss']),
            'startUTCTimes': previous_away_stats['season']['startUTCTimes'],
            'startUTCDates': previous_away_stats['season']['startUTCDates'],
            'easternOffsets': previous_away_stats['season']['easternOffsets'],
            'venueOffsets': previous_away_stats['season']['venueOffsets'],
          }
        },
        'team': {
          f'{homeTeamId}': {
            'rest': float(homeRest),
            'lastRest': homeLastRest,
          },
          f'{awayTeamId}': {
            'rest': float(awayRest),
            'lastRest': awayLastRest,
          }
        }
      }
    }
    Boxscores.update_one({'id': game['id']}, {'$set': addition})
    print(f'{seasonId} Updated game: {game["id"]} | {i+1}/{len(games)}')
  print(f'{seasonId} DONE')


SEASONS = [
  # 20052006,
  # 20062007,
  # 20072008,
  # 20082009,
  # 20092010,
  # 20102011,
  # 20112012,
  # 20122013,
  # 20132014,
  # 20142015,
  # 20152016,
  # 20162017,
  # 20172018,
  # 20182019,
  # 20192020,
  # 20202021,
  # 20212022,
  # 20222023,
  20232024,
]
for season in SEASONS:
  print(f'Updating season: {season}')
  update_boxscores(season)

# print(time_between_games(2022030326, 20222023, 54, 25, games_back=5))
