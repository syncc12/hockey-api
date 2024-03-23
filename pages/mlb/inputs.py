import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

import numpy as np
import pandas as pd
from joblib import load, Parallel, delayed
import requests
from datetime import datetime
from pages.mlb.input_helpers import player_stats
from sklearn.model_selection import train_test_split
from constants.constants import RANDOM_STATE

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
  return obj

def previousGames(game):
  awayId = game['gameData']['teams']['away']['id']
  homeId = game['gameData']['teams']['home']['id']
  season = game['gameData']['game']['season']
  gameDateTime = datetime.strptime(game['gameData']['datetime']['dateTime'], '%Y-%m-%dT%H:%M:%SZ')
  away_res = requests.get(f"https://statsapi.mlb.com/api/v1/schedule?lang=en&sportIds=1&season={season}&teamId={awayId}").json()
  home_res = requests.get(f"https://statsapi.mlb.com/api/v1/schedule?lang=en&sportIds=1&season={season}&teamId={homeId}").json()
  awayPKs = []
  homePKs = []
  for date in away_res['dates']:
    for game in date['games']:
      print({'gamePk':game['gamePk'],'dateTime':datetime.strptime(game['gameDate'], '%Y-%m-%dT%H:%M:%SZ')})
      awayPKs.append({'gamePk':game['gamePk'],'dateTime':datetime.strptime(game['gameDate'], '%Y-%m-%dT%H:%M:%SZ')})
  for date in home_res['dates']:
    for game in date['games']:
      homePKs.append({'gamePk':game['gamePk'],'dateTime':datetime.strptime(game['gameDate'], '%Y-%m-%dT%H:%M:%SZ')})
  if len(awayPKs) > 0:
    awayPk = sorted((item for item in awayPKs if item['dateTime'] < gameDateTime),
                      key=lambda x: x['dateTime'],
                      reverse=True)[0]['gamePk']
    awayData = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{awayPk}/feed/live").json()
  else:
    awayData = {}
  if len(homePKs) > 0:
    homePk = sorted((item for item in homePKs if item['dateTime'] < gameDateTime),
                    key=lambda x: x['dateTime'],
                    reverse=True)[0]['gamePk']
    homeData = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{homePk}/feed/live").json()
  else:
    homeData = {}
  return awayData, homeData



def raw_data(game):
  data = {
    'gamePk': safe_chain(game,'gamePk'),
    'gameData': safe_chain(game,'gameData'),
    'liveData': {
      'linescore': safe_chain(game,'liveData','linescore'),
      'boxscore': safe_chain(game,'liveData','boxscore'),
    },
  }
  return data

def parseHeight(height):
  cleanHeight = height.replace('"','').replace("'",'').split(' ')
  return ((int(cleanHeight[0])*12) + int(cleanHeight[1]))/12



def base_inputs(game,prediction=False):
  isProjectedLineup = False
  homeId = safe_chain(game,'gameData','teams','home','id')
  awayId = safe_chain(game,'gameData','teams','away','id')
  if prediction:
    awayScore = 0
    homeScore = 0
  else:
    awayScore = safe_chain(game,'liveData','linescore','teams','away','runs',default=0)
    homeScore = safe_chain(game,'liveData','linescore','teams','home','runs',default=0)
  awayBatters = safe_chain(game,'liveData','boxscore','teams','away','batters',default=[])
  homeBatters = safe_chain(game,'liveData','boxscore','teams','home','batters',default=[])
  awayPitchers = safe_chain(game,'liveData','boxscore','teams','away','pitchers',default=[])
  homePitchers = safe_chain(game,'liveData','boxscore','teams','home','pitchers',default=[])
  awayBench = safe_chain(game,'liveData','boxscore','teams','away','bench',default=[])
  homeBench = safe_chain(game,'liveData','boxscore','teams','home','bench',default=[])
  awayBullpen = safe_chain(game,'liveData','boxscore','teams','away','bullpen',default=[])
  homeBullpen = safe_chain(game,'liveData','boxscore','teams','home','bullpen',default=[])
  awayBattingOrder = safe_chain(game,'liveData','boxscore','teams','away','battingOrder',default=[])
  homeBattingOrder = safe_chain(game,'liveData','boxscore','teams','home','battingOrder',default=[])
  awayDefense = {}
  homeDefense = {}
  awayPlayers = awayBatters + awayPitchers + awayBench + awayBullpen
  homePlayers = homeBatters + homePitchers + homeBench + homeBullpen
  players = safe_chain(game,'gameData','boxscore','players',default=[])
  awayStartingPitcher = safe_chain(game,'gameData','probablePitchers','away','id')
  homeStartingPitcher = safe_chain(game,'gameData','probablePitchers','home','id')
  awayStartingPitcherHand = safe_chain(players,f'ID{awayStartingPitcher}','pitchHand','code')
  homeStartingPitcherHand = safe_chain(players,f'ID{homeStartingPitcher}','pitchHand','code')
  awayAges = [safe_chain(players,f'ID{i}','currentAge') for i in awayPlayers]
  homeAges = [safe_chain(players,f'ID{i}','currentAge') for i in homePlayers]
  awayBatterAges = [safe_chain(players,f'ID{i}','currentAge') for i in awayBatters]
  homeBatterAges = [safe_chain(players,f'ID{i}','currentAge') for i in homeBatters]
  awayPitcherAges = [safe_chain(players,f'ID{i}','currentAge') for i in awayPitchers]
  homePitcherAges = [safe_chain(players,f'ID{i}','currentAge') for i in homePitchers]
  awayBenchAges = [safe_chain(players,f'ID{i}','currentAge') for i in awayBench]
  homeBenchAges = [safe_chain(players,f'ID{i}','currentAge') for i in homeBench]
  awayBullpenAges = [safe_chain(players,f'ID{i}','currentAge') for i in awayBullpen]
  homeBullpenAges = [safe_chain(players,f'ID{i}','currentAge') for i in homeBullpen]
  awayBattingOrderAges = [safe_chain(players,f'ID{i}','currentAge') for i in awayBattingOrder]
  homeBattingOrderAges = [safe_chain(players,f'ID{i}','currentAge') for i in homeBattingOrder]
  awayHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in awayPlayers]
  homeHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in homePlayers]
  awayBatterHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in awayBatters]
  homeBatterHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in homeBatters]
  awayPitcherHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in awayPitchers]
  homePitcherHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in homePitchers]
  awayBenchHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in awayBench]
  homeBenchHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in homeBench]
  awayBullpenHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in awayBullpen]
  homeBullpenHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in homeBullpen]
  awayBattingOrderHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in awayBattingOrder]
  homeBattingOrderHeights = [parseHeight(safe_chain(players,f'ID{i}','height',default="-1' "+'0"')) for i in homeBattingOrder]
  awayWeights = [safe_chain(players,f'ID{i}','weight') for i in awayPlayers]
  homeWeights = [safe_chain(players,f'ID{i}','weight') for i in homePlayers]
  awayBatterWeights = [safe_chain(players,f'ID{i}','weight') for i in awayBatters]
  homeBatterWeights = [safe_chain(players,f'ID{i}','weight') for i in homeBatters]
  awayPitcherWeights = [safe_chain(players,f'ID{i}','weight') for i in awayPitchers]
  homePitcherWeights = [safe_chain(players,f'ID{i}','weight') for i in homePitchers]
  awayBenchWeights = [safe_chain(players,f'ID{i}','weight') for i in awayBench]
  homeBenchWeights = [safe_chain(players,f'ID{i}','weight') for i in homeBench]
  awayBullpenWeights = [safe_chain(players,f'ID{i}','weight') for i in awayBullpen]
  homeBullpenWeights = [safe_chain(players,f'ID{i}','weight') for i in homeBullpen]
  awayBattingOrderWeights = [safe_chain(players,f'ID{i}','weight') for i in awayBattingOrder]
  homeBattingOrderWeights = [safe_chain(players,f'ID{i}','weight') for i in homeBattingOrder]
  awayPitchHand = [safe_chain(players,f'ID{i}','pitchHand','code') for i in awayPlayers]
  homePitchHand = [safe_chain(players,f'ID{i}','pitchHand','code') for i in homePlayers]
  awayBatHand = [safe_chain(players,f'ID{i}','batSide','code') for i in awayPlayers]
  homeBatHand = [safe_chain(players,f'ID{i}','batSide','code') for i in homePlayers]
  awayPitcherHand = [safe_chain(players,f'ID{i}','pitchHand','code') for i in awayPitchers]
  homePitcherHand = [safe_chain(players,f'ID{i}','pitchHand','code') for i in homePitchers]
  awayBatterHand = [safe_chain(players,f'ID{i}','batSide','code') for i in awayBatters]
  homeBatterHand = [safe_chain(players,f'ID{i}','batSide','code') for i in homeBatters]
  awayBenchBatHand = [safe_chain(players,f'ID{i}','batSide','code') for i in awayBench]
  homeBenchBatHand = [safe_chain(players,f'ID{i}','batSide','code') for i in homeBench]
  awayBullpenBatHand = [safe_chain(players,f'ID{i}','batSide','code') for i in awayBullpen]
  homeBullpenBatHand = [safe_chain(players,f'ID{i}','batSide','code') for i in homeBullpen]
  awayBattingOrderBatHand = [safe_chain(players,f'ID{i}','batSide','code') for i in awayBattingOrder]
  homeBattingOrderBatHand = [safe_chain(players,f'ID{i}','batSide','code') for i in homeBattingOrder]
  awayBenchPitchHand = [safe_chain(players,f'ID{i}','pitchHand','code') for i in awayBench]
  homeBenchPitchHand = [safe_chain(players,f'ID{i}','pitchHand','code') for i in homeBench]
  awayBullpenPitchHand = [safe_chain(players,f'ID{i}','pitchHand','code') for i in awayBullpen]
  homeBullpenPitchHand = [safe_chain(players,f'ID{i}','pitchHand','code') for i in homeBullpen]
  awayBattingOrderPitchHand = [safe_chain(players,f'ID{i}','pitchHand','code') for i in awayBattingOrder]
  homeBattingOrderPitchHand = [safe_chain(players,f'ID{i}','pitchHand','code') for i in homeBattingOrder]
  awayPlayerStats = player_stats(awayPlayers,players,prefix='awayPlayer')
  homePlayerStats = player_stats(homePlayers,players,prefix='homePlayer')
  awayBatterStats = player_stats(awayBatters,players,prefix='awayBatter')
  homeBatterStats = player_stats(homeBatters,players,prefix='homeBatter')
  awayPitcherStats = player_stats(awayPitchers,players,prefix='awayPitcher')
  homePitcherStats = player_stats(homePitchers,players,prefix='homePitcher')
  awayBenchStats = player_stats(awayBench,players,prefix='awayBench')
  homeBenchStats = player_stats(homeBench,players,prefix='homeBench')
  awayBullpenStats = player_stats(awayBullpen,players,prefix='awayBullpen')
  homeBullpenStats = player_stats(homeBullpen,players,prefix='homeBullpen')
  awayBattingOrderStats = player_stats(awayBattingOrder,players,prefix='awayBattingOrder',average=False)
  homeBattingOrderStats = player_stats(homeBattingOrder,players,prefix='homeBattingOrder',average=False)
  awayStartingPitcherStats = player_stats([awayStartingPitcher],players,prefix='awayStartingPitcher',average=False)
  homeStartingPitcherStats = player_stats([homeStartingPitcher],players,prefix='homeStartingPitcher',average=False)
  awayRightBatters = awayBatterHand.count('R')  
  homeRightBatters = homeBatterHand.count('R')
  awayLeftBatters = awayBatterHand.count('L')
  homeLeftBatters = homeBatterHand.count('L')
  awaySwitchBatters = awayBatterHand.count('S')
  homeSwitchBatters = homeBatterHand.count('S')
  awayRightPitchers = awayPitcherHand.count('R')
  homeRightPitchers = homePitcherHand.count('R')
  awayLeftPitchers = awayPitcherHand.count('L')
  homeLeftPitchers = homePitcherHand.count('L')
  awaySwitchPitchers = awayPitcherHand.count('S')
  homeSwitchPitchers = homePitcherHand.count('S')
  awayBatRight = awayBatHand.count('R')
  homeBatRight = homeBatHand.count('R')
  awayBatLeft = awayBatHand.count('L')
  homeBatLeft = homeBatHand.count('L')
  awayBatSwitch = awayBatHand.count('S')
  homeBatSwitch = homeBatHand.count('S')
  awayPitchRight = awayPitchHand.count('R')
  homePitchRight = homePitchHand.count('R')
  awayPitchLeft = awayPitchHand.count('L')
  homePitchLeft = homePitchHand.count('L')
  awayPitchSwitch = awayPitchHand.count('S')
  homePitchSwitch = homePitchHand.count('S')
  awayBenchBatRight = awayBenchBatHand.count('R')
  homeBenchBatRight = homeBenchBatHand.count('R')
  awayBenchBatLeft = awayBenchBatHand.count('L')
  homeBenchBatLeft = homeBenchBatHand.count('L')
  awayBenchBatSwitch = awayBenchBatHand.count('S')
  homeBenchBatSwitch = homeBenchBatHand.count('S')
  awayBenchPitchRight = awayBenchPitchHand.count('R')
  homeBenchPitchRight = homeBenchPitchHand.count('R')
  awayBenchPitchLeft = awayBenchPitchHand.count('L')
  homeBenchPitchLeft = homeBenchPitchHand.count('L')
  awayBenchPitchSwitch = awayBenchPitchHand.count('S')
  homeBenchPitchSwitch = homeBenchPitchHand.count('S')
  awayBullpenBatRight = awayBullpenBatHand.count('R')
  homeBullpenBatRight = homeBullpenBatHand.count('R')
  awayBullpenBatLeft = awayBullpenBatHand.count('L')
  homeBullpenBatLeft = homeBullpenBatHand.count('L')
  awayBullpenBatSwitch = awayBullpenBatHand.count('S')
  homeBullpenBatSwitch = homeBullpenBatHand.count('S')
  awayBullpenPitchRight = awayBullpenPitchHand.count('R')
  homeBullpenPitchRight = homeBullpenPitchHand.count('R')
  awayBullpenPitchLeft = awayBullpenPitchHand.count('L')
  homeBullpenPitchLeft = homeBullpenPitchHand.count('L')
  awayBullpenPitchSwitch = awayBullpenPitchHand.count('S')
  homeBullpenPitchSwitch = homeBullpenPitchHand.count('S')
  awayBattingOrderBatRight = awayBattingOrderBatHand.count('R')
  homeBattingOrderBatRight = homeBattingOrderBatHand.count('R')
  awayBattingOrderBatLeft = awayBattingOrderBatHand.count('L')
  homeBattingOrderBatLeft = homeBattingOrderBatHand.count('L')
  awayBattingOrderBatSwitch = awayBattingOrderBatHand.count('S')
  homeBattingOrderBatSwitch = homeBattingOrderBatHand.count('S')
  awayBattingOrderPitchRight = awayBattingOrderPitchHand.count('R')
  homeBattingOrderPitchRight = homeBattingOrderPitchHand.count('R')
  awayBattingOrderPitchLeft = awayBattingOrderPitchHand.count('L')
  homeBattingOrderPitchLeft = homeBattingOrderPitchHand.count('L')
  awayBattingOrderPitchSwitch = awayBattingOrderPitchHand.count('S')
  homeBattingOrderPitchSwitch = homeBattingOrderPitchHand.count('S')

  for i in safe_chain(game,'liveData','boxscore','teams','away','players',default=[]):
    player = safe_chain(game,'liveData','boxscore','teams','away','players',i)
    position_code = safe_chain(player,'position','code')
    if type(position_code) != int:
      if position_code.isdigit():
        position_code = int(position_code)
      else:
        continue
    player_id = safe_chain(player,'person','id')
    if position_code not in awayDefense:
      awayDefense[position_code] = [player_id]
    else:
      awayDefense[position_code].append(player_id)
  for i in safe_chain(game,'liveData','boxscore','teams','home','players',default=[]):
    player = safe_chain(game,'liveData','boxscore','teams','home','players',i)
    position_code = safe_chain(player,'position','code')
    if type(position_code) != int:
      if position_code.isdigit():
        position_code = int(position_code)
      else:
        continue
    player_id = safe_chain(player,'person','id')
    if position_code not in homeDefense:
      homeDefense[position_code] = [player_id]
    else:
      homeDefense[position_code].append(player_id)
  if prediction and (len(awayBatters) == 0 or len(homeBatters) == 0 or len(awayPitchers) == 0 or len(homePitchers) == 0 or len(awayBullpen) == 0 or len(homeBullpen) == 0 or len(awayBench) == 0 or len(homeBench) == 0 or len(awayBattingOrder) == 0 or len(homeBattingOrder) == 0):
    awayData, homeData = previousGames(game)
    isProjectedLineup = True
    if len(awayBatters) == 0 and len(awayData) > 0:
      awayBatters = safe_chain(awayData,'liveData','boxscore','teams','away','batters',default=[])
    if len(homeBatters) == 0 and len(homeData) > 0:
      homeBatters = safe_chain(homeData,'liveData','boxscore','teams','home','batters',default=[])
    if len(awayPitchers) == 0 and len(awayData) > 0:
      awayPitchers = safe_chain(awayData,'liveData','boxscore','teams','away','pitchers',default=[])
    if len(homePitchers) == 0 and len(homeData) > 0:
      homePitchers = safe_chain(homeData,'liveData','boxscore','teams','home','pitchers',default=[])
    if len(awayBullpen) == 0 and len(awayData) > 0:
      awayBullpen = safe_chain(awayData,'liveData','boxscore','teams','away','bullpen',default=[])
    if len(homeBullpen) == 0 and len(homeData) > 0:
      homeBullpen = safe_chain(homeData,'liveData','boxscore','teams','home','bullpen',default=[])
    if len(awayBench) == 0 and len(awayData) > 0:
      awayBench = safe_chain(awayData,'liveData','boxscore','teams','away','bench',default=[])
    if len(homeBench) == 0 and len(homeData) > 0:
      homeBench = safe_chain(homeData,'liveData','boxscore','teams','home','bench',default=[])
    if len(awayBattingOrder) == 0 and len(awayData) > 0:
      awayBattingOrder = safe_chain(awayData,'liveData','boxscore','teams','away','battingOrder',default=[])
    if len(homeBattingOrder) == 0 and len(homeData) > 0:
      homeBattingOrder = safe_chain(homeData,'liveData','boxscore','teams','home','battingOrder',default=[])
  features = {
    'id': safe_chain(game,'gamePk'),
    'season': int(safe_chain(game,'gameData','game','season')),
    'gameType': safe_chain(game,'gameData','game','type'),
    'doubleHeader': safe_chain(game,'gameData','game','doubleHeader'),
    'venue': safe_chain(game,'gameData','venue','id'),
    'homeTeam': homeId,
    'awayTeam': awayId,
    'homeScore': homeScore,
    'awayScore': awayScore,
    'totalRuns': homeScore + awayScore,
    'spread': abs(homeScore - awayScore),
    'winner': 0 if homeScore > awayScore else 1,
    'datetime': safe_chain(game,'gameData','datetime','dateTime'),
    'awayStartingPitcher': awayStartingPitcher,
    'awayStartingPitcherHand': awayStartingPitcherHand,
    'awayGamesPlayed': safe_chain(game,'gameData','teams','away','record','gamesPlayed'),
    'awayWins': safe_chain(game,'gameData','teams','away','record','wins'),
    'awayLosses': safe_chain(game,'gameData','teams','away','record','losses'),
    'awayBatters': sum(awayBatters) / len(awayBatters) if len(awayBatters) > 0 else 0,
    'awayPitchers': sum(awayPitchers) / len(awayPitchers) if len(awayPitchers) > 0 else 0,
    'awayBench': sum(awayBench) / len(awayBench) if len(awayBench) > 0 else 0,
    'awayBullpen': sum(awayBullpen) / len(awayBullpen) if len(awayBullpen) > 0 else 0,
    'awayBatter1': awayBattingOrder[0] if len(awayBattingOrder) > 0 else 0,
    'awayBatter1Hand': awayBattingOrderBatHand[0] if len(awayBattingOrderBatHand) > 0 else 0,
    'awayBatter2': awayBattingOrder[1] if len(awayBattingOrder) > 1 else 0,
    'awayBatter2Hand': awayBattingOrderBatHand[1] if len(awayBattingOrderBatHand) > 1 else 0,
    'awayBatter3': awayBattingOrder[2] if len(awayBattingOrder) > 2 else 0,
    'awayBatter3Hand': awayBattingOrderBatHand[2] if len(awayBattingOrderBatHand) > 2 else 0,
    'awayBatter4': awayBattingOrder[3] if len(awayBattingOrder) > 3 else 0,
    'awayBatter4Hand': awayBattingOrderBatHand[3] if len(awayBattingOrderBatHand) > 3 else 0,
    'awayBatter5': awayBattingOrder[4] if len(awayBattingOrder) > 4 else 0,
    'awayBatter5Hand': awayBattingOrderBatHand[4] if len(awayBattingOrderBatHand) > 4 else 0,
    'awayBatter6': awayBattingOrder[5] if len(awayBattingOrder) > 5 else 0,
    'awayBatter6Hand': awayBattingOrderBatHand[5] if len(awayBattingOrderBatHand) > 5 else 0,
    'awayBatter7': awayBattingOrder[6] if len(awayBattingOrder) > 6 else 0,
    'awayBatter7Hand': awayBattingOrderBatHand[6] if len(awayBattingOrderBatHand) > 6 else 0,
    'awayBatter8': awayBattingOrder[7] if len(awayBattingOrder) > 7 else 0,
    'awayBatter8Hand': awayBattingOrderBatHand[7] if len(awayBattingOrderBatHand) > 7 else 0,
    'awayBatter9': awayBattingOrder[8] if len(awayBattingOrder) > 8 else 0,
    'awayBatter9Hand': awayBattingOrderBatHand[8] if len(awayBattingOrderBatHand) > 8 else 0,
    'awayPitcher': sum(awayDefense.get(1,[])) / len(awayDefense.get(1,[])) if len(awayDefense.get(1,[])) > 0 else 0,
    'awayCatcher': sum(awayDefense.get(2,[])) / len(awayDefense.get(2,[])) if len(awayDefense.get(2,[])) > 0 else 0,
    'awayFirstBase': sum(awayDefense.get(3,[])) / len(awayDefense.get(3,[])) if len(awayDefense.get(3,[])) > 0 else 0,
    'awaySecondBase': sum(awayDefense.get(4,[])) / len(awayDefense.get(4,[])) if len(awayDefense.get(4,[])) > 0 else 0,
    'awayThirdBase': sum(awayDefense.get(5,[])) / len(awayDefense.get(5,[])) if len(awayDefense.get(5,[])) > 0 else 0,
    'awayShortstop': sum(awayDefense.get(6,[])) / len(awayDefense.get(6,[])) if len(awayDefense.get(6,[])) > 0 else 0,
    'awayLeftField': sum(awayDefense.get(7,[])) / len(awayDefense.get(7,[])) if len(awayDefense.get(7,[])) > 0 else 0,
    'awayCenterField': sum(awayDefense.get(8,[])) / len(awayDefense.get(8,[])) if len(awayDefense.get(8,[])) > 0 else 0,
    'awayRightField': sum(awayDefense.get(9,[])) / len(awayDefense.get(9,[])) if len(awayDefense.get(9,[])) > 0 else 0,
    'awayDH': sum(awayDefense.get(10,[])) / len(awayDefense.get(10,[])) if len(awayDefense.get(10,[])) > 0 else 0,
    'awayAverageAges': sum(awayAges) / len(awayAges) if len(awayAges) > 0 else 0,
    'awayAverageHeights': sum(awayHeights) / len(awayHeights) if len(awayHeights) > 0 else 0,
    'awayAverageWeights': sum(awayWeights) / len(awayWeights) if len(awayWeights) > 0 else 0,
    'awayBatterAverageAges': sum(awayBatterAges) / len(awayBatterAges) if len(awayBatterAges) > 0 else 0,
    'awayBatterAverageHeights': sum(awayBatterHeights) / len(awayBatterHeights) if len(awayBatterHeights) > 0 else 0,
    'awayBatterAverageWeights': sum(awayBatterWeights) / len(awayBatterWeights) if len(awayBatterWeights) > 0 else 0,
    'awayPitcherAverageAges': sum(awayPitcherAges) / len(awayPitcherAges) if len(awayPitcherAges) > 0 else 0,
    'awayPitcherAverageHeights': sum(awayPitcherHeights) / len(awayPitcherHeights) if len(awayPitcherHeights) > 0 else 0,
    'awayPitcherAverageWeights': sum(awayPitcherWeights) / len(awayPitcherWeights) if len(awayPitcherWeights) > 0 else 0,
    'awayBenchAverageAges': sum(awayBenchAges) / len(awayBenchAges) if len(awayBenchAges) > 0 else 0,
    'awayBenchAverageHeights': sum(awayBenchHeights) / len(awayBenchHeights) if len(awayBenchHeights) > 0 else 0,
    'awayBenchAverageWeights': sum(awayBenchWeights) / len(awayBenchWeights) if len(awayBenchWeights) > 0 else 0,
    'awayBullpenAverageAges': sum(awayBullpenAges) / len(awayBullpenAges) if len(awayBullpenAges) > 0 else 0,
    'awayBullpenAverageHeights': sum(awayBullpenHeights) / len(awayBullpenHeights) if len(awayBullpenHeights) > 0 else 0,
    'awayBullpenAverageWeights': sum(awayBullpenWeights) / len(awayBullpenWeights) if len(awayBullpenWeights) > 0 else 0,
    'awayBattingOrderAverageAges': sum(awayBattingOrderAges) / len(awayBattingOrderAges) if len(awayBattingOrderAges) > 0 else 0,
    'awayBattingOrderAverageHeights': sum(awayBattingOrderHeights) / len(awayBattingOrderHeights) if len(awayBattingOrderHeights) > 0 else 0,
    'awayBattingOrderAverageWeights': sum(awayBattingOrderWeights) / len(awayBattingOrderWeights) if len(awayBattingOrderWeights) > 0 else 0,
    'awayRightBatterCount': awayRightBatters,
    'awayLeftBatterCount': awayLeftBatters,
    'awaySwitchBatterCount': awaySwitchBatters,
    'awayRightPitcherCount': awayRightPitchers,
    'awayLeftPitcherCount': awayLeftPitchers,
    'awaySwitchPitcherCount': awaySwitchPitchers,
    'awayRightBatCount': awayBatRight,
    'awayLeftBatCount': awayBatLeft,
    'awaySwitchBatCount': awayBatSwitch,
    'awayRightPitchCount': awayPitchRight,
    'awayLeftPitchCount': awayPitchLeft,
    'awaySwitchPitchCount': awayPitchSwitch,
    'awayBenchRightBatCount': awayBenchBatRight,
    'awayBenchLeftBatCount': awayBenchBatLeft,
    'awayBenchSwitchBatCount': awayBenchBatSwitch,
    'awayBenchRightPitchCount': awayBenchPitchRight,
    'awayBenchLeftPitchCount': awayBenchPitchLeft,
    'awayBenchSwitchPitchCount': awayBenchPitchSwitch,
    'awayBullpenRightBatCount': awayBullpenBatRight,
    'awayBullpenLeftBatCount': awayBullpenBatLeft,
    'awayBullpenSwitchBatCount': awayBullpenBatSwitch,
    'awayBullpenRightPitchCount': awayBullpenPitchRight,
    'awayBullpenLeftPitchCount': awayBullpenPitchLeft,
    'awayBullpenSwitchPitchCount': awayBullpenPitchSwitch,
    'awayBattingOrderRightBatCount': awayBattingOrderBatRight,
    'awayBattingOrderLeftBatCount': awayBattingOrderBatLeft,
    'awayBattingOrderSwitchBatCount': awayBattingOrderBatSwitch,
    'awayBattingOrderRightPitchCount': awayBattingOrderPitchRight,
    'awayBattingOrderLeftPitchCount': awayBattingOrderPitchLeft,
    'awayBattingOrderSwitchPitchCount': awayBattingOrderPitchSwitch,
    **awayPlayerStats,
    **awayBatterStats,
    **awayPitcherStats,
    **awayBenchStats,
    **awayBullpenStats,
    **awayBattingOrderStats,
    **awayStartingPitcherStats,

    'homeStartingPitcher': homeStartingPitcher,
    'homeStartingPitcherHand': homeStartingPitcherHand,
    'homeGamesPlayed': safe_chain(game,'gameData','teams','home','record','gamesPlayed'),
    'homeWins': safe_chain(game,'gameData','teams','home','record','wins'),
    'homeLosses': safe_chain(game,'gameData','teams','home','record','losses'),
    'homeBatters': sum(homeBatters) / len(homeBatters) if len(homeBatters) > 0 else 0,
    'homePitchers': sum(homePitchers) / len(homePitchers) if len(homePitchers) > 0 else 0,
    'homeBench': sum(homeBench) / len(homeBench) if len(homeBench) > 0 else 0,
    'homeBullpen': sum(homeBullpen) / len(homeBullpen) if len(homeBullpen) > 0 else 0,
    'homeBatter1': homeBattingOrder[0] if len(homeBattingOrder) > 0 else 0,
    'homeBatter1Hand': homeBattingOrderBatHand[0] if len(homeBattingOrderBatHand) > 0 else 0,
    'homeBatter2': homeBattingOrder[1] if len(homeBattingOrder) > 1 else 0,
    'homeBatter2Hand': homeBattingOrderBatHand[1] if len(homeBattingOrderBatHand) > 1 else 0,
    'homeBatter3': homeBattingOrder[2] if len(homeBattingOrder) > 2 else 0,
    'homeBatter3Hand': homeBattingOrderBatHand[2] if len(homeBattingOrderBatHand) > 2 else 0,
    'homeBatter4': homeBattingOrder[3] if len(homeBattingOrder) > 3 else 0,
    'homeBatter4Hand': homeBattingOrderBatHand[3] if len(homeBattingOrderBatHand) > 3 else 0,
    'homeBatter5': homeBattingOrder[4] if len(homeBattingOrder) > 4 else 0,
    'homeBatter5Hand': homeBattingOrderBatHand[4] if len(homeBattingOrderBatHand) > 4 else 0,
    'homeBatter6': homeBattingOrder[5] if len(homeBattingOrder) > 5 else 0,
    'homeBatter6Hand': homeBattingOrderBatHand[5] if len(homeBattingOrderBatHand) > 5 else 0,
    'homeBatter7': homeBattingOrder[6] if len(homeBattingOrder) > 6 else 0,
    'homeBatter7Hand': homeBattingOrderBatHand[6] if len(homeBattingOrderBatHand) > 6 else 0,
    'homeBatter8': homeBattingOrder[7] if len(homeBattingOrder) > 7 else 0,
    'homeBatter8Hand': homeBattingOrderBatHand[7] if len(homeBattingOrderBatHand) > 7 else 0,
    'homeBatter9': homeBattingOrder[8] if len(homeBattingOrder) > 8 else 0,
    'homeBatter9Hand': homeBattingOrderBatHand[8] if len(homeBattingOrderBatHand) > 8 else 0,
    'homePitcher': sum(homeDefense.get(1,[])) / len(homeDefense.get(1,[])) if len(homeDefense.get(1,[])) > 0 else 0,
    'homeCatcher': sum(homeDefense.get(2,[])) / len(homeDefense.get(2,[])) if len(homeDefense.get(2,[])) > 0 else 0,
    'homeFirstBase': sum(homeDefense.get(3,[])) / len(homeDefense.get(3,[])) if len(homeDefense.get(3,[])) > 0 else 0,
    'homeSecondBase': sum(homeDefense.get(4,[])) / len(homeDefense.get(4,[])) if len(homeDefense.get(4,[])) > 0 else 0,
    'homeThirdBase': sum(homeDefense.get(5,[])) / len(homeDefense.get(5,[])) if len(homeDefense.get(5,[])) > 0 else 0,
    'homeShortstop': sum(homeDefense.get(6,[])) / len(homeDefense.get(6,[])) if len(homeDefense.get(6,[])) > 0 else 0,
    'homeLeftField': sum(homeDefense.get(7,[])) / len(homeDefense.get(7,[])) if len(homeDefense.get(7,[])) > 0 else 0,
    'homeCenterField': sum(homeDefense.get(8,[])) / len(homeDefense.get(8,[])) if len(homeDefense.get(8,[])) > 0 else 0,
    'homeRightField': sum(homeDefense.get(9,[])) / len(homeDefense.get(9,[])) if len(homeDefense.get(9,[])) > 0 else 0,
    'homeDH': sum(homeDefense.get(10,[])) / len(homeDefense.get(10,[])) if len(homeDefense.get(10,[])) > 0 else 0,
    'homeAverageAges': sum(homeAges) / len(homeAges) if len(homeAges) > 0 else 0,
    'homeAverageHeights': sum(homeHeights) / len(homeHeights) if len(homeHeights) > 0 else 0,
    'homeAverageWeights': sum(homeWeights) / len(homeWeights) if len(homeWeights) > 0 else 0,
    'homeBatterAverageAges': sum(homeBatterAges) / len(homeBatterAges) if len(homeBatterAges) > 0 else 0,
    'homeBatterAverageHeights': sum(homeBatterHeights) / len(homeBatterHeights) if len(homeBatterHeights) > 0 else 0,
    'homeBatterAverageWeights': sum(homeBatterWeights) / len(homeBatterWeights) if len(homeBatterWeights) > 0 else 0,
    'homePitcherAverageAges': sum(homePitcherAges) / len(homePitcherAges) if len(homePitcherAges) > 0 else 0,
    'homePitcherAverageHeights': sum(homePitcherHeights) / len(homePitcherHeights) if len(homePitcherHeights) > 0 else 0,
    'homePitcherAverageWeights': sum(homePitcherWeights) / len(homePitcherWeights) if len(homePitcherWeights) > 0 else 0,
    'homeBenchAverageAges': sum(homeBenchAges) / len(homeBenchAges) if len(homeBenchAges) > 0 else 0,
    'homeBenchAverageHeights': sum(homeBenchHeights) / len(homeBenchHeights) if len(homeBenchHeights) > 0 else 0,
    'homeBenchAverageWeights': sum(homeBenchWeights) / len(homeBenchWeights) if len(homeBenchWeights) > 0 else 0,
    'homeBullpenAverageAges': sum(homeBullpenAges) / len(homeBullpenAges) if len(homeBullpenAges) > 0 else 0,
    'homeBullpenAverageHeights': sum(homeBullpenHeights) / len(homeBullpenHeights) if len(homeBullpenHeights) > 0 else 0,
    'homeBullpenAverageWeights': sum(homeBullpenWeights) / len(homeBullpenWeights) if len(homeBullpenWeights) > 0 else 0,
    'homeBattingOrderAverageAges': sum(homeBattingOrderAges) / len(homeBattingOrderAges) if len(homeBattingOrderAges) > 0 else 0,
    'homeBattingOrderAverageHeights': sum(homeBattingOrderHeights) / len(homeBattingOrderHeights) if len(homeBattingOrderHeights) > 0 else 0,
    'homeBattingOrderAverageWeights': sum(homeBattingOrderWeights) / len(homeBattingOrderWeights) if len(homeBattingOrderWeights) > 0 else 0,
    'homeRightBatterCount': homeRightBatters,
    'homeLeftBatterCount': homeLeftBatters,
    'homeSwitchBatterCount': homeSwitchBatters,
    'homeRightPitcherCount': homeRightPitchers,
    'homeLeftPitcherCount': homeLeftPitchers,
    'homeSwitchPitcherCount': homeSwitchPitchers,
    'homeRightBatCount': homeBatRight,
    'homeLeftBatCount': homeBatLeft,
    'homeSwitchBatCount': homeBatSwitch,
    'homeRightPitchCount': homePitchRight,
    'homeLeftPitchCount': homePitchLeft,
    'homeSwitchPitchCount': homePitchSwitch,
    'homeBenchRightBatCount': homeBenchBatRight,
    'homeBenchLeftBatCount': homeBenchBatLeft,
    'homeBenchSwitchBatCount': homeBenchBatSwitch,
    'homeBenchRightPitchCount': homeBenchPitchRight,
    'homeBenchLeftPitchCount': homeBenchPitchLeft,
    'homeBenchSwitchPitchCount': homeBenchPitchSwitch,
    'homeBullpenRightBatCount': homeBullpenBatRight,
    'homeBullpenLeftBatCount': homeBullpenBatLeft,
    'homeBullpenSwitchBatCount': homeBullpenBatSwitch,
    'homeBullpenRightPitchCount': homeBullpenPitchRight,
    'homeBullpenLeftPitchCount': homeBullpenPitchLeft,
    'homeBullpenSwitchPitchCount': homeBullpenPitchSwitch,
    'homeBattingOrderRightBatCount': homeBattingOrderBatRight,
    'homeBattingOrderLeftBatCount': homeBattingOrderBatLeft,
    'homeBattingOrderSwitchBatCount': homeBattingOrderBatSwitch,
    'homeBattingOrderRightPitchCount': homeBattingOrderPitchRight,
    'homeBattingOrderLeftPitchCount': homeBattingOrderPitchLeft,
    'homeBattingOrderSwitchPitchCount': homeBattingOrderPitchSwitch,
    **homePlayerStats,
    **homeBatterStats,
    **homePitcherStats,
    **homeBenchStats,
    **homeBullpenStats,
    **homeBattingOrderStats,
    **homeStartingPitcherStats,
  }
  if prediction:
    return features, isProjectedLineup
  else:
    return features


ENCODE_COLUMNS = [
  'gameType',
  'doubleHeader',
  # 'awayStartingPitcherHand',
  # 'awayBatter1Hand',
  # 'awayBatter2Hand',
  # 'awayBatter3Hand',
  # 'awayBatter4Hand',
  # 'awayBatter5Hand',
  # 'awayBatter6Hand',
  # 'awayBatter7Hand',
  # 'awayBatter8Hand',
  # 'awayBatter9Hand',
  # 'homeStartingPitcherHand',
  # 'homeBatter1Hand',
  # 'homeBatter2Hand',
  # 'homeBatter3Hand',
  # 'homeBatter4Hand',
  # 'homeBatter5Hand',
  # 'homeBatter6Hand',
  # 'homeBatter7Hand',
  # 'homeBatter8Hand',
  # 'homeBatter9Hand',
]

X_INPUTS_MLB_S = [
  'gameType',
  'homeGamesPlayed',
  'season',
  'awayPitcher',
  'awayTeam',
  'homeBatters',
  'homeFirstBase',
  'awayDH',
  'homeBatter6',
  'venue',
  'awayGamesPlayed',
  'awayBatter3',
  'homeThirdBase',
  'homeCenterField',
  'homeBullpen',
  'homePitcher',
  'awayBatter4',
  'homeBatter3',
  'homeBatter7',
  'homeRightField',
  'awayBullpen',
  'awayFirstBase',
  'awayShortstop',
  'homeBench',
  'homeBatter1',
  'homeLeftField',
  'homeDH',
  'homeTeam',
  'awayCenterField',
  'awayBatter5',
  'awayThirdBase',
  'awayBatter6',
  'awayCatcher',
  'awayBench',
  'awayRightField',
  'homeBatter9',
  'homeCatcher',
  'homeSecondBase',
  'awayStartingPitcher',
  'awayBatters',
  'awayBatter8',
  'awayLeftField',
  'awayBatter2',
  'homeBatter2',
  'awaySecondBase',
  'homeStartingPitcher',
  'homeBatter5',
  'awayBatter1',
  'homeBatter8',
  'homeBatter4',
  'homeShortstop',
  'awayBatter7',
  'awayBatter9',
  'awayPitchers',
  'homePitchers',
  'homeWins',
  'awayWins',
  'awayLosses',
  'homeLosses',
]

X_INPUTS_MLB = [
  'season',
  'gameType',
  'doubleHeader',
  'venue',
  'homeTeam',
  'awayTeam',
  # 'datetime',
  'awayStartingPitcher',
  # 'awayStartingPitcherHand',
  'awayGamesPlayed',
  'awayWins',
  'awayLosses',
  'awayBatters',
  'awayPitchers',
  'awayBench',
  'awayBullpen',
  'awayBatter1',
  # 'awayBatter1Hand',
  'awayBatter2',
  # 'awayBatter2Hand',
  'awayBatter3',
  # 'awayBatter3Hand',
  'awayBatter4',
  # 'awayBatter4Hand',
  'awayBatter5',
  # 'awayBatter5Hand',
  'awayBatter6',
  # 'awayBatter6Hand',
  'awayBatter7',
  # 'awayBatter7Hand',
  'awayBatter8',
  # 'awayBatter8Hand',
  'awayBatter9',
  # 'awayBatter9Hand',
  'awayPitcher',
  'awayCatcher',
  'awayFirstBase',
  'awaySecondBase',
  'awayThirdBase',
  'awayShortstop',
  'awayLeftField',
  'awayCenterField',
  'awayRightField',
  'awayDH',
  'awayAverageAges',
  'awayAverageHeights',
  'awayAverageWeights',
  'awayBatterAverageAges',
  'awayBatterAverageHeights',
  'awayBatterAverageWeights',
  'awayPitcherAverageAges',
  'awayPitcherAverageHeights',
  'awayPitcherAverageWeights',
  'awayBenchAverageAges',
  'awayBenchAverageHeights',
  'awayBenchAverageWeights',
  'awayBullpenAverageAges',
  'awayBullpenAverageHeights',
  'awayBullpenAverageWeights',
  'awayBattingOrderAverageAges',
  'awayBattingOrderAverageHeights',
  'awayBattingOrderAverageWeights',
  'awayRightBatterCount',
  'awayLeftBatterCount',
  'awaySwitchBatterCount',
  'awayRightPitcherCount',
  'awayLeftPitcherCount',
  'awaySwitchPitcherCount',
  'awayRightBatCount',
  'awayLeftBatCount',
  'awaySwitchBatCount',
  'awayRightPitchCount',
  'awayLeftPitchCount',
  'awaySwitchPitchCount',
  'awayBenchRightBatCount',
  'awayBenchLeftBatCount',
  'awayBenchSwitchBatCount',
  'awayBenchRightPitchCount',
  'awayBenchLeftPitchCount',
  'awayBenchSwitchPitchCount',
  'awayBullpenRightBatCount',
  'awayBullpenLeftBatCount',
  'awayBullpenSwitchBatCount',
  'awayBullpenRightPitchCount',
  'awayBullpenLeftPitchCount',
  'awayBullpenSwitchPitchCount',
  'awayBattingOrderRightBatCount',
  'awayBattingOrderLeftBatCount',
  'awayBattingOrderSwitchBatCount',
  'awayBattingOrderRightPitchCount',
  'awayBattingOrderLeftPitchCount',
  'awayBattingOrderSwitchPitchCount',
  'homeStartingPitcher',
  # 'homeStartingPitcherHand',
  'homeGamesPlayed',
  'homeWins',
  'homeLosses',
  'homeBatters',
  'homePitchers',
  'homeBench',
  'homeBullpen',
  'homeBatter1',
  # 'homeBatter1Hand',
  'homeBatter2',
  # 'homeBatter2Hand',
  'homeBatter3',
  # 'homeBatter3Hand',
  'homeBatter4',
  # 'homeBatter4Hand',
  'homeBatter5',
  # 'homeBatter5Hand',
  'homeBatter6',
  # 'homeBatter6Hand',
  'homeBatter7',
  # 'homeBatter7Hand',
  'homeBatter8',
  # 'homeBatter8Hand',
  'homeBatter9',
  # 'homeBatter9Hand',
  'homePitcher',
  'homeCatcher',
  'homeFirstBase',
  'homeSecondBase',
  'homeThirdBase',
  'homeShortstop',
  'homeLeftField',
  'homeCenterField',
  'homeRightField',
  'homeDH',
  'homeAverageAges',
  'homeAverageHeights',
  'homeAverageWeights',
  'homeBatterAverageAges',
  'homeBatterAverageHeights',
  'homeBatterAverageWeights',
  'homePitcherAverageAges',
  'homePitcherAverageHeights',
  'homePitcherAverageWeights',
  'homeBenchAverageAges',
  'homeBenchAverageHeights',
  'homeBenchAverageWeights',
  'homeBullpenAverageAges',
  'homeBullpenAverageHeights',
  'homeBullpenAverageWeights',
  'homeBattingOrderAverageAges',
  'homeBattingOrderAverageHeights',
  'homeBattingOrderAverageWeights',
  'homeRightBatterCount',
  'homeLeftBatterCount',
  'homeSwitchBatterCount',
  'homeRightPitcherCount',
  'homeLeftPitcherCount',
  'homeSwitchPitcherCount',
  'homeRightBatCount',
  'homeLeftBatCount',
  'homeSwitchBatCount',
  'homeRightPitchCount',
  'homeLeftPitchCount',
  'homeSwitchPitchCount',
  'homeBenchRightBatCount',
  'homeBenchLeftBatCount',
  'homeBenchSwitchBatCount',
  'homeBenchRightPitchCount',
  'homeBenchLeftPitchCount',
  'homeBenchSwitchPitchCount',
  'homeBullpenRightBatCount',
  'homeBullpenLeftBatCount',
  'homeBullpenSwitchBatCount',
  'homeBullpenRightPitchCount',
  'homeBullpenLeftPitchCount',
  'homeBullpenSwitchPitchCount',
  'homeBattingOrderRightBatCount',
  'homeBattingOrderLeftBatCount',
  'homeBattingOrderSwitchBatCount',
  'homeBattingOrderRightPitchCount',
  'homeBattingOrderLeftPitchCount',
  'homeBattingOrderSwitchPitchCount',
]

PREFIXES = [
  'awayPlayer',
  'homePlayer',
  'awayBatter',
  'homeBatter',
  'awayPitcher',
  'homePitcher',
  'awayBench',
  'homeBench',
  'awayBullpen',
  'homeBullpen',
  'awayBattingOrder',
  'homeBattingOrder',
  'awayStartingPitcher',
  'homeStartingPitcher',
]

for prefix in PREFIXES:
  X_INPUTS_MLB.append(f'{prefix}BattingGamesPlayed')
  X_INPUTS_MLB.append(f'{prefix}BattingFlyOuts')
  X_INPUTS_MLB.append(f'{prefix}BattingGroundOuts')
  X_INPUTS_MLB.append(f'{prefix}BattingRuns')
  X_INPUTS_MLB.append(f'{prefix}BattingDoubles')
  X_INPUTS_MLB.append(f'{prefix}BattingTriples')
  X_INPUTS_MLB.append(f'{prefix}BattingHomeRuns')
  X_INPUTS_MLB.append(f'{prefix}BattingStrikeOuts')
  X_INPUTS_MLB.append(f'{prefix}BattingBaseOnBalls')
  X_INPUTS_MLB.append(f'{prefix}BattingIntentionalWalks')
  X_INPUTS_MLB.append(f'{prefix}BattingHits')
  X_INPUTS_MLB.append(f'{prefix}BattingHitByPitch')
  X_INPUTS_MLB.append(f'{prefix}BattingAtBats')
  X_INPUTS_MLB.append(f'{prefix}BattingCaughtStealing')
  X_INPUTS_MLB.append(f'{prefix}BattingStolenBases')
  X_INPUTS_MLB.append(f'{prefix}BattingGroundIntoDoublePlay')
  X_INPUTS_MLB.append(f'{prefix}BattingGroundIntoTriplePlay')
  X_INPUTS_MLB.append(f'{prefix}BattingPlateAppearances')
  X_INPUTS_MLB.append(f'{prefix}BattingTotalBases')
  X_INPUTS_MLB.append(f'{prefix}BattingRbi')
  X_INPUTS_MLB.append(f'{prefix}BattingLeftOnBase')
  X_INPUTS_MLB.append(f'{prefix}BattingSacrificeBunts')
  X_INPUTS_MLB.append(f'{prefix}BattingSacrificeFlies')
  X_INPUTS_MLB.append(f'{prefix}BattingCatchersInterference')
  X_INPUTS_MLB.append(f'{prefix}BattingPickoffs')
  X_INPUTS_MLB.append(f'{prefix}PitchingGamesPlayed')
  X_INPUTS_MLB.append(f'{prefix}PitchingGamesStarted')
  X_INPUTS_MLB.append(f'{prefix}PitchingGroundOuts')
  X_INPUTS_MLB.append(f'{prefix}PitchingAirOuts')
  X_INPUTS_MLB.append(f'{prefix}PitchingRuns')
  X_INPUTS_MLB.append(f'{prefix}PitchingDoubles')
  X_INPUTS_MLB.append(f'{prefix}PitchingTriples')
  X_INPUTS_MLB.append(f'{prefix}PitchingHomeRuns')
  X_INPUTS_MLB.append(f'{prefix}PitchingStrikeOuts')
  X_INPUTS_MLB.append(f'{prefix}PitchingBaseOnBalls')
  X_INPUTS_MLB.append(f'{prefix}PitchingIntentionalWalks')
  X_INPUTS_MLB.append(f'{prefix}PitchingHits')
  X_INPUTS_MLB.append(f'{prefix}PitchingHitByPitch')
  X_INPUTS_MLB.append(f'{prefix}PitchingAtBats')
  X_INPUTS_MLB.append(f'{prefix}PitchingCaughtStealing')
  X_INPUTS_MLB.append(f'{prefix}PitchingStolenBases')
  X_INPUTS_MLB.append(f'{prefix}PitchingNumberOfPitches')
  X_INPUTS_MLB.append(f'{prefix}PitchingInningsPitched')
  X_INPUTS_MLB.append(f'{prefix}PitchingWins')
  X_INPUTS_MLB.append(f'{prefix}PitchingLosses')
  X_INPUTS_MLB.append(f'{prefix}PitchingSaves')
  X_INPUTS_MLB.append(f'{prefix}PitchingSaveOpportunities')
  X_INPUTS_MLB.append(f'{prefix}PitchingHolds')
  X_INPUTS_MLB.append(f'{prefix}PitchingBlownSaves')
  X_INPUTS_MLB.append(f'{prefix}PitchingEarnedRuns')
  X_INPUTS_MLB.append(f'{prefix}PitchingBattersFaced')
  X_INPUTS_MLB.append(f'{prefix}PitchingOuts')
  X_INPUTS_MLB.append(f'{prefix}PitchingGamesPitched')
  X_INPUTS_MLB.append(f'{prefix}PitchingCompleteGames')
  X_INPUTS_MLB.append(f'{prefix}PitchingShutouts')
  X_INPUTS_MLB.append(f'{prefix}PitchingPitchesThrown')
  X_INPUTS_MLB.append(f'{prefix}PitchingBalls')
  X_INPUTS_MLB.append(f'{prefix}PitchingStrikes')
  X_INPUTS_MLB.append(f'{prefix}PitchingHitBatsmen')
  X_INPUTS_MLB.append(f'{prefix}PitchingBalks')
  X_INPUTS_MLB.append(f'{prefix}PitchingWildPitches')
  X_INPUTS_MLB.append(f'{prefix}PitchingPickoffs')
  X_INPUTS_MLB.append(f'{prefix}PitchingRBI')
  X_INPUTS_MLB.append(f'{prefix}PitchingGamesFinished')
  X_INPUTS_MLB.append(f'{prefix}PitchingInheritedRunners')
  X_INPUTS_MLB.append(f'{prefix}PitchingInheritedRunnersScored')
  X_INPUTS_MLB.append(f'{prefix}PitchingCathersInterference')
  X_INPUTS_MLB.append(f'{prefix}PitchingSacrificeBunts')
  X_INPUTS_MLB.append(f'{prefix}PitchingSacrificeFlies')
  X_INPUTS_MLB.append(f'{prefix}PitchingPassedBall')
  X_INPUTS_MLB.append(f'{prefix}FieldingCaughtStealing')
  X_INPUTS_MLB.append(f'{prefix}FieldingStolenBases')
  X_INPUTS_MLB.append(f'{prefix}FieldingAssists')
  X_INPUTS_MLB.append(f'{prefix}FieldingPutOuts')
  X_INPUTS_MLB.append(f'{prefix}FieldingErrors')
  X_INPUTS_MLB.append(f'{prefix}FieldingChances')
  X_INPUTS_MLB.append(f'{prefix}FieldingPassedBall')
  X_INPUTS_MLB.append(f'{prefix}FieldingPickoffs')

X_INPUTS_MLB_T = [
  'season',
  'gameType',
  'doubleHeader',
  'venue',
  'team',
  'opponent',
  # 'datetime',
  'StartingPitcher',
  # 'StartingPitcherHand',
  'GamesPlayed',
  'Wins',
  'Losses',
  'Batters',
  'Pitchers',
  'Bench',
  'Bullpen',
  'Batter1',
  # 'Batter1Hand',
  'Batter2',
  # 'Batter2Hand',
  'Batter3',
  # 'Batter3Hand',
  'Batter4',
  # 'Batter4Hand',
  'Batter5',
  # 'Batter5Hand',
  'Batter6',
  # 'Batter6Hand',
  'Batter7',
  # 'Batter7Hand',
  'Batter8',
  # 'Batter8Hand',
  'Batter9',
  # 'Batter9Hand',
  'Pitcher',
  'Catcher',
  'FirstBase',
  'SecondBase',
  'ThirdBase',
  'Shortstop',
  'LeftField',
  'CenterField',
  'RightField',
  'DH',
  'AverageAges',
  'AverageHeights',
  'AverageWeights',
  'BatterAverageAges',
  'BatterAverageHeights',
  'BatterAverageWeights',
  'PitcherAverageAges',
  'PitcherAverageHeights',
  'PitcherAverageWeights',
  'BenchAverageAges',
  'BenchAverageHeights',
  'BenchAverageWeights',
  'BullpenAverageAges',
  'BullpenAverageHeights',
  'BullpenAverageWeights',
  'BattingOrderAverageAges',
  'BattingOrderAverageHeights',
  'BattingOrderAverageWeights',
  'RightBatterCount',
  'LeftBatterCount',
  'SwitchBatterCount',
  'RightPitcherCount',
  'LeftPitcherCount',
  'SwitchPitcherCount',
  'RightBatCount',
  'LeftBatCount',
  'SwitchBatCount',
  'RightPitchCount',
  'LeftPitchCount',
  'SwitchPitchCount',
  'BenchRightBatCount',
  'BenchLeftBatCount',
  'BenchSwitchBatCount',
  'BenchRightPitchCount',
  'BenchLeftPitchCount',
  'BenchSwitchPitchCount',
  'BullpenRightBatCount',
  'BullpenLeftBatCount',
  'BullpenSwitchBatCount',
  'BullpenRightPitchCount',
  'BullpenLeftPitchCount',
  'BullpenSwitchPitchCount',
  'BattingOrderRightBatCount',
  'BattingOrderLeftBatCount',
  'BattingOrderSwitchBatCount',
  'BattingOrderRightPitchCount',
  'BattingOrderLeftPitchCount',
  'BattingOrderSwitchPitchCount',
  'opponentStartingPitcher',
  # 'opponentStartingPitcherHand',
  'opponentGamesPlayed',
  'opponentWins',
  'opponentLosses',
  'opponentBatters',
  'opponentPitchers',
  'opponentBench',
  'opponentBullpen',
  'opponentBatter1',
  # 'opponentBatter1Hand',
  'opponentBatter2',
  # 'opponentBatter2Hand',
  'opponentBatter3',
  # 'opponentBatter3Hand',
  'opponentBatter4',
  # 'opponentBatter4Hand',
  'opponentBatter5',
  # 'opponentBatter5Hand',
  'opponentBatter6',
  # 'opponentBatter6Hand',
  'opponentBatter7',
  # 'opponentBatter7Hand',
  'opponentBatter8',
  # 'opponentBatter8Hand',
  'opponentBatter9',
  # 'opponentBatter9Hand',
  'opponentPitcher',
  'opponentCatcher',
  'opponentFirstBase',
  'opponentSecondBase',
  'opponentThirdBase',
  'opponentShortstop',
  'opponentLeftField',
  'opponentCenterField',
  'opponentRightField',
  'opponentDH',
  'opponentAverageAges',
  'opponentAverageHeights',
  'opponentAverageWeights',
  'opponentBatterAverageAges',
  'opponentBatterAverageHeights',
  'opponentBatterAverageWeights',
  'opponentPitcherAverageAges',
  'opponentPitcherAverageHeights',
  'opponentPitcherAverageWeights',
  'opponentBenchAverageAges',
  'opponentBenchAverageHeights',
  'opponentBenchAverageWeights',
  'opponentBullpenAverageAges',
  'opponentBullpenAverageHeights',
  'opponentBullpenAverageWeights',
  'opponentBattingOrderAverageAges',
  'opponentBattingOrderAverageHeights',
  'opponentBattingOrderAverageWeights',
  'opponentRightBatterCount',
  'opponentLeftBatterCount',
  'opponentSwitchBatterCount',
  'opponentRightPitcherCount',
  'opponentLeftPitcherCount',
  'opponentSwitchPitcherCount',
  'opponentRightBatCount',
  'opponentLeftBatCount',
  'opponentSwitchBatCount',
  'opponentRightPitchCount',
  'opponentLeftPitchCount',
  'opponentSwitchPitchCount',
  'opponentBenchRightBatCount',
  'opponentBenchLeftBatCount',
  'opponentBenchSwitchBatCount',
  'opponentBenchRightPitchCount',
  'opponentBenchLeftPitchCount',
  'opponentBenchSwitchPitchCount',
  'opponentBullpenRightBatCount',
  'opponentBullpenLeftBatCount',
  'opponentBullpenSwitchBatCount',
  'opponentBullpenRightPitchCount',
  'opponentBullpenLeftPitchCount',
  'opponentBullpenSwitchPitchCount',
  'opponentBattingOrderRightBatCount',
  'opponentBattingOrderLeftBatCount',
  'opponentBattingOrderSwitchBatCount',
  'opponentBattingOrderRightPitchCount',
  'opponentBattingOrderLeftPitchCount',
  'opponentBattingOrderSwitchPitchCount',
]

PREFIXES_T = [
  'Player',
  'opponentPlayer',
  'Batter',
  'opponentBatter',
  'Pitcher',
  'opponentPitcher',
  'Bench',
  'opponentBench',
  'Bullpen',
  'opponentBullpen',
  'BattingOrder',
  'opponentBattingOrder',
  'StartingPitcher',
  'opponentStartingPitcher',
]

for prefix in PREFIXES_T:
  X_INPUTS_MLB_T.append(f'{prefix}BattingGamesPlayed')
  X_INPUTS_MLB_T.append(f'{prefix}BattingFlyOuts')
  X_INPUTS_MLB_T.append(f'{prefix}BattingGroundOuts')
  X_INPUTS_MLB_T.append(f'{prefix}BattingRuns')
  X_INPUTS_MLB_T.append(f'{prefix}BattingDoubles')
  X_INPUTS_MLB_T.append(f'{prefix}BattingTriples')
  X_INPUTS_MLB_T.append(f'{prefix}BattingHomeRuns')
  X_INPUTS_MLB_T.append(f'{prefix}BattingStrikeOuts')
  X_INPUTS_MLB_T.append(f'{prefix}BattingBaseOnBalls')
  X_INPUTS_MLB_T.append(f'{prefix}BattingIntentionalWalks')
  X_INPUTS_MLB_T.append(f'{prefix}BattingHits')
  X_INPUTS_MLB_T.append(f'{prefix}BattingHitByPitch')
  X_INPUTS_MLB_T.append(f'{prefix}BattingAtBats')
  X_INPUTS_MLB_T.append(f'{prefix}BattingCaughtStealing')
  X_INPUTS_MLB_T.append(f'{prefix}BattingStolenBases')
  X_INPUTS_MLB_T.append(f'{prefix}BattingGroundIntoDoublePlay')
  X_INPUTS_MLB_T.append(f'{prefix}BattingGroundIntoTriplePlay')
  X_INPUTS_MLB_T.append(f'{prefix}BattingPlateAppearances')
  X_INPUTS_MLB_T.append(f'{prefix}BattingTotalBases')
  X_INPUTS_MLB_T.append(f'{prefix}BattingRbi')
  X_INPUTS_MLB_T.append(f'{prefix}BattingLeftOnBase')
  X_INPUTS_MLB_T.append(f'{prefix}BattingSacrificeBunts')
  X_INPUTS_MLB_T.append(f'{prefix}BattingSacrificeFlies')
  X_INPUTS_MLB_T.append(f'{prefix}BattingCatchersInterference')
  X_INPUTS_MLB_T.append(f'{prefix}BattingPickoffs')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingGamesPlayed')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingGamesStarted')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingGroundOuts')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingAirOuts')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingRuns')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingDoubles')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingTriples')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingHomeRuns')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingStrikeOuts')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingBaseOnBalls')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingIntentionalWalks')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingHits')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingHitByPitch')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingAtBats')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingCaughtStealing')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingStolenBases')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingNumberOfPitches')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingInningsPitched')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingWins')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingLosses')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingSaves')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingSaveOpportunities')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingHolds')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingBlownSaves')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingEarnedRuns')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingBattersFaced')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingOuts')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingGamesPitched')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingCompleteGames')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingShutouts')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingPitchesThrown')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingBalls')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingStrikes')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingHitBatsmen')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingBalks')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingWildPitches')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingPickoffs')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingRBI')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingGamesFinished')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingInheritedRunners')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingInheritedRunnersScored')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingCathersInterference')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingSacrificeBunts')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingSacrificeFlies')
  X_INPUTS_MLB_T.append(f'{prefix}PitchingPassedBall')
  X_INPUTS_MLB_T.append(f'{prefix}FieldingCaughtStealing')
  X_INPUTS_MLB_T.append(f'{prefix}FieldingStolenBases')
  X_INPUTS_MLB_T.append(f'{prefix}FieldingAssists')
  X_INPUTS_MLB_T.append(f'{prefix}FieldingPutOuts')
  X_INPUTS_MLB_T.append(f'{prefix}FieldingErrors')
  X_INPUTS_MLB_T.append(f'{prefix}FieldingChances')
  X_INPUTS_MLB_T.append(f'{prefix}FieldingPassedBall')
  X_INPUTS_MLB_T.append(f'{prefix}FieldingPickoffs')

def load_training_data(file_path):
  return load(file_path)

def mlb_training_input(seasons,**kwargs):
  training_data_paths = [f'pages/mlb/data/training_data_{season}.joblib' for season in seasons]
  training_data = Parallel(n_jobs=-1)(delayed(load_training_data)(file) for file in training_data_paths)
  # training_data = np.concatenate([load(f'pages/mlb/data/training_data_{season}.joblib') for season in seasons]).tolist()
  training_data = np.concatenate(training_data).tolist()
  print(f'Seasons Loaded {len(training_data)}')
  if any(key in kwargs for key in ['dataframe','df','encode','inputs','outputs']):
    training_data = pd.DataFrame(training_data)
  if kwargs.get('encode'):
    for column in ENCODE_COLUMNS:
      encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
      training_data = training_data[training_data[column] != -1]
      training_data[column] = encoder.transform(training_data[column])
  if all(key in kwargs for key in ['inputs','outputs']):
    x_train = training_data [kwargs['inputs']]
    y_train = training_data [kwargs['outputs']]
    if any(key in kwargs for key in ['test_split','split','train_test_split']):
      x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
    #   if any(key in kwargs for key in ['to_numpy','torch']):
    #     x_test = x_test [kwargs['inputs']].to_numpy()
    #     y_test = y_test [kwargs['outputs']].to_numpy()
    #   if any(key in kwargs for key in ['tf','tensorflow']):
    #     x_test = x_test [kwargs['inputs']]
    #     y_test = y_test [kwargs['outputs']].to_numpy()
      return x_train, y_train, x_test, y_test
    # if any(key in kwargs for key in ['to_numpy','torch']):
    #   x_train = x_train [kwargs['inputs']].to_numpy()
    #   y_train = y_train [kwargs['outputs']].to_numpy()
    # if any(key in kwargs for key in ['tf','tensorflow']):
    #   x_train = x_train [kwargs['inputs']]
    #   y_train = y_train [kwargs['outputs']].to_numpy()
    return x_train, y_train
  else:
    return training_data

def mlb_test_input(seasons=False):
  if seasons:
    if isinstance(seasons, list):
      test_data = np.concatenate([load(f'pages/mlb/data/training_data_{season}.joblib') for season in seasons]).tolist()
    else:
      test_data = load(f'pages/mlb/data/training_data_{seasons}.joblib')
  else:
    test_data = load(f'pages/mlb/data/training_data_2023.joblib')
  return test_data