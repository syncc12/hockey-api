import numpy as np
from joblib import load

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

def base_inputs(game):
  homeId = safe_chain(game,'gameData','teams','home','id')
  awayId = safe_chain(game,'gameData','teams','away','id')
  homeScore = safe_chain(game,'liveData','linescore','teams','home','runs',default=0)
  awayScore = safe_chain(game,'liveData','linescore','teams','home','runs',default=0)
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
  return {
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
    'awayBatters': sum(awayBatters) / len(awayBatters) if len(awayBatters) > 0 else 0,
    'awayPitchers': sum(awayPitchers) / len(awayPitchers) if len(awayPitchers) > 0 else 0,
    'awayBench': sum(awayBench) / len(awayBench) if len(awayBench) > 0 else 0,
    'awayBullpen': sum(awayBullpen) / len(awayBullpen) if len(awayBullpen) > 0 else 0,
    'awayBatter1': awayBattingOrder[0] if len(awayBattingOrder) > 0 else 0,
    'awayBatter2': awayBattingOrder[1] if len(awayBattingOrder) > 1 else 0,
    'awayBatter3': awayBattingOrder[2] if len(awayBattingOrder) > 2 else 0,
    'awayBatter4': awayBattingOrder[3] if len(awayBattingOrder) > 3 else 0,
    'awayBatter5': awayBattingOrder[4] if len(awayBattingOrder) > 4 else 0,
    'awayBatter6': awayBattingOrder[5] if len(awayBattingOrder) > 5 else 0,
    'awayBatter7': awayBattingOrder[6] if len(awayBattingOrder) > 6 else 0,
    'awayBatter8': awayBattingOrder[7] if len(awayBattingOrder) > 7 else 0,
    'awayBatter9': awayBattingOrder[8] if len(awayBattingOrder) > 8 else 0,
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
    'homeBatters': sum(homeBatters) / len(homeBatters) if len(homeBatters) > 0 else 0,
    'homePitchers': sum(homePitchers) / len(homePitchers) if len(homePitchers) > 0 else 0,
    'homeBench': sum(homeBench) / len(homeBench) if len(homeBench) > 0 else 0,
    'homeBullpen': sum(homeBullpen) / len(homeBullpen) if len(homeBullpen) > 0 else 0,
    'homeBatter1': homeBattingOrder[0] if len(homeBattingOrder) > 0 else 0,
    'homeBatter2': homeBattingOrder[1] if len(homeBattingOrder) > 1 else 0,
    'homeBatter3': homeBattingOrder[2] if len(homeBattingOrder) > 2 else 0,
    'homeBatter4': homeBattingOrder[3] if len(homeBattingOrder) > 3 else 0,
    'homeBatter5': homeBattingOrder[4] if len(homeBattingOrder) > 4 else 0,
    'homeBatter6': homeBattingOrder[5] if len(homeBattingOrder) > 5 else 0,
    'homeBatter7': homeBattingOrder[6] if len(homeBattingOrder) > 6 else 0,
    'homeBatter8': homeBattingOrder[7] if len(homeBattingOrder) > 7 else 0,
    'homeBatter9': homeBattingOrder[8] if len(homeBattingOrder) > 8 else 0,
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
  }


ENCODE_COLUMNS = [
  'gameType',
  'doubleHeader',
]

X_INPUTS_MLB = [
  'season',
  'gameType',
  'doubleHeader',
  'venue',
  'homeTeam',
  'awayTeam',
  # 'datetime',
  'awayBatters',
  'awayPitchers',
  'awayBench',
  'awayBullpen',
  'awayBatter1',
  'awayBatter2',
  'awayBatter3',
  'awayBatter4',
  'awayBatter5',
  'awayBatter6',
  'awayBatter7',
  'awayBatter8',
  'awayBatter9',
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
  'homeBatters',
  'homePitchers',
  'homeBench',
  'homeBullpen',
  'homeBatter1',
  'homeBatter2',
  'homeBatter3',
  'homeBatter4',
  'homeBatter5',
  'homeBatter6',
  'homeBatter7',
  'homeBatter8',
  'homeBatter9',
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
]

def mlb_training_input(seasons):
  training_data = np.concatenate([load(f'pages/mlb/data/training_data_{season}.joblib') for season in seasons]).tolist()
  print('Seasons Loaded')
  return training_data