from util.helpers import safe_chain, n2n, false_chain, getPlayer, getAge, b2n, isNaN, formatDate, formatDatetime, formatTime, safe_none, collect_players
import math
import requests
from pymongo import MongoClient
from datetime import datetime
from util.query import get_last_game_player_stats, get_last_game_team_stats, last_player_game_stats, list_players_in_game
from inputs.goalies import goalie
from inputs.forwards import forwards
from inputs.defense import defense
from inputs.base import base_inputs

REPLACE_VALUE = -1

def master_inputs(db, game, isProjectedLineup=False):
  
  Players = db['dev_players']
  startTime = REPLACE_VALUE
  if 'startTimeUTC' in game and false_chain(game,'startTimeUTC'):
    startTime = formatDatetime(safe_chain(game,'startTimeUTC'))
  date = REPLACE_VALUE
  if 'gameDate' in game and game['gameDate']:
    date = formatDate(game['gameDate'])
  homeTeam = safe_chain(game,'homeTeam')
  awayTeam = safe_chain(game,'awayTeam')
  gi = safe_chain(game,'boxscore','gameInfo')
  pbgs = safe_chain(game,'boxscore','playerByGameStats')
  allPlayerIds = []
  awayForwardIds = []
  awayDefenseIds = []
  homeForwardIds = []
  homeDefenseIds = []
  if safe_chain(pbgs,'awayTeam','forwards') != REPLACE_VALUE:
    for p in pbgs['awayTeam']['forwards']:
      allPlayerIds.append({'playerId': p['playerId']})
      awayForwardIds.append(p['playerId'])
  if safe_chain(pbgs,'awayTeam','defense') != REPLACE_VALUE:
    for p in pbgs['awayTeam']['defense']:
      allPlayerIds.append({'playerId': p['playerId']})
      awayDefenseIds.append(p['playerId'])
  if safe_chain(pbgs,'awayTeam','goalies') != REPLACE_VALUE:
    for p in pbgs['awayTeam']['goalies']:
      allPlayerIds.append({'playerId': p['playerId']})
  if safe_chain(pbgs,'homeTeam','forwards') != REPLACE_VALUE:
    for p in pbgs['homeTeam']['forwards']:
      allPlayerIds.append({'playerId': p['playerId']})
      homeForwardIds.append(p['playerId'])
  if safe_chain(pbgs,'homeTeam','defense') != REPLACE_VALUE:
    for p in pbgs['homeTeam']['defense']:
      allPlayerIds.append({'playerId': p['playerId']})
      homeDefenseIds.append(p['playerId'])
  if safe_chain(pbgs,'homeTeam','goalies') != REPLACE_VALUE:
    for p in pbgs['homeTeam']['goalies']:
      allPlayerIds.append({'playerId': p['playerId']})
  if len(allPlayerIds) > 0:
    allPlayers = list(Players.find(
      {'$or': allPlayerIds},
      {'_id': 0, 'playerId': 1, 'birthDate': 1, 'shootsCatches': 1, 'weightInPounds': 1, 'heightInInches': 1}
    ))
  else:
    allPlayers = []
  
  # allPlayers = collect_players(db,allPlayerIds)

  homeStartingGoalieID = -1
  homeBackupGoalieID = -1
  awayStartingGoalieID = -1
  awayBackupGoalieID = -1

  if false_chain(pbgs,'awayTeam','goalies'):
    if len(pbgs['awayTeam']['goalies']) == 1:
      awayStartingGoalieID = pbgs['awayTeam']['goalies'][0]['playerId']
    
    if len(pbgs['awayTeam']['goalies']) > 1:
      startingTOI = safe_none(formatTime(pbgs['awayTeam']['goalies'][0]['toi']))
      startingID = pbgs['awayTeam']['goalies'][0]['playerId']
      backupID = pbgs['awayTeam']['goalies'][1]['playerId']
      for g in pbgs['awayTeam']['goalies']:
        if safe_none(formatTime(g['toi'])) > safe_none(startingTOI):
          startingTOI = safe_none(formatTime(g['toi']))
          backupID = startingID
          startingID = g['playerId']
      awayStartingGoalieID = startingID
      awayBackupGoalieID = backupID

  if false_chain(pbgs,'awayTeam','goalies'):
    if len(pbgs['homeTeam']['goalies']) == 1:
      homeStartingGoalieID = pbgs['homeTeam']['goalies'][0]['playerId']
    
    if len(pbgs['homeTeam']['goalies']) > 1:
      startingTOI = safe_none(formatTime(pbgs['homeTeam']['goalies'][0]['toi']))
      startingID =  pbgs['homeTeam']['goalies'][0]['playerId']
      backupID = pbgs['homeTeam']['goalies'][1]['playerId']
      for g in pbgs['homeTeam']['goalies']:
        if safe_none(formatTime(g['toi'])) > safe_none(startingTOI):
          startingTOI = safe_none(formatTime(g['toi']))
          backupID = startingID
          startingID = g['playerId']
      homeStartingGoalieID = startingID
      homeBackupGoalieID = backupID
  
  # lastPlayerStats = last_player_game_stats(db=db,gameId=game['id'],playerIDs=allPlayerIds)

  try:
    all_inputs = {}
    if isProjectedLineup:
      all_inputs = {
        f'{awayStartingGoalieID}|{homeStartingGoalieID}': {
          **base_inputs(db,awayTeam,homeTeam,game,gi,startTime,date),
          **forwards(db,awayForwardIds,allPlayers,game,isAway=True),
          **defense(db,awayDefenseIds,allPlayers,game,isAway=True),
          **goalie(db,awayStartingGoalieID,allPlayers,game,isStarting=True,isAway=True),
          **goalie(db,awayBackupGoalieID,allPlayers,game,isStarting=False,isAway=True),
          **forwards(db,homeForwardIds,allPlayers,game,isAway=False),
          **defense(db,homeDefenseIds,allPlayers,game,isAway=False),
          **goalie(db,homeStartingGoalieID,allPlayers,game,isStarting=True,isAway=False),
          **goalie(db,homeBackupGoalieID,allPlayers,game,isStarting=False,isAway=False),
        },
        f'{awayStartingGoalieID}|{homeBackupGoalieID}': {
          **base_inputs(db,awayTeam,homeTeam,game,gi,startTime,date),
          **forwards(db,awayForwardIds,allPlayers,game,isAway=True),
          **defense(db,awayDefenseIds,allPlayers,game,isAway=True),
          **goalie(db,awayStartingGoalieID,allPlayers,game,isStarting=True,isAway=True),
          **goalie(db,awayBackupGoalieID,allPlayers,game,isStarting=False,isAway=True),
          **forwards(db,homeForwardIds,allPlayers,game,isAway=False),
          **defense(db,homeDefenseIds,allPlayers,game,isAway=False),
          **goalie(db,homeBackupGoalieID,allPlayers,game,isStarting=True,isAway=False),
          **goalie(db,homeStartingGoalieID,allPlayers,game,isStarting=False,isAway=False),
        },
        f'{awayBackupGoalieID}|{homeStartingGoalieID}': {
          **base_inputs(db,awayTeam,homeTeam,game,gi,startTime,date),
          **forwards(db,awayForwardIds,allPlayers,game,isAway=True),
          **defense(db,awayDefenseIds,allPlayers,game,isAway=True),
          **goalie(db,awayBackupGoalieID,allPlayers,game,isStarting=True,isAway=True),
          **goalie(db,awayStartingGoalieID,allPlayers,game,isStarting=False,isAway=True),
          **forwards(db,homeForwardIds,allPlayers,game,isAway=False),
          **defense(db,homeDefenseIds,allPlayers,game,isAway=False),
          **goalie(db,homeStartingGoalieID,allPlayers,game,isStarting=True,isAway=False),
          **goalie(db,homeBackupGoalieID,allPlayers,game,isStarting=False,isAway=False),
        },
        f'{awayBackupGoalieID}|{homeBackupGoalieID}': {
          **base_inputs(db,awayTeam,homeTeam,game,gi,startTime,date),
          **forwards(db,awayForwardIds,allPlayers,game,isAway=True),
          **defense(db,awayDefenseIds,allPlayers,game,isAway=True),
          **goalie(db,awayBackupGoalieID,allPlayers,game,isStarting=True,isAway=True),
          **goalie(db,awayStartingGoalieID,allPlayers,game,isStarting=False,isAway=True),
          **forwards(db,homeForwardIds,allPlayers,game,isAway=False),
          **defense(db,homeDefenseIds,allPlayers,game,isAway=False),
          **goalie(db,homeBackupGoalieID,allPlayers,game,isStarting=True,isAway=False),
          **goalie(db,homeStartingGoalieID,allPlayers,game,isStarting=False,isAway=False),
        },
      }
    else:
      all_inputs = {
        **base_inputs(db,awayTeam,homeTeam,game,gi,startTime,date),
        **forwards(db,awayForwardIds,allPlayers,game,isAway=True),
        **defense(db,awayDefenseIds,allPlayers,game,isAway=True),
        **goalie(db,awayStartingGoalieID,allPlayers,game,isStarting=True,isAway=True),
        **goalie(db,awayBackupGoalieID,allPlayers,game,isStarting=False,isAway=True),
        **forwards(db,homeForwardIds,allPlayers,game,isAway=False),
        **defense(db,homeDefenseIds,allPlayers,game,isAway=False),
        **goalie(db,homeStartingGoalieID,allPlayers,game,isStarting=True,isAway=False),
        **goalie(db,homeBackupGoalieID,allPlayers,game,isStarting=False,isAway=False),
      }

    return {
      'data': all_inputs,
      'options': {
        'projectedLineup': isProjectedLineup,
      },
    }
    # return all_inputs
  except Exception as error:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('ERROR','INPUTS')
    print('id', safe_chain(game,'id'))
    print(safe_chain(game,'homeTeam'))
    print(safe_chain(game,'awayTeam'))
    print('error',error)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')