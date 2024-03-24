from util.helpers import safe_chain, n2n, false_chain, getPlayer, getAge, b2n, isNaN, formatDate, formatDatetime, formatTime, safe_none, collect_players, projected_roster
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

def master_inputs(db, boxscore, isProjectedRoster=False, isProjectedLineup=False, training=False):
  Players = db['dev_players']
  startTime = REPLACE_VALUE
  if 'startTimeUTC' in boxscore and false_chain(boxscore,'startTimeUTC'):
    startTime = formatDatetime(safe_chain(boxscore,'startTimeUTC'))
  date = REPLACE_VALUE
  if 'gameDate' in boxscore and boxscore['gameDate']:
    date = formatDate(boxscore['gameDate'])
  homeTeam = safe_chain(boxscore,'homeTeam')
  awayTeam = safe_chain(boxscore,'awayTeam')
  gi = safe_chain(boxscore,'summary','gameInfo')
  pbgs = safe_chain(boxscore,'playerByGameStats')
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
      if isProjectedRoster:
        awayStartingGoalieID = safe_chain(pbgs,'awayTeam','goalies',0,'playerId')
      else:
        awayStartingGoalieID = pbgs['awayTeam']['goalies'][0]['playerId']
    
    if len(pbgs['awayTeam']['goalies']) > 1:
      if isProjectedRoster:
        awayStartingGoalieID = safe_chain(pbgs,'awayTeam','goalies',0,'playerId')
        awayBackupGoalieID = safe_chain(pbgs,'awayTeam','goalies',1,'playerId')
      else:
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

  if false_chain(pbgs,'homeTeam','goalies'):
    if len(pbgs['homeTeam']['goalies']) == 1:
      if isProjectedRoster:
        homeStartingGoalieID = safe_chain(pbgs,'homeTeam','goalies',0,'playerId')
      else:
        homeStartingGoalieID = pbgs['homeTeam']['goalies'][0]['playerId']
    
    if len(pbgs['homeTeam']['goalies']) > 1:
      if isProjectedRoster:
        homeStartingGoalieID = safe_chain(pbgs,'homeTeam','goalies',0,'playerId')
        homeBackupGoalieID = safe_chain(pbgs,'homeTeam','goalies',1,'playerId')
      else:
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

  all_inputs = {}
  if isProjectedLineup:
    if training:
      all_inputs = [
        {
          **base_inputs(db,awayTeam,homeTeam,boxscore,gi,startTime,date),
          **forwards(db,awayForwardIds,allPlayers,boxscore,isAway=True),
          **defense(db,awayDefenseIds,allPlayers,boxscore,isAway=True),
          **goalie(db,awayStartingGoalieID,allPlayers,boxscore,isStarting=True,isAway=True),
          **goalie(db,awayBackupGoalieID,allPlayers,boxscore,isStarting=False,isAway=True),
          **forwards(db,homeForwardIds,allPlayers,boxscore,isAway=False),
          **defense(db,homeDefenseIds,allPlayers,boxscore,isAway=False),
          **goalie(db,homeStartingGoalieID,allPlayers,boxscore,isStarting=True,isAway=False),
          **goalie(db,homeBackupGoalieID,allPlayers,boxscore,isStarting=False,isAway=False),
        },{
          **base_inputs(db,awayTeam,homeTeam,boxscore,gi,startTime,date),
          **forwards(db,awayForwardIds,allPlayers,boxscore,isAway=True),
          **defense(db,awayDefenseIds,allPlayers,boxscore,isAway=True),
          **goalie(db,awayStartingGoalieID,allPlayers,boxscore,isStarting=True,isAway=True),
          **goalie(db,awayBackupGoalieID,allPlayers,boxscore,isStarting=False,isAway=True),
          **forwards(db,homeForwardIds,allPlayers,boxscore,isAway=False),
          **defense(db,homeDefenseIds,allPlayers,boxscore,isAway=False),
          **goalie(db,homeBackupGoalieID,allPlayers,boxscore,isStarting=True,isAway=False),
          **goalie(db,homeStartingGoalieID,allPlayers,boxscore,isStarting=False,isAway=False),
        },{
          **base_inputs(db,awayTeam,homeTeam,boxscore,gi,startTime,date),
          **forwards(db,awayForwardIds,allPlayers,boxscore,isAway=True),
          **defense(db,awayDefenseIds,allPlayers,boxscore,isAway=True),
          **goalie(db,awayBackupGoalieID,allPlayers,boxscore,isStarting=True,isAway=True),
          **goalie(db,awayStartingGoalieID,allPlayers,boxscore,isStarting=False,isAway=True),
          **forwards(db,homeForwardIds,allPlayers,boxscore,isAway=False),
          **defense(db,homeDefenseIds,allPlayers,boxscore,isAway=False),
          **goalie(db,homeStartingGoalieID,allPlayers,boxscore,isStarting=True,isAway=False),
          **goalie(db,homeBackupGoalieID,allPlayers,boxscore,isStarting=False,isAway=False),
        },{
          **base_inputs(db,awayTeam,homeTeam,boxscore,gi,startTime,date),
          **forwards(db,awayForwardIds,allPlayers,boxscore,isAway=True),
          **defense(db,awayDefenseIds,allPlayers,boxscore,isAway=True),
          **goalie(db,awayBackupGoalieID,allPlayers,boxscore,isStarting=True,isAway=True),
          **goalie(db,awayStartingGoalieID,allPlayers,boxscore,isStarting=False,isAway=True),
          **forwards(db,homeForwardIds,allPlayers,boxscore,isAway=False),
          **defense(db,homeDefenseIds,allPlayers,boxscore,isAway=False),
          **goalie(db,homeBackupGoalieID,allPlayers,boxscore,isStarting=True,isAway=False),
          **goalie(db,homeStartingGoalieID,allPlayers,boxscore,isStarting=False,isAway=False),
        }
      ]

      
    else:
      all_inputs = {}
      
      af = forwards(db,awayForwardIds,allPlayers,boxscore,isAway=True)
      ad = defense(db,awayDefenseIds,allPlayers,boxscore,isAway=True)
      hf = forwards(db,homeForwardIds,allPlayers,boxscore,isAway=False)
      hd = defense(db,homeDefenseIds,allPlayers,boxscore,isAway=False)
      awayStartingGoalieAge = goalie(db,awayStartingGoalieID,allPlayers,boxscore,isStarting=True,isAway=True)['awayStartingGoalieAge']
      awayBackupGoalieAge = goalie(db,awayBackupGoalieID,allPlayers,boxscore,isStarting=False,isAway=True)['awayBackupGoalieAge']
      homeStartingGoalieAge = goalie(db,homeStartingGoalieID,allPlayers,boxscore,isStarting=True,isAway=False)['homeStartingGoalieAge']
      homeBackupGoalieAge = goalie(db,homeBackupGoalieID,allPlayers,boxscore,isStarting=False,isAway=False)['homeBackupGoalieAge']
      awayForwardAverage = -1 if len(awayForwardIds) == 0 else (sum(awayForwardIds) / len(awayForwardIds))
      awayDefenseAverage = -1 if len(awayDefenseIds) == 0 else (sum(awayDefenseIds) / len(awayDefenseIds))
      awayGoalieAverage = (sum([awayStartingGoalieID,awayBackupGoalieID]) / 2)
      homeForwardAverage = -1 if len(homeForwardIds) == 0 else (sum(homeForwardIds) / len(homeForwardIds))
      homeDefenseAverage = -1 if len(homeDefenseIds) == 0 else (sum(homeDefenseIds) / len(homeDefenseIds))
      homeGoalieAverage = (sum([homeStartingGoalieID,homeBackupGoalieID]) / 2)
      awayForwardAges = []
      awayDefenseAges = []
      awayGoalieAges = [awayStartingGoalieAge,awayBackupGoalieAge]
      homeForwardAges = []
      homeDefenseAges = []
      homeGoalieAges = [homeStartingGoalieAge,homeBackupGoalieAge]
      for i in range(0,13):
        if not af[f'awayForward{i+1}Age'] == -1:
          awayForwardAges.append(af[f'awayForward{i+1}Age'])
        if not hf[f'homeForward{i+1}Age'] == -1:
          homeForwardAges.append(hf[f'homeForward{i+1}Age'])
      for i in range(0,7):
        if not ad[f'awayDefenseman{i+1}Age'] == -1:
          awayDefenseAges.append(ad[f'awayDefenseman{i+1}Age'])
        if not hd[f'homeDefenseman{i+1}Age'] == -1:
          homeDefenseAges.append(hd[f'homeDefenseman{i+1}Age'])
      
      awayForwardAverageAge = -1 if len(awayForwardAges) == 0 else (sum(awayForwardAges) / len(awayForwardAges))
      awayDefenseAverageAge = -1 if len(awayDefenseAges) == 0 else (sum(awayDefenseAges) / len(awayDefenseAges))
      awayGoalieAverageAge = -1 if len(awayGoalieAges) == 0 else (sum(awayGoalieAges) / len(awayGoalieAges))
      homeForwardAverageAge = -1 if len(homeForwardAges) == 0 else (sum(homeForwardAges) / len(homeForwardAges))
      homeDefenseAverageAge = -1 if len(homeDefenseAges) == 0 else (sum(homeDefenseAges) / len(homeDefenseAges))
      homeGoalieAverageAge = -1 if len(homeGoalieAges) == 0 else (sum(homeGoalieAges) / len(homeGoalieAges))

      all_inputs = {
        **base_inputs(db,awayTeam,homeTeam,boxscore,gi,startTime,date),
        'awayForwardAverage': awayForwardAverage,
        'awayDefenseAverage': awayDefenseAverage,
        'awayGoalieAverage': awayGoalieAverage,
        'homeForwardAverage': homeForwardAverage,
        'homeDefenseAverage': homeDefenseAverage,
        'homeGoalieAverage': homeGoalieAverage,
        'awayForwardAverageAge': awayForwardAverageAge,
        'awayDefenseAverageAge': awayDefenseAverageAge,
        'awayGoalieAverageAge': awayGoalieAverageAge,
        'homeForwardAverageAge': homeForwardAverageAge,
        'homeDefenseAverageAge': homeDefenseAverageAge,
        'homeGoalieAverageAge': homeGoalieAverageAge,
      }
  else:
    all_inputs = {
      **base_inputs(db,awayTeam,homeTeam,boxscore,gi,startTime,date),
      **forwards(db,awayForwardIds,allPlayers,boxscore,isAway=True),
      **defense(db,awayDefenseIds,allPlayers,boxscore,isAway=True),
      **goalie(db,awayStartingGoalieID,allPlayers,boxscore,isStarting=True,isAway=True),
      **goalie(db,awayBackupGoalieID,allPlayers,boxscore,isStarting=False,isAway=True),
      **forwards(db,homeForwardIds,allPlayers,boxscore,isAway=False),
      **defense(db,homeDefenseIds,allPlayers,boxscore,isAway=False),
      **goalie(db,homeStartingGoalieID,allPlayers,boxscore,isStarting=True,isAway=False),
      **goalie(db,homeBackupGoalieID,allPlayers,boxscore,isStarting=False,isAway=False),
    }
    if isProjectedRoster:
      awayForwardAverage = safe_chain(pbgs,'awayTeam','forwardAverage')
      awayDefenseAverage = safe_chain(pbgs,'awayTeam','defenseAverage')
      awayGoalieAverage = safe_chain(pbgs,'awayTeam','goalieAverage')
      homeForwardAverage = safe_chain(pbgs,'homeTeam','forwardAverage')
      homeDefenseAverage = safe_chain(pbgs,'homeTeam','defenseAverage')
      homeGoalieAverage = safe_chain(pbgs,'homeTeam','goalieAverage')
      awayForwardAverageAge = safe_chain(pbgs,'awayTeam','forwardAverageAge')
      awayDefenseAverageAge = safe_chain(pbgs,'awayTeam','defenseAverageAge')
      awayGoalieAverageAge = safe_chain(pbgs,'awayTeam','goalieAverageAge')
      homeForwardAverageAge = safe_chain(pbgs,'homeTeam','forwardAverageAge')
      homeDefenseAverageAge = safe_chain(pbgs,'homeTeam','defenseAverageAge')
      homeGoalieAverageAge = safe_chain(pbgs,'homeTeam','goalieAverageAge')
    else:
      awayForwardAverage = -1 if len(awayForwardIds) == 0 else (sum(awayForwardIds) / len(awayForwardIds))
      awayDefenseAverage = -1 if len(awayDefenseIds) == 0 else (sum(awayDefenseIds) / len(awayDefenseIds))
      awayGoalieAverage = (sum([awayStartingGoalieID,awayBackupGoalieID]) / 2)
      homeForwardAverage = -1 if len(homeForwardIds) == 0 else (sum(homeForwardIds) / len(homeForwardIds))
      homeDefenseAverage = -1 if len(homeDefenseIds) == 0 else (sum(homeDefenseIds) / len(homeDefenseIds))
      homeGoalieAverage = (sum([homeStartingGoalieID,homeBackupGoalieID]) / 2)
      awayForwardAges = []
      awayDefenseAges = []
      awayGoalieAges = [all_inputs['awayStartingGoalieAge'],all_inputs['awayBackupGoalieAge']]
      homeForwardAges = []
      homeDefenseAges = []
      homeGoalieAges = [all_inputs['homeStartingGoalieAge'],all_inputs['homeBackupGoalieAge']]
      for i in range(0,13):
        if not all_inputs[f'awayForward{i+1}Age'] == -1:
          awayForwardAges.append(all_inputs[f'awayForward{i+1}Age'])
        if not all_inputs[f'homeForward{i+1}Age'] == -1:
          homeForwardAges.append(all_inputs[f'homeForward{i+1}Age'])
      for i in range(0,7):
        if not all_inputs[f'awayDefenseman{i+1}Age'] == -1:
          awayDefenseAges.append(all_inputs[f'awayDefenseman{i+1}Age'])
        if not all_inputs[f'homeDefenseman{i+1}Age'] == -1:
          homeDefenseAges.append(all_inputs[f'homeDefenseman{i+1}Age'])
      
      awayForwardAverageAge = -1 if len(awayForwardAges) == 0 else (sum(awayForwardAges) / len(awayForwardAges))
      awayDefenseAverageAge = -1 if len(awayDefenseAges) == 0 else (sum(awayDefenseAges) / len(awayDefenseAges))
      awayGoalieAverageAge = -1 if len(awayGoalieAges) == 0 else (sum(awayGoalieAges) / len(awayGoalieAges))
      homeForwardAverageAge = -1 if len(homeForwardAges) == 0 else (sum(homeForwardAges) / len(homeForwardAges))
      homeDefenseAverageAge = -1 if len(homeDefenseAges) == 0 else (sum(homeDefenseAges) / len(homeDefenseAges))
      homeGoalieAverageAge = -1 if len(homeGoalieAges) == 0 else (sum(homeGoalieAges) / len(homeGoalieAges))

    all_inputs['awayForwardAverage'] = awayForwardAverage
    all_inputs['awayDefenseAverage'] = awayDefenseAverage
    all_inputs['awayGoalieAverage'] = awayGoalieAverage
    all_inputs['homeForwardAverage'] = homeForwardAverage
    all_inputs['homeDefenseAverage'] = homeDefenseAverage
    all_inputs['homeGoalieAverage'] = homeGoalieAverage
    all_inputs['awayForwardAverageAge'] = awayForwardAverageAge
    all_inputs['awayDefenseAverageAge'] = awayDefenseAverageAge
    all_inputs['awayGoalieAverageAge'] = awayGoalieAverageAge
    all_inputs['homeForwardAverageAge'] = homeForwardAverageAge
    all_inputs['homeDefenseAverageAge'] = homeDefenseAverageAge
    all_inputs['homeGoalieAverageAge'] = homeGoalieAverageAge
    
  # print('all_inputs',all_inputs)
  return {
    'data': all_inputs,
    'options': {
      'projectedLineup': isProjectedLineup,
      'projectedRoster': isProjectedRoster,
    },
  }
  
  # try:
  # except Exception as error:
  #   print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
  #   print('ERROR','INPUTS')
  #   print('id', safe_chain(game,'id'))
  #   print(safe_chain(game,'homeTeam'))
  #   print(safe_chain(game,'awayTeam'))
  #   print('error',error)
  #   print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def master_inputs2(db, boxscore, isProjectedRoster=False, isProjectedLineup=False, training=False):
  pass