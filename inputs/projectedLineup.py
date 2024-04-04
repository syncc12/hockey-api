import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api\constants')

from inputs.goalies import goalie
from inputs.forwards import forwards
from inputs.defense import defense
from inputs.base import base_inputs
from util.helpers import safe_chain, false_chain, formatDate, formatDatetime, formatTime, safe_none, projectedLineup
from constants.inputConstants import X_INPUTS

REPLACE_VALUE = -1

def testProjectedLineup(db,game):
  
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

  if false_chain(pbgs,'homeTeam','goalies'):
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

  lineups = [{
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
  {
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
  {
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
  {
    **base_inputs(db,awayTeam,homeTeam,game,gi,startTime,date),
    **forwards(db,awayForwardIds,allPlayers,game,isAway=True),
    **defense(db,awayDefenseIds,allPlayers,game,isAway=True),
    **goalie(db,awayBackupGoalieID,allPlayers,game,isStarting=True,isAway=True),
    **goalie(db,awayStartingGoalieID,allPlayers,game,isStarting=False,isAway=True),
    **forwards(db,homeForwardIds,allPlayers,game,isAway=False),
    **defense(db,homeDefenseIds,allPlayers,game,isAway=False),
    **goalie(db,homeBackupGoalieID,allPlayers,game,isStarting=True,isAway=False),
    **goalie(db,homeStartingGoalieID,allPlayers,game,isStarting=False,isAway=False),
  }]

  lineup_data = []
  for lineup in lineups:
    lineup_data.append([lineup[i] for i in X_INPUTS])

  return lineup_data

def last_game(test_data, team_id, game_id):
  test_data = [data for data in test_data if (data['homeTeam'] == team_id or data['awayTeam'] == team_id) and data['id'] < game_id]
  if len(test_data) == 0:
    return None, None
  last_game = max(test_data, key=lambda x: x.get('id', float('-inf')))
  home_away = 'home' if last_game['homeTeam'] == team_id else 'away'
  return last_game, home_away

def projected_shift(test_data, y_test):
  shift_keys = ['HeadCoach','HeadCoachT','Forward1','Forward1Age','Forward2','Forward2Age','Forward3','Forward3Age','Forward4','Forward4Age','Forward5','Forward5Age','Forward6','Forward6Age','Forward7','Forward7Age','Forward8','Forward8Age','Forward9','Forward9Age','Forward10','Forward10Age','Forward11','Forward11Age','Forward12','Forward12Age','Forward13','Forward13Age','Defenseman1','Defenseman1Age','Defenseman2','Defenseman2Age','Defenseman3','Defenseman3Age','Defenseman4','Defenseman4Age','Defenseman5','Defenseman5Age','Defenseman6','Defenseman6Age','Defenseman7','Defenseman7Age','StartingGoalie','StartingGoalieCatches','StartingGoalieCatchesT','StartingGoalieAge','BackupGoalie','BackupGoalieCatches','BackupGoalieCatchesT','BackupGoalieAge','ForwardAverage','DefenseAverage','GoalieAverage','ForwardAverageAge','DefenseAverageAge','GoalieAverageAge']
  # away_shift_keys = ['awayHeadCoach','awayHeadCoachT','awayForward1','awayForward1Age','awayForward2','awayForward2Age','awayForward3','awayForward3Age','awayForward4','awayForward4Age','awayForward5','awayForward5Age','awayForward6','awayForward6Age','awayForward7','awayForward7Age','awayForward8','awayForward8Age','awayForward9','awayForward9Age','awayForward10','awayForward10Age','awayForward11','awayForward11Age','awayForward12','awayForward12Age','awayForward13','awayForward13Age','awayDefenseman1','awayDefenseman1Age','awayDefenseman2','awayDefenseman2Age','awayDefenseman3','awayDefenseman3Age','awayDefenseman4','awayDefenseman4Age','awayDefenseman5','awayDefenseman5Age','awayDefenseman6','awayDefenseman6Age','awayDefenseman7','awayDefenseman7Age','awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieCatchesT','awayStartingGoalieAge','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieCatchesT','awayBackupGoalieAge','awayForwardAverage','awayDefenseAverage','awayGoalieAverage','awayForwardAverageAge','awayDefenseAverageAge','awayGoalieAverageAge']
  # home_shift_keys = ['homeHeadCoach','homeHeadCoachT','homeForward1','homeForward1Age','homeForward2','homeForward2Age','homeForward3','homeForward3Age','homeForward4','homeForward4Age','homeForward5','homeForward5Age','homeForward6','homeForward6Age','homeForward7','homeForward7Age','homeForward8','homeForward8Age','homeForward9','homeForward9Age','homeForward10','homeForward10Age','homeForward11','homeForward11Age','homeForward12','homeForward12Age','homeForward13','homeForward13Age','homeDefenseman1','homeDefenseman1Age','homeDefenseman2','homeDefenseman2Age','homeDefenseman3','homeDefenseman3Age','homeDefenseman4','homeDefenseman4Age','homeDefenseman5','homeDefenseman5Age','homeDefenseman6','homeDefenseman6Age','homeDefenseman7','homeDefenseman7Age','homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieCatchesT','homeStartingGoalieAge','homeBackupGoalie','homeBackupGoalieCatches','homeBackupGoalieCatchesT','homeBackupGoalieAge','homeForwardAverage','homeDefenseAverage','homeGoalieAverage','homeForwardAverageAge','homeDefenseAverageAge','homeGoalieAverageAge']
  projected_test_data = []
  projected_y_test = []
  for i, data in enumerate(test_data):
    last_awayTeam_game, away_home_away = last_game(test_data, data['awayTeam'], data['id'])
    last_homeTeam_game, home_home_away = last_game(test_data, data['homeTeam'], data['id'])
    if last_homeTeam_game is None or last_awayTeam_game is None:
      continue
    else:
      for ii in shift_keys:
        data[f'away{ii}'] = last_awayTeam_game[f'{away_home_away}{ii}']
        data[f'home{ii}'] = last_homeTeam_game[f'{home_home_away}{ii}']
      projected_test_data.append(data)
      projected_y_test.append(y_test[i])
  return projected_test_data, projected_y_test


