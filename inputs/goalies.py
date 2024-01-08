import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

from util.helpers import safe_chain, getPlayer, getAge, n2n, false_chain, getGamesPlayed


def goalies(db,pbgs,allPlayers,game,isAway=True,REPLACE_VALUE=-1):
  homeAway = 'away' if isAway else 'home'
  startingId = safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId')
  backupId = safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId')
  return {
    f'{homeAway}StartingGoalie': startingId,
    f'{homeAway}StartingGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,startingId),0,'shootsCatches')),
    f'{homeAway}StartingGoalieAge': getAge(getPlayer(allPlayers,startingId),game['gameDate']) if false_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId') else REPLACE_VALUE,
    f'{homeAway}StartingGoalieHeight': safe_chain(getPlayer(allPlayers,startingId),0,'heightInInches'),
    f'{homeAway}StartingGoalieGamesPlayed': getGamesPlayed(db,startingId,game['id'],'goalies'),
    f'{homeAway}BackupGoalie': backupId,
    f'{homeAway}BackupGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,backupId),0,'shootsCatches')),
    f'{homeAway}BackupGoalieAge': getAge(getPlayer(allPlayers,backupId),game['gameDate']) if false_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId') else REPLACE_VALUE,
    f'{homeAway}BackupGoalieHeight': safe_chain(getPlayer(allPlayers,backupId),0,'heightInInches'),
  }

def goalie(db,playerId,allPlayers,game,isStarting=True,isAway=True,REPLACE_VALUE=-1):
  homeAway = 'away' if isAway else 'home'
  startingBackup = 'Starting' if isStarting else 'Backup'
  return {
    f'{homeAway}{startingBackup}Goalie': playerId,
    f'{homeAway}{startingBackup}GoalieCatches': n2n(safe_chain(getPlayer(allPlayers,playerId),0,'shootsCatches')),
    f'{homeAway}{startingBackup}GoalieAge': getAge(getPlayer(allPlayers,playerId),game['gameDate']) if playerId != -1 else REPLACE_VALUE,
    # f'{homeAway}{startingBackup}GoalieHeight': safe_chain(getPlayer(allPlayers,playerId),0,'heightInInches'),
    f'{homeAway}{startingBackup}GoalieGamesPlayed': getGamesPlayed(db,playerId,game['id'],'goalies'),
  }