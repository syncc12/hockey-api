import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from util.helpers import safe_chain, getPlayer, getAge, n2n, false_chain, getGamesPlayed


def goalies(db,pbgs,allPlayers,game,isAway=True,REPLACE_VALUE=-1,prefix=''):
  homeAway = 'away' if isAway else 'home'
  startingId = safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId')
  backupId = safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId')
  return {
    f'{prefix}{homeAway}StartingGoalie': startingId,
    f'{prefix}{homeAway}StartingGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,startingId),0,'shootsCatches')),
    f'{prefix}{homeAway}StartingGoalieCatchesT': str(safe_chain(getPlayer(allPlayers,startingId),0,'shootsCatches',default=0)),
    f'{prefix}{homeAway}StartingGoalieAge': getAge(getPlayer(allPlayers,startingId),game['gameDate']) if false_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId') else REPLACE_VALUE,
    f'{prefix}{homeAway}StartingGoalieHeight': safe_chain(getPlayer(allPlayers,startingId),0,'heightInInches'),
    f'{prefix}{homeAway}StartingGoalieGamesPlayed': getGamesPlayed(db,startingId,game['id'],'goalies'),
    f'{prefix}{homeAway}BackupGoalie': backupId,
    f'{prefix}{homeAway}BackupGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,backupId),0,'shootsCatches')),
    f'{prefix}{homeAway}BackupGoalieCatchesT': str(safe_chain(getPlayer(allPlayers,backupId),0,'shootsCatches',default=0)),
    f'{prefix}{homeAway}BackupGoalieAge': getAge(getPlayer(allPlayers,backupId),game['gameDate']) if false_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId') else REPLACE_VALUE,
    f'{prefix}{homeAway}BackupGoalieHeight': safe_chain(getPlayer(allPlayers,backupId),0,'heightInInches'),
  }

def goalie(db,playerId,allPlayers,game,isStarting=True,isAway=True,REPLACE_VALUE=-1,prefix=''):
  homeAway = 'away' if isAway else 'home'
  startingBackup = 'Starting' if isStarting else 'Backup'
  return {
    f'{prefix}{homeAway}{startingBackup}Goalie': playerId,
    f'{prefix}{homeAway}{startingBackup}GoalieCatches': n2n(safe_chain(getPlayer(allPlayers,playerId),0,'shootsCatches')),
    f'{prefix}{homeAway}{startingBackup}GoalieCatchesT': str(safe_chain(getPlayer(allPlayers,playerId),0,'shootsCatches',default=0)),
    f'{prefix}{homeAway}{startingBackup}GoalieAge': getAge(getPlayer(allPlayers,playerId),game['gameDate']) if playerId != -1 else REPLACE_VALUE,
    # # f'{homeAway}{startingBackup}GoalieHeight': safe_chain(getPlayer(allPlayers,playerId),0,'heightInInches'),
    # f'{homeAway}{startingBackup}GoalieGamesPlayed': getGamesPlayed(db,playerId,game['id'],'goalies'),
  }