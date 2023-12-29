import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

from util.helpers import safe_chain, getPlayer, getAge, n2n, false_chain


def goalies(pbgs,allPlayers,game,isAway=True,REPLACE_VALUE=-1):
  homeAway = 'away' if isAway else 'home'
  return {
    f'{homeAway}StartingGoalie': safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId'),
    f'{homeAway}StartingGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId')),0,'shootsCatches')),
    f'{homeAway}StartingGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId')),game['gameDate']) if false_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId') else REPLACE_VALUE,
    f'{homeAway}StartingGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId')),0,'heightInInches'),
    f'{homeAway}BackupGoalie': safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId'),
    f'{homeAway}BackupGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId')),0,'shootsCatches')),
    f'{homeAway}BackupGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId')),game['gameDate']) if false_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId') else REPLACE_VALUE,
    f'{homeAway}BackupGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId')),0,'heightInInches'),
  }

def goalie(playerId,allPlayers,game,isStarting=True,isAway=True,REPLACE_VALUE=-1):
  homeAway = 'away' if isAway else 'home'
  startingBackup = 'Starting' if isStarting else 'Backup'
  return {
    f'{homeAway}{startingBackup}Goalie': playerId,
    f'{homeAway}{startingBackup}GoalieCatches': n2n(safe_chain(getPlayer(allPlayers,playerId),0,'shootsCatches')),
    f'{homeAway}{startingBackup}GoalieAge': getAge(getPlayer(allPlayers,playerId),game['gameDate']) if playerId != -1 else REPLACE_VALUE,
    f'{homeAway}{startingBackup}GoalieHeight': safe_chain(getPlayer(allPlayers,playerId),0,'heightInInches'),
  }