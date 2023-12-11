from util.helpers import safe_chain, getPlayer, getAge, n2n, false_chain
def goalies(pbgs,allPlayers,game,isAway=True,REPLACE_VALUE=-1):
  homeAway = 'away' if isAway else 'home'
  return {
    f'{homeAway}StartingGoalie': safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId'),
    f'{homeAway}StartingGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId')),0,'shootsCatches')),
    f'{homeAway}StartingGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId')),game['gameDate']) if false_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId') else REPLACE_VALUE,
    f'{homeAway}StartingGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId')),0,'heightInInches'),
    f'{homeAway}StartingGoalieWeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',0,'playerId')),0,'weightInPounds'),
    f'{homeAway}BackupGoalie': safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId'),
    f'{homeAway}BackupGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId')),0,'shootsCatches')),
    f'{homeAway}BackupGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId')),game['gameDate']) if false_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId') else REPLACE_VALUE,
    f'{homeAway}BackupGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId')),0,'heightInInches'),
    f'{homeAway}BackupGoalieWeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',1,'playerId')),0,'weightInPounds'),
    # f'{homeAway}ThirdGoalie': safe_chain(pbgs,f'{homeAway}Team','goalies',2,'playerId'),
    # f'{homeAway}ThirdGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',2,'playerId')),0,'shootsCatches')),
    # f'{homeAway}ThirdGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',2,'playerId')),game['gameDate']) if false_chain(pbgs,f'{homeAway}Team','goalies',2,'playerId') else REPLACE_VALUE,
    # f'{homeAway}ThirdGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',2,'playerId')),0,'heightInInches'),
    # f'{homeAway}ThirdGoalieWeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,f'{homeAway}Team','goalies',2,'playerId')),0,'weightInPounds'),
  }