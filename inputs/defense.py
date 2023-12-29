import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

from util.helpers import getPlayer, getAge, pad_list

def defense(ids,allPlayers,game,isAway=True,REPLACE_VALUE=-1):
  homeAway = 'away' if isAway else 'home'

  defense_dict = {}

  ids = pad_list(ids,7,-1)

  for i in range(0,len(ids)):
    defense_dict[f'{homeAway}Defenseman{i+1}'] = ids[i]
    defense_dict[f'{homeAway}Defenseman{i+1}Age'] = getAge(getPlayer(allPlayers,ids[i]),game['gameDate']) if ids[i] != -1 else REPLACE_VALUE

  return defense_dict