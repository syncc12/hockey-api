import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

from util.helpers import getPlayer, getAge, pad_list

def forwards(ids,allPlayers,game,isAway=True,REPLACE_VALUE=-1):
  homeAway = 'away' if isAway else 'home'

  forward_dict = {}

  ids = pad_list(ids,13,-1)

  for i in range(0,len(ids)):
    forward_dict[f'{homeAway}Forward{i+1}'] = ids[i]
    forward_dict[f'{homeAway}Forward{i+1}Age'] = getAge(getPlayer(allPlayers,ids[i]),game['gameDate']) if ids[i] != -1 else REPLACE_VALUE

  return forward_dict