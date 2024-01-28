import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from util.helpers import getPlayer, getAge, pad_list, getGamesPlayed, getAllGamesPlayed, getPlayerStats

def forwards(db,ids,allPlayers,game,isAway=True,REPLACE_VALUE=-1,prefix=''):
  homeAway = 'away' if isAway else 'home'

  forward_dict = {}

  ids = pad_list(ids,13,-1)

  # all_games_played = getAllGamesPlayed(db,ids,game['id'],'forwards')

  for i in range(0,len(ids)):
    # player_stats = getPlayerStats(db,ids[i],game['season'],game['id'],'forwards')
    forward_dict[f'{prefix}{homeAway}Forward{i+1}'] = ids[i]
    forward_dict[f'{prefix}{homeAway}Forward{i+1}Age'] = getAge(getPlayer(allPlayers,ids[i]),game['gameDate']) if ids[i] != -1 else REPLACE_VALUE
    # forward_dict[f'{homeAway}Forward{i+1}GamesPlayed'] = all_games_played[ids[i]]
    # forward_dict[f'{homeAway}Forward{i+1}Goals'] = player_stats['goals']
    # forward_dict[f'{homeAway}Forward{i+1}Assists'] = player_stats['assists']
    # forward_dict[f'{homeAway}Forward{i+1}Points'] = player_stats['points']


  return forward_dict