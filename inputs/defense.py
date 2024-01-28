import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from util.helpers import getPlayer, getAge, pad_list, getGamesPlayed, getAllGamesPlayed, getPlayerStats

def defense(db,ids,allPlayers,game,isAway=True,REPLACE_VALUE=-1,prefix=''):
  homeAway = 'away' if isAway else 'home'

  defense_dict = {}

  ids = pad_list(ids,7,-1)

  # all_games_played = getAllGamesPlayed(db,ids,game['id'],'defense')

  for i in range(0,len(ids)):
    # player_stats = getPlayerStats(db,ids[i],game['season'],game['id'],'defense')
    defense_dict[f'{prefix}{homeAway}Defenseman{i+1}'] = ids[i]
    defense_dict[f'{prefix}{homeAway}Defenseman{i+1}Age'] = getAge(getPlayer(allPlayers,ids[i]),game['gameDate']) if ids[i] != -1 else REPLACE_VALUE
    # defense_dict[f'{homeAway}Defenseman{i+1}GamesPlayed'] = all_games_played[ids[i]]
    # defense_dict[f'{homeAway}Defenseman{i+1}Goals'] = player_stats['goals']
    # defense_dict[f'{homeAway}Defenseman{i+1}Assists'] = player_stats['assists']
    # defense_dict[f'{homeAway}Defenseman{i+1}Points'] = player_stats['points']

  return defense_dict