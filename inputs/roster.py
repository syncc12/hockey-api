import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from util.helpers import safe_chain, n2n, false_chain, getPlayer, getAge, b2n, isNaN, formatDate, formatDatetime, formatTime, safe_none, collect_players, projected_roster, pad_list
from util.query import get_last_game_player_stats, get_last_game_team_stats, last_player_game_stats, list_players_in_game

REPLACE_VALUE = -1

POOL_FORWARDS = 32 # min: 12 max: 32 average: 20.672849915682967
POOL_DEFENSE = 21 # min: 6 max: 21 average: 10.806070826306915
POOL_GOALIES = 8 # min: 2 max: 8 average: 3.1720067453625633

def roster_forwards(ids):
  forward_dict = {}
  ids = pad_list(ids,13,-1)
  for i in range(0,len(ids)):
    forward_dict[f'forward{i+1}'] = ids[i]
  return forward_dict

def roster_defense(ids):
  defenseman_dict = {}
  ids = pad_list(ids,7,-1)
  for i in range(0,len(ids)):
    defenseman_dict[f'defenseman{i+1}'] = ids[i]
  return defenseman_dict

def forwards(ids):
  forward_dict = {}
  ids = pad_list(ids,POOL_FORWARDS,-1)
  for i in range(0,len(ids)):
    forward_dict[f'forwardPool{i+1}'] = ids[i]
  return forward_dict

def defense(ids):
  defenseman_dict = {}
  ids = pad_list(ids,POOL_DEFENSE,-1)
  for i in range(0,len(ids)):
    defenseman_dict[f'defensemanPool{i+1}'] = ids[i]
  return defenseman_dict

def goalies(ids):
  goalie_dict = {}
  ids = pad_list(ids,POOL_GOALIES,-1)
  for i in range(0,len(ids)):
    goalie_dict[f'goaliePool{i+1}'] = ids[i]
  return goalie_dict

def roster_inputs(db, boxscore, roster, homeAway='homeTeam'):
  try:
    date = REPLACE_VALUE
    if 'gameDate' in boxscore and boxscore['gameDate']:
      date = formatDate(boxscore['gameDate'])
    team_ha = homeAway
    opponent_ha = 'awayTeam' if homeAway == 'homeTeam' else 'homeTeam'
    team = safe_chain(boxscore,team_ha)
    opponent = safe_chain(boxscore,opponent_ha)
    gi = safe_chain(boxscore,'boxscore','gameInfo')
    pbgs = safe_chain(boxscore,'boxscore','playerByGameStats')
    allPlayerIds = []
    forwardIds = []
    defenseIds = []
    goalieIds = []
    if safe_chain(pbgs,homeAway,'forwards') != REPLACE_VALUE:
      for p in pbgs[homeAway]['forwards']:
        allPlayerIds.append({'playerId': p['playerId']})
        forwardIds.append(p['playerId'])
    if safe_chain(pbgs,homeAway,'defense') != REPLACE_VALUE:
      for p in pbgs[homeAway]['defense']:
        allPlayerIds.append({'playerId': p['playerId']})
        defenseIds.append(p['playerId'])
    if safe_chain(pbgs,homeAway,'goalies') != REPLACE_VALUE:
      for p in pbgs[homeAway]['goalies']:
        allPlayerIds.append({'playerId': p['playerId']})
        goalieIds.append(p['playerId'])
    
    startingTOI = safe_none(formatTime(pbgs[homeAway]['goalies'][0]['toi']))
    startingID =  pbgs[homeAway]['goalies'][0]['playerId']
    backupID = pbgs[homeAway]['goalies'][1]['playerId']
    for g in pbgs[homeAway]['goalies']:
      if safe_none(formatTime(g['toi'])) > safe_none(startingTOI):
        startingTOI = safe_none(formatTime(g['toi']))
        backupID = startingID
        startingID = g['playerId']
    startingGoalie = startingID
    backupGoalie = backupID
    
    # print(safe_chain(boxscore,'id'),safe_chain(boxscore,'season'),team['id'],opponent['id'],roster)
    forwardPoolIds = [p['id'] for p in roster['roster']['forwards']]
    defensePoolIds = [p['id'] for p in roster['roster']['defensemen']]
    goaliePoolIds = [p['id'] for p in roster['roster']['goalies']]

    all_inputs = {
      'id': safe_chain(boxscore,'id'),
      'season': safe_chain(boxscore,'season'),
      'gameType': safe_chain(boxscore,'gameType'),
      'venue': n2n(safe_chain(boxscore,'venue','default')),
      'venueT': str(safe_chain(boxscore,'venue','default',default=0)),
      'neutralSite': b2n(safe_chain(boxscore,'neutralSite')),
      'neutralSiteB': safe_chain(boxscore,'neutralSite'),
      'team': team['id'],
      'opponent': opponent['id'],
      'date': date,
      'headCoach': n2n(safe_chain(gi,homeAway,'headCoach','default')),
      'headCoachT': str(safe_chain(gi,homeAway,'headCoach','default',default=0)),
      'opponentHeadCoach': n2n(safe_chain(gi,'homeTeam','headCoach','default')),
      'opponentHeadCoachT': str(safe_chain(gi,'homeTeam','headCoach','default',default=0)),
      **roster_forwards(forwardIds),
      **roster_defense(defenseIds),
      **roster_forwards(forwardIds),
      'startingGoalie': startingGoalie,
      'backupGoalie': backupGoalie,
      **forwards(forwardPoolIds),
      **defense(defensePoolIds),
      **goalies(goaliePoolIds),
    }

    return all_inputs
  except:
    return False