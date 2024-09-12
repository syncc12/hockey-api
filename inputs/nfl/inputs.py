import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs\nfl')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from util.helpers import safe_chain, n2n
import math
import requests
from pymongo import MongoClient
from datetime import datetime
from inputs.nfl.nfl_helpers import formatDatetime

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["nfl"]
Games = db["games"]

def master_inputs(game):

  # Participation = db["participation"]
  Officials = db["officials"]

  homeTeam = game['homeTeamId']
  awayTeam = game['visitorTeamId']
  gameId = game['gameId']

  # homeRoster = list(Participation.find(
  #   {'gameId': gameId, 'teamId': homeTeam},
  #   {'_id':0,'playerId':1,'position':1,'dob':1}
  # ))

  # awayRoster = list(Participation.find(
  #   {'gameId': gameId, 'teamId': awayTeam},
  #   {'_id':0,'playerId':1,'position':1,'dob':1}
  # ))

  # print('homeRoster',homeTeam, len(homeRoster))
  # print('awayRoster',awayTeam, len(awayRoster))

  gameOfficials = list(Officials.find(
    {'gameId': gameId},
    {'_id':0,'officialId':1,'officialPosition':1}
  ))

  refs = {}
  for official in gameOfficials:
    splitPosition = official['officialPosition'].split(' ')
    refPosition = []
    for i in range(0, len(splitPosition)):
      p = splitPosition[i]
      if i == 0:
        refPosition.append(p.lower())
      else:
        refPosition.append(p)
    refPosition = ''.join(refPosition)
    refs[refPosition] = official['officialId']

  homeScore = game['homeTeamFinalScore']
  awayScore = game['visitingTeamFinalScore']
  scoreDifferential = abs(homeScore - awayScore)
  return {
    'gameId': gameId,
    'gameType': n2n(game['seasonType']),
    'season': game['season'],
    'week': game['week'],
    'date': formatDatetime(game['gameDate']),
    'venue': safe_chain(game,'siteId'),
    'homeTeam': homeTeam,
    'awayTeam': awayTeam,
    'homeScore': homeScore,
    'awayScore': awayScore,
    'winner': game['winningTeam'],
    'totalScore': homeScore + awayScore,
    'scoreDifferential': scoreDifferential,
    'lineJudge': safe_chain(refs,'lineJudge'),
    'referee': safe_chain(refs,'referee'),
    'videoOperator': safe_chain(refs,'videoOperator'),
    'replayAssistant': safe_chain(refs,'replayAssistant'),
    'umpire': safe_chain(refs,'umpire'),
    'headLinesman': safe_chain(refs,'headLinesman'),
    'backJudge': safe_chain(refs,'backJudge'),
    'sideJudge': safe_chain(refs,'sideJudge'),
    'fieldJudge': safe_chain(refs,'fieldJudge'),
  }
