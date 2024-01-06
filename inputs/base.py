import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

from util.helpers import safe_chain, getPlayer, getAge, n2n, b2n


def base_inputs(db,awayTeam,homeTeam,game,gi,startTime,date):
  return {
    'id': safe_chain(game,'id'),
    'season': safe_chain(game,'season'),
    'gameType': safe_chain(game,'gameType'),
    'venue': n2n(safe_chain(game,'venue','default')),
    'neutralSite': b2n(safe_chain(game,'neutralSite')),
    'homeTeam': homeTeam['id'],
    'awayTeam': awayTeam['id'],
    'homeScore': safe_chain(homeTeam,'score'),
    'awayScore': safe_chain(awayTeam,'score'),
    'totalGoals': safe_chain(homeTeam,'score') + safe_chain(awayTeam,'score'),
    'goalDifferential': safe_chain(homeTeam,'score') - safe_chain(awayTeam,'score') if safe_chain(homeTeam,'score') >= safe_chain(awayTeam,'score') else safe_chain(awayTeam,'score') - safe_chain(homeTeam,'score'),
    'winner': homeTeam['id'] if safe_chain(homeTeam,'score') > safe_chain(awayTeam,'score') else awayTeam['id'],
    'startTime': startTime,
    'date': date,
    'awayHeadCoach': n2n(safe_chain(gi,'awayTeam','headCoach','default')),
    'homeHeadCoach': n2n(safe_chain(gi,'homeTeam','headCoach','default')),
    'ref1': n2n(safe_chain(gi,'referees',0,'default')),
    'ref2': n2n(safe_chain(gi,'referees',1,'default')),
    'linesman1': n2n(safe_chain(gi,'linesmen',0,'default')),
    'linesman2': n2n(safe_chain(gi,'linesmen',1,'default')),
  }