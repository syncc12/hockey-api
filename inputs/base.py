import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from util.helpers import safe_chain, getPlayer, getAge, n2n, b2n, parse_start_time, parse_utc_offset
from sklearn.preprocessing import LabelEncoder


def base_inputs(db,awayTeam,homeTeam,game,gi,startTime,date):
  homeId = safe_chain(homeTeam,'id')
  awayId = safe_chain(awayTeam,'id')
  start_date, start_time = parse_start_time(safe_chain(game,'startTimeUTCOffset')) if safe_chain(game,'startTimeUTCOffset') else (-1,-1)
  return {
    'id': safe_chain(game,'id'),
    'season': safe_chain(game,'season'),
    'gameType': safe_chain(game,'gameType'),
    'venue': n2n(safe_chain(game,'venue','default')),
    'venueT': str(safe_chain(game,'venue','default',default=0)),
    'neutralSite': b2n(safe_chain(game,'neutralSite')),
    'neutralSiteB': safe_chain(game,'neutralSite'),
    'homeTeam': homeId,
    'awayTeam': awayId,
    'homeScore': safe_chain(homeTeam,'score'),
    'awayScore': safe_chain(awayTeam,'score'),
    'totalGoals': safe_chain(homeTeam,'score') + safe_chain(awayTeam,'score'),
    'goalDifferential': safe_chain(homeTeam,'score') - safe_chain(awayTeam,'score') if safe_chain(homeTeam,'score') >= safe_chain(awayTeam,'score') else safe_chain(awayTeam,'score') - safe_chain(homeTeam,'score'),
    'winner': homeId if safe_chain(homeTeam,'score') > safe_chain(awayTeam,'score') else awayId,
    'winnerB': 0 if safe_chain(homeTeam,'score') > safe_chain(awayTeam,'score') else 1,
    'startTime': startTime,
    'easternOffset': parse_utc_offset(safe_chain(game,'easternUTCOffset')),
    'venueOffset':  parse_utc_offset(safe_chain(game,'venueUTCOffset')),
    'date': date,
    'awayHeadCoach': n2n(safe_chain(gi,'awayTeam','headCoach','default')),
    'awayHeadCoachT': str(safe_chain(gi,'awayTeam','headCoach','default',default=0)),
    'homeHeadCoach': n2n(safe_chain(gi,'homeTeam','headCoach','default')),
    'homeHeadCoachT': str(safe_chain(gi,'homeTeam','headCoach','default',default=0)),
    'ref1': n2n(safe_chain(gi,'referees',0,'default')),
    'ref1T': str(safe_chain(gi,'referees',0,'default',default=0)),
    'ref2': n2n(safe_chain(gi,'referees',1,'default')),
    'ref2T': str(safe_chain(gi,'referees',1,'default',default=0)),
    'linesman1': n2n(safe_chain(gi,'linesmen',0,'default')),
    'linesman1T': str(safe_chain(gi,'linesmen',0,'default',default=0)),
    'linesman2': n2n(safe_chain(gi,'linesmen',1,'default')),
    'linesman2T': str(safe_chain(gi,'linesmen',1,'default',default=0)),
    'matchupHomeScores': safe_chain(game,'pregame','matchup',homeId,'scores'),
    'matchupHomeShots': safe_chain(game,'pregame','matchup',homeId,'shots'),
    'matchupHomeHits': safe_chain(game,'pregame','matchup',homeId,'hits'),
    'matchupHomePIM': safe_chain(game,'pregame','matchup',homeId,'pim'),
    'matchupHomePowerplays': safe_chain(game,'pregame','matchup',homeId,'powerplays'),
    'matchupHomePowerplayGoals': safe_chain(game,'pregame','matchup',homeId,'powerplayGoals'),
    'matchupHomeFaceoffWinPercent': safe_chain(game,'pregame','matchup',homeId,'faceoffWinPercent'),
    'matchupHomeBlocks': safe_chain(game,'pregame','matchup',homeId,'blocks'),
    'matchupHomeWinLoss': safe_chain(game,'pregame','matchup',homeId,'winLoss'),
    'matchupAwayScores': safe_chain(game,'pregame','matchup',awayId,'scores'),
    'matchupAwayShots': safe_chain(game,'pregame','matchup',awayId,'shots'),
    'matchupAwayHits': safe_chain(game,'pregame','matchup',awayId,'hits'),
    'matchupAwayPIM': safe_chain(game,'pregame','matchup',awayId,'pim'),
    'matchupAwayPowerplays': safe_chain(game,'pregame','matchup',awayId,'powerplays'),
    'matchupAwayPowerplayGoals': safe_chain(game,'pregame','matchup',awayId,'powerplayGoals'),
    'matchupAwayFaceoffWinPercent': safe_chain(game,'pregame','matchup',awayId,'faceoffWinPercent'),
    'matchupAwayBlocks': safe_chain(game,'pregame','matchup',awayId,'blocks'),
    'matchupAwayWinLoss': safe_chain(game,'pregame','matchup',awayId,'winLoss'),
    'last5HomeScores': safe_chain(game,'pregame','last5','home','scores'),
    'last5HomeShots': safe_chain(game,'pregame','last5','home','shots'),
    'last5HomeHits': safe_chain(game,'pregame','last5','home','hits'),
    'last5HomePIM': safe_chain(game,'pregame','last5','home','pim'),
    'last5HomePowerplays': safe_chain(game,'pregame','last5','home','powerplays'),
    'last5HomePowerplayGoals': safe_chain(game,'pregame','last5','home','powerplayGoals'),
    'last5HomeFaceoffWinPercent': safe_chain(game,'pregame','last5','home','faceoffWinPercent'),
    'last5HomeBlocks': safe_chain(game,'pregame','last5','home','blocks'),
    'last5HomeWinLoss': safe_chain(game,'pregame','last5','home','winLoss'),
    'last5HomeStartUTCTimes': safe_chain(game,'pregame','last5','home','startUTCTimes'),
    'last5HomeEasternOffsets': safe_chain(game,'pregame','last5','home','easternOffsets'),
    'last5HomeVenueOffsets': safe_chain(game,'pregame','last5','home','venueOffsets'),
    'last5AwayScores': safe_chain(game,'pregame','last5','away','scores'),
    'last5AwayShots': safe_chain(game,'pregame','last5','away','shots'),
    'last5AwayHits': safe_chain(game,'pregame','last5','away','hits'),
    'last5AwayPIM': safe_chain(game,'pregame','last5','away','pim'),
    'last5AwayPowerplays': safe_chain(game,'pregame','last5','away','powerplays'),
    'last5AwayPowerplayGoals': safe_chain(game,'pregame','last5','away','powerplayGoals'),
    'last5AwayFaceoffWinPercent': safe_chain(game,'pregame','last5','away','faceoffWinPercent'),
    'last5AwayBlocks': safe_chain(game,'pregame','last5','away','blocks'),
    'last5AwayWinLoss': safe_chain(game,'pregame','last5','away','winLoss'),
    'last5AwayStartUTCTimes': safe_chain(game,'pregame','last5','away','startUTCTimes'),
    'last5AwayEasternOffsets': safe_chain(game,'pregame','last5','away','easternOffsets'),
    'last5AwayVenueOffsets': safe_chain(game,'pregame','last5','away','venueOffsets'),
    'last10HomeScores': safe_chain(game,'pregame','last10','home','scores'),
    'last10HomeShots': safe_chain(game,'pregame','last10','home','shots'),
    'last10HomeHits': safe_chain(game,'pregame','last10','home','hits'),
    'last10HomePIM': safe_chain(game,'pregame','last10','home','pim'),
    'last10HomePowerplays': safe_chain(game,'pregame','last10','home','powerplays'),
    'last10HomePowerplayGoals': safe_chain(game,'pregame','last10','home','powerplayGoals'),
    'last10HomeFaceoffWinPercent': safe_chain(game,'pregame','last10','home','faceoffWinPercent'),
    'last10HomeBlocks': safe_chain(game,'pregame','last10','home','blocks'),
    'last10HomeWinLoss': safe_chain(game,'pregame','last10','home','winLoss'),
    'last10HomeStartUTCTimes': safe_chain(game,'pregame','last10','home','startUTCTimes'),
    'last10HomeEasternOffsets': safe_chain(game,'pregame','last10','home','easternOffsets'),
    'last10HomeVenueOffsets': safe_chain(game,'pregame','last10','home','venueOffsets'),
    'last10AwayScores': safe_chain(game,'pregame','last10','away','scores'),
    'last10AwayShots': safe_chain(game,'pregame','last10','away','shots'),
    'last10AwayHits': safe_chain(game,'pregame','last10','away','hits'),
    'last10AwayPIM': safe_chain(game,'pregame','last10','away','pim'),
    'last10AwayPowerplays': safe_chain(game,'pregame','last10','away','powerplays'),
    'last10AwayPowerplayGoals': safe_chain(game,'pregame','last10','away','powerplayGoals'),
    'last10AwayFaceoffWinPercent': safe_chain(game,'pregame','last10','away','faceoffWinPercent'),
    'last10AwayBlocks': safe_chain(game,'pregame','last10','away','blocks'),
    'last10AwayWinLoss': safe_chain(game,'pregame','last10','away','winLoss'),
    'last10AwayStartUTCTimes': safe_chain(game,'pregame','last10','away','startUTCTimes'),
    'last10AwayEasternOffsets': safe_chain(game,'pregame','last10','away','easternOffsets'),
    'last10AwayVenueOffsets': safe_chain(game,'pregame','last10','away','venueOffsets'),
    'seasonHomeScores': safe_chain(game,'pregame','season','home','scores'),
    'seasonHomeShots': safe_chain(game,'pregame','season','home','shots'),
    'seasonHomeHits': safe_chain(game,'pregame','season','home','hits'),
    'seasonHomePIM': safe_chain(game,'pregame','season','home','pim'),
    'seasonHomePowerplays': safe_chain(game,'pregame','season','home','powerplays'),
    'seasonHomePowerplayGoals': safe_chain(game,'pregame','season','home','powerplayGoals'),
    'seasonHomeFaceoffWinPercent': safe_chain(game,'pregame','season','home','faceoffWinPercent'),
    'seasonHomeBlocks': safe_chain(game,'pregame','season','home','blocks'),
    'seasonHomeWinLoss': safe_chain(game,'pregame','season','home','winLoss'),
    'seasonHomeStartUTCTimes': safe_chain(game,'pregame','season','home','startUTCTimes'),
    'seasonHomeEasternOffsets': safe_chain(game,'pregame','season','home','easternOffsets'),
    'seasonHomeVenueOffsets': safe_chain(game,'pregame','season','home','venueOffsets'),
    'seasonAwayScores': safe_chain(game,'pregame','season','away','scores'),
    'seasonAwayShots': safe_chain(game,'pregame','season','away','shots'),
    'seasonAwayHits': safe_chain(game,'pregame','season','away','hits'),
    'seasonAwayPIM': safe_chain(game,'pregame','season','away','pim'),
    'seasonAwayPowerplays': safe_chain(game,'pregame','season','away','powerplays'),
    'seasonAwayPowerplayGoals': safe_chain(game,'pregame','season','away','powerplayGoals'),
    'seasonAwayFaceoffWinPercent': safe_chain(game,'pregame','season','away','faceoffWinPercent'),
    'seasonAwayBlocks': safe_chain(game,'pregame','season','away','blocks'),
    'seasonAwayWinLoss': safe_chain(game,'pregame','season','away','winLoss'),
    'seasonAwayStartUTCTimes': safe_chain(game,'pregame','season','away','startUTCTimes'),
    'seasonAwayEasternOffsets': safe_chain(game,'pregame','season','away','easternOffsets'),
    'seasonAwayVenueOffsets': safe_chain(game,'pregame','season','away','venueOffsets'),
    'homeRest': safe_chain(game,'pregame','team',homeId,'rest'),
    'awayRest': safe_chain(game,'pregame','team',awayId,'rest'),
    'homeLastRest': safe_chain(game,'pregame','team',homeId,'lastRest'),
    'awayLastRest': safe_chain(game,'pregame','team',awayId,'lastRest'),
    'finalPeriod': safe_chain(game,'period'),
    'pastRegulation': 1 if safe_chain(game,'period') > 3 else 0,
    'awayShots': safe_chain(game,'awayTeam','sog'),
    'homeShots': safe_chain(game,'homeTeam','sog'),
    'awayShotsPeriod1': safe_chain(game,'boxscore','shotsByPeriod',0,'away'),
    'homeShotsPeriod1': safe_chain(game,'boxscore','shotsByPeriod',0,'home'),
    'awayShotsPeriod2': safe_chain(game,'boxscore','shotsByPeriod',1,'away'),
    'homeShotsPeriod2': safe_chain(game,'boxscore','shotsByPeriod',1,'home'),
    'awayShotsPeriod3': safe_chain(game,'boxscore','shotsByPeriod',2,'away'),
    'homeShotsPeriod3': safe_chain(game,'boxscore','shotsByPeriod',2,'home'),
    'awayShotsPeriod4': safe_chain(game,'boxscore','shotsByPeriod',3,'away'),
    'homeShotsPeriod4': safe_chain(game,'boxscore','shotsByPeriod',3,'home'),
    'awayShotsPeriod5': safe_chain(game,'boxscore','shotsByPeriod',4,'away'),
    'homeShotsPeriod5': safe_chain(game,'boxscore','shotsByPeriod',4,'home'),
    'awayScorePeriod1': safe_chain(game,'boxscore','linescore','byPeriod',0,'away'),
    'homeScorePeriod1': safe_chain(game,'boxscore','linescore','byPeriod',0,'home'),
    'awayScorePeriod2': safe_chain(game,'boxscore','linescore','byPeriod',1,'away'),
    'homeScorePeriod2': safe_chain(game,'boxscore','linescore','byPeriod',1,'home'),
    'awayScorePeriod3': safe_chain(game,'boxscore','linescore','byPeriod',2,'away'),
    'homeScorePeriod3': safe_chain(game,'boxscore','linescore','byPeriod',2,'home'),
    'awayScorePeriod4': safe_chain(game,'boxscore','linescore','byPeriod',3,'away'),
    'homeScorePeriod4': safe_chain(game,'boxscore','linescore','byPeriod',3,'home'),
    'awayScorePeriod5': safe_chain(game,'boxscore','linescore','byPeriod',4,'away'),
    'homeScorePeriod5': safe_chain(game,'boxscore','linescore','byPeriod',4,'home'),
    'period1PuckLine': abs(safe_chain(game,'boxscore','linescore','byPeriod',0,'away') - safe_chain(game,'boxscore','linescore','byPeriod',0,'home')),
    'period2PuckLine': abs(safe_chain(game,'boxscore','linescore','byPeriod',1,'away') - safe_chain(game,'boxscore','linescore','byPeriod',1,'home')),
    'period3PuckLine': abs(safe_chain(game,'boxscore','linescore','byPeriod',2,'away') - safe_chain(game,'boxscore','linescore','byPeriod',2,'home')),
  }