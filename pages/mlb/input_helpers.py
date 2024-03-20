


def safe_chain(obj, *keys, default=-1):
  for key in keys:
    if key == default:
      return default
    else:
      if type(key) == int:
        if len(obj) > key:
          obj = obj[key]
        else:
          return default
      else:
        try:
          obj = getattr(obj, key, default) if hasattr(obj, key) else obj[key]
        except (KeyError, TypeError, AttributeError):
          return default
  return obj

def average_list(lst, escape_hatch=False):
  if len(lst) == 0:
    return 0
  if escape_hatch:
    return lst
  else:
    return sum(lst) / len(lst) if len(lst) > 0 else 0

def player_stats(players,playerIDs,prefix='',average=True):
  return {
    f'{prefix}BattingGamesPlayed': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','gamesPlayed') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingFlyOuts': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','flyOuts') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingGroundOuts': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','groundOuts') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingRuns': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','runs') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingDoubles': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','doubles') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingTriples': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','triples') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingHomeRuns': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','homeRuns') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingStrikeOuts': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','strikeOuts') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingBaseOnBalls': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','baseOnBalls') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingIntentionalWalks': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','intentionalWalks') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingHits': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','hits') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingHitByPitch': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','hitByPitch') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingAtBats': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','atBats') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingCaughtStealing': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','caughtStealing') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingStolenBases': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','stolenBases') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingGroundIntoDoublePlay': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','groundIntoDoublePlay') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingGroundIntoTriplePlay': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','groundIntoTriplePlay') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingPlateAppearances': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','plateAppearances') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingTotalBases': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','totalBases') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingRbi': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','rbi') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingLeftOnBase': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','leftOnBase') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingSacrificeBunts': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','sacBunts') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingSacrificeFlies': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','sacFlies') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingCatchersInterference': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','catchersInterference') for i in playerIDs],escape_hatch=average),
    f'{prefix}BattingPickoffs': average_list([safe_chain(players,f'ID{i}','seasonStats','batting','pickoffs') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingGamesPlayed': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','gamesPlayed') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingGamesStarted': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','gamesStarted') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingGroundOuts': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','groundOuts') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingAirOuts': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','airOuts') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingRuns': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','runs') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingDoubles': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','doubles') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingTriples': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','triples') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingHomeRuns': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','homeRuns') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingStrikeOuts': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','strikeOuts') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingBaseOnBalls': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','baseOnBalls') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingIntentionalWalks': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','intentionalWalks') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingHits': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','hits') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingHitByPitch': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','hitByPitch') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingAtBats': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','atBats') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingCaughtStealing': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','caughtStealing') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingStolenBases': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','stolenBases') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingNumberOfPitches': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','numberOfPitches') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingInningsPitched': average_list([float(safe_chain(players,f'ID{i}','seasonStats','pitching','inningsPitched')) for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingWins': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','wins') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingLosses': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','losses') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingSaves': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','saves') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingSaveOpportunities': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','saveOpportunities') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingHolds': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','holds') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingBlownSaves': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','blownSaves') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingEarnedRuns': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','earnedRuns') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingBattersFaced': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','battersFaced') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingOuts': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','outs') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingGamesPitched': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','gamesPlayed') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingCompleteGames': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','completeGames') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingShutouts': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','shutouts') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingPitchesThrown': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','pitchesThrown') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingBalls': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','balls') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingStrikes': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','strikes') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingHitBatsmen': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','hitBatsmen') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingBalks': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','balks') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingWildPitches': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','wildPitches') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingPickoffs': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','pickoffs') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingRBI': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','rbi') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingGamesFinished': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','gamesFinished') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingInheritedRunners': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','inheritedRunners') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingInheritedRunnersScored': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','inheritedRunnersScored') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingCathersInterference': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','catchersInterference') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingSacrificeBunts': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','sacBunts') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingSacrificeFlies': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','sacFlies') for i in playerIDs],escape_hatch=average),
    f'{prefix}PitchingPassedBall': average_list([safe_chain(players,f'ID{i}','seasonStats','pitching','passedBall') for i in playerIDs],escape_hatch=average),
    f'{prefix}FieldingCaughtStealing': average_list([safe_chain(players,f'ID{i}','seasonStats','fielding','caughtStealing') for i in playerIDs],escape_hatch=average),
    f'{prefix}FieldingStolenBases': average_list([safe_chain(players,f'ID{i}','seasonStats','fielding','stolenBases') for i in playerIDs],escape_hatch=average),
    f'{prefix}FieldingAssists': average_list([safe_chain(players,f'ID{i}','seasonStats','fielding','assists') for i in playerIDs],escape_hatch=average),
    f'{prefix}FieldingPutOuts': average_list([safe_chain(players,f'ID{i}','seasonStats','fielding','putOuts') for i in playerIDs],escape_hatch=average),
    f'{prefix}FieldingErrors': average_list([safe_chain(players,f'ID{i}','seasonStats','fielding','errors') for i in playerIDs],escape_hatch=average),
    f'{prefix}FieldingChances': average_list([safe_chain(players,f'ID{i}','seasonStats','fielding','chances') for i in playerIDs],escape_hatch=average),
    f'{prefix}FieldingPassedBall': average_list([safe_chain(players,f'ID{i}','seasonStats','fielding','passedBall') for i in playerIDs],escape_hatch=average),
    f'{prefix}FieldingPickoffs': average_list([safe_chain(players,f'ID{i}','seasonStats','fielding','pickoffs') for i in playerIDs],escape_hatch=average),
  }