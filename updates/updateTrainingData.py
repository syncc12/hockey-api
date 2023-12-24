from joblib import dump, load

SEASONS = [20052006,20062007,20072008,20082009,20092010,20102011,20112012,20122013,20132014,20142015,20152016,20162017,20172018,20182019,20192020,20202021,20212022,20222023,20232024]
# SEASONS = [20232024]

def update():
  for season in SEASONS:
    print(f'START {season}')
    move(season,5,6)
    # training_data = load(f'training_data/v5/training_data_v5_{season}.joblib')
    # training_data = addTotalGoals(training_data)
    # training_data = addGoalDifference(training_data)
    # dump(training_data, f'training_data/v5/training_data_v5_{season}.joblib')
    print(f'DONE  {season}')


def addTotalGoals(training_data):
  for data in range(0,len(training_data)):
    training_data[data]['totalGoals'] = training_data[data]['homeScore'] + training_data[data]['awayScore']
  return training_data

def addGoalDifference(training_data):
  for data in range(0,len(training_data)):
    homeScore = training_data[data]['homeScore']
    awayScore = training_data[data]['awayScore']
    if homeScore >= awayScore:
      score1 = homeScore
      score2 = awayScore
    else:
      score1 = awayScore
      score2 = homeScore

    training_data[data]['goalDifferential'] = score1 - score2
  return training_data

def move(season,oldVersion,newVersion):
  training_data = load(f'training_data/v{oldVersion}/training_data_v{oldVersion}_{season}.joblib')
  dump(training_data, f'training_data/v{newVersion}/training_data_v{newVersion}_{season}.joblib')

def printData(inSeason):
  data = load(f'training_data/v5/training_data_v5_{inSeason}.joblib')
  print(data)

update()
# printData(20232024)