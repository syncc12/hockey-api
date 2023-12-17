import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

from joblib import load, dump
import os
from helpers import n2n


# def check_training_data(season):
#   path = f'training_data/training_data_v3_{season}.joblib'
#   # path = f'training_data/training_data_v3.joblib'
#   return load(path)

# SEASONS = [
#   '19171918','19181919','19191920','19201921','19211922','19221923','19231924','19241925','19251926','19261927','19271928','19281929','19291930','19301931',
#   '19311932','19321933','19331934','19341935','19351936','19361937','19371938','19381939','19391940','19401941','19411942','19421943','19431944','19441945',
#   '19451946','19461947','19471948','19481949','19491950','19501951','19511952','19521953','19531954','19541955','19551956','19561957','19571958','19581959',
#   '19591960','19601961','19611962','19621963','19631964','19641965','19651966','19661967','19671968','19681969','19691970','19701971','19711972','19721973',
#   '19731974','19741975','19751976','19761977','19771978','19781979','19791980','19801981','19811982','19821983','19831984','19841985','19851986','19861987',
#   '19871988','19881989','19891990','19901991','19911992','19921993','19931994','19941995','19951996','19961997','19971998','19981999','19992000','20002001',
#   '20012002','20022003','20032004','20052006','20062007','20072008','20082009','20092010','20102011','20112012','20122013','20132014','20142015','20152016',
#   '20162017','20172018','20182019','20192020','20202021','20212022','20222023','20232024',
# ]

# training_data = []

# for season in SEASONS:
#   print(f'fired {season}')
#   data = check_training_data(season=season)
#   data_len = len(data)
#   for i in range(0,data_len):
#     line = data[i]
#     training_data.append(line)
#     print(f'{season} {i+1}/{data_len}')
#   print(f'DONE {season}')

# dump(training_data,'training_data/training_data_v3.joblib')

# VERSION = 4

# tdList = os.listdir(f'training_data/v{VERSION}')
# seasons = [td.replace(f'training_data_v{VERSION}_','').replace('.joblib','') for td in tdList] if len(tdList) > 0 and not f'training_data_v{VERSION}.joblib' in os.listdir('training_data') else []
# print(seasons)


# SEASONS = [
#   '19171918','19181919','19191920','19201921','19211922','19221923','19231924','19241925','19251926','19261927','19271928','19281929','19291930','19301931',
#   '19311932','19321933','19331934','19341935','19351936','19361937','19371938','19381939','19391940','19401941','19411942','19421943','19431944','19441945',
#   '19451946','19461947','19471948','19481949','19491950','19501951','19511952','19521953','19531954','19541955','19551956','19561957','19571958','19581959',
#   '19591960','19601961','19611962','19621963','19631964','19641965','19651966','19661967','19671968','19681969','19691970','19701971','19711972','19721973',
#   '19731974','19741975','19751976','19761977','19771978','19781979','19791980','19801981','19811982','19821983','19831984','19841985','19851986','19861987',
#   '19871988','19881989','19891990','19901991','19911992','19921993','19931994','19941995','19951996','19961997','19971998','19981999','19992000','20002001',
#   '20012002','20022003','20032004','20052006','20062007','20072008','20082009','20092010','20102011','20112012','20122013','20132014','20142015','20152016',
#   '20162017','20172018','20182019','20192020','20202021','20212022','20222023','20232024',
# ]

# for season in SEASONS:
#   training_data = load(f'training_data/v4/training_data_v4_{season}.joblib')
#   for i in range(0,len(training_data)):
#     try:
#       for j in range(0,5):
#         training_data[i][f'awayTeamBack{j+1}GameId'] = training_data[i][f'awayTeamBack{j+1}GameId']
#         training_data[i][f'awayTeamBack{j+1}GameDate'] = training_data[i][f'awayTeamBack{j+1}GameDate']
#         training_data[i][f'awayTeamBack{j+1}GameType'] = training_data[i][f'awayTeamBack{j+1}GameType']
#         training_data[i][f'awayTeamBack{j+1}GameVenue'] = training_data[i][f'awayTeamBack{j+1}GameVenue']
#         training_data[i][f'awayTeamBack{j+1}GameStartTime'] = training_data[i][f'awayTeamBack{j+1}GameStartTime']
#         training_data[i][f'awayTeamBack{j+1}GameEasternOffset'] = training_data[i][f'awayTeamBack{j+1}GameEasternOffset']
#         training_data[i][f'awayTeamBack{j+1}GameVenueOffset'] = training_data[i][f'awayTeamBack{j+1}GameVenueOffset']
#         training_data[i][f'awayTeamBack{j+1}GameOutcome'] = training_data[i][f'awayTeamBack{j+1}GameOutcome']
#         training_data[i][f'awayTeamBack{j+1}GameHomeAway'] = n2n(training_data[i][f'awayTeamBack{j+1}GameHomeAway'])
#         training_data[i][f'awayTeamBack{j+1}GameFinalPeriod'] = training_data[i][f'awayTeamBack{j+1}GameFinalPeriod']
#         training_data[i][f'awayTeamBack{j+1}GameScore'] = training_data[i][f'awayTeamBack{j+1}GameScore']
#         training_data[i][f'awayTeamBack{j+1}GameShots'] = training_data[i][f'awayTeamBack{j+1}GameShots']
#         training_data[i][f'awayTeamBack{j+1}GameFaceoffWinPercentage'] = training_data[i][f'awayTeamBack{j+1}GameFaceoffWinPercentage']
#         training_data[i][f'awayTeamBack{j+1}GamePowerPlays'] = training_data[i][f'awayTeamBack{j+1}GamePowerPlays']
#         training_data[i][f'awayTeamBack{j+1}GamePowerPlayPercentage'] = training_data[i][f'awayTeamBack{j+1}GamePowerPlayPercentage']
#         training_data[i][f'awayTeamBack{j+1}GamePIM'] = training_data[i][f'awayTeamBack{j+1}GamePIM']
#         training_data[i][f'awayTeamBack{j+1}GameHits'] = training_data[i][f'awayTeamBack{j+1}GameHits']
#         training_data[i][f'awayTeamBack{j+1}GameBlocks'] = training_data[i][f'awayTeamBack{j+1}GameBlocks']
#         training_data[i][f'awayTeamBack{j+1}GameOpponent'] = training_data[i][f'awayTeamBack{j+1}GameOpponent']
#         training_data[i][f'awayTeamBack{j+1}GameOpponentScore'] = training_data[i][f'awayTeamBack{j+1}GameOpponentScore']
#         training_data[i][f'awayTeamBack{j+1}GameOpponentShots'] = training_data[i][f'awayTeamBack{j+1}GameOpponentShots']
#         training_data[i][f'awayTeamBack{j+1}GameOpponentFaceoffWinPercentage'] = training_data[i][f'awayTeamBack{j+1}GameOpponentFaceoffWinPercentage']
#         training_data[i][f'awayTeamBack{j+1}GameOpponentPowerPlays'] = training_data[i][f'awayTeamBack{j+1}GameOpponentPowerPlays']
#         training_data[i][f'awayTeamBack{j+1}GameOpponentPowerPlayPercentage'] = training_data[i][f'awayTeamBack{j+1}GameOpponentPowerPlayPercentage']
#         training_data[i][f'awayTeamBack{j+1}GameOpponentPIM'] = training_data[i][f'awayTeamBack{j+1}GameOpponentPIM']
#         training_data[i][f'awayTeamBack{j+1}GameOpponentHits'] = training_data[i][f'awayTeamBack{j+1}GameOpponentHits']
#         training_data[i][f'awayTeamBack{j+1}GameOpponentBlocks'] = training_data[i][f'awayTeamBack{j+1}GameOpponentBlocks']

#         training_data[i][f'homeTeamBack{j+1}GameId'] = training_data[i][f'homeTeamBack{j+1}GameId']
#         training_data[i][f'homeTeamBack{j+1}GameDate'] = training_data[i][f'homeTeamBack{j+1}GameDate']
#         training_data[i][f'homeTeamBack{j+1}GameType'] = training_data[i][f'homeTeamBack{j+1}GameType']
#         training_data[i][f'homeTeamBack{j+1}GameVenue'] = training_data[i][f'homeTeamBack{j+1}GameVenue']
#         training_data[i][f'homeTeamBack{j+1}GameStartTime'] = training_data[i][f'homeTeamBack{j+1}GameStartTime']
#         training_data[i][f'homeTeamBack{j+1}GameEasternOffset'] = training_data[i][f'homeTeamBack{j+1}GameEasternOffset']
#         training_data[i][f'homeTeamBack{j+1}GameVenueOffset'] = training_data[i][f'homeTeamBack{j+1}GameVenueOffset']
#         training_data[i][f'homeTeamBack{j+1}GameOutcome'] = training_data[i][f'homeTeamBack{j+1}GameOutcome']
#         training_data[i][f'homeTeamBack{j+1}GameHomeAway'] = n2n(training_data[i][f'homeTeamBack{j+1}GameHomeAway'])
#         training_data[i][f'homeTeamBack{j+1}GameFinalPeriod'] = training_data[i][f'homeTeamBack{j+1}GameFinalPeriod']
#         training_data[i][f'homeTeamBack{j+1}GameScore'] = training_data[i][f'homeTeamBack{j+1}GameScore']
#         training_data[i][f'homeTeamBack{j+1}GameShots'] = training_data[i][f'homeTeamBack{j+1}GameShots']
#         training_data[i][f'homeTeamBack{j+1}GameFaceoffWinPercentage'] = training_data[i][f'homeTeamBack{j+1}GameFaceoffWinPercentage']
#         training_data[i][f'homeTeamBack{j+1}GamePowerPlays'] = training_data[i][f'homeTeamBack{j+1}GamePowerPlays']
#         training_data[i][f'homeTeamBack{j+1}GamePowerPlayPercentage'] = training_data[i][f'homeTeamBack{j+1}GamePowerPlayPercentage']
#         training_data[i][f'homeTeamBack{j+1}GamePIM'] = training_data[i][f'homeTeamBack{j+1}GamePIM']
#         training_data[i][f'homeTeamBack{j+1}GameHits'] = training_data[i][f'homeTeamBack{j+1}GameHits']
#         training_data[i][f'homeTeamBack{j+1}GameBlocks'] = training_data[i][f'homeTeamBack{j+1}GameBlocks']
#         training_data[i][f'homeTeamBack{j+1}GameOpponent'] = training_data[i][f'homeTeamBack{j+1}GameOpponent']
#         training_data[i][f'homeTeamBack{j+1}GameOpponentScore'] = training_data[i][f'homeTeamBack{j+1}GameOpponentScore']
#         training_data[i][f'homeTeamBack{j+1}GameOpponentShots'] = training_data[i][f'homeTeamBack{j+1}GameOpponentShots']
#         training_data[i][f'homeTeamBack{j+1}GameOpponentFaceoffWinPercentage'] = training_data[i][f'homeTeamBack{j+1}GameOpponentFaceoffWinPercentage']
#         training_data[i][f'homeTeamBack{j+1}GameOpponentPowerPlays'] = training_data[i][f'homeTeamBack{j+1}GameOpponentPowerPlays']
#         training_data[i][f'homeTeamBack{j+1}GameOpponentPowerPlayPercentage'] = training_data[i][f'homeTeamBack{j+1}GameOpponentPowerPlayPercentage']
#         training_data[i][f'homeTeamBack{j+1}GameOpponentPIM'] = training_data[i][f'homeTeamBack{j+1}GameOpponentPIM']
#         training_data[i][f'homeTeamBack{j+1}GameOpponentHits'] = training_data[i][f'homeTeamBack{j+1}GameOpponentHits']
#         training_data[i][f'homeTeamBack{j+1}GameOpponentBlocks'] = training_data[i][f'homeTeamBack{j+1}GameOpponentBlocks']


#     except KeyError as error:
#       training_data[i][error] = -1
    
#     print(f'{season} {i+1}/{len(training_data)}')
    

#   dump(training_data,f'training_data/v4/training_data_v4_{season}.joblib')

training_data = load(f'training_data/training_data_v4.joblib')
for i in range(0,len(training_data)):
  try:
    for j in range(0,5):
      if type(training_data[i][f'awayTeamBack{j+1}GameHomeAway']) == str or type(training_data[i][f'homeTeamBack{j+1}GameHomeAway']) == str:
        print(i)


  except KeyError as error:
    print(i,'error',training_data[i])
    
