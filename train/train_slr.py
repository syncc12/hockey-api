from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from joblib import load, dump

VERSION=6
TRAINING_DATA_VERSION = 5

def predict(inData):
  data = pd.DataFrame(inData)
  x = data[[
    'id','season','gameType','venue','neutralSite','homeTeam','homeTeamBack1GameId','homeTeamBack1GameDate','homeTeamBack1GameType','homeTeamBack1GameVenue',
    'homeTeamBack1GameStartTime','homeTeamBack1GameEasternOffset','homeTeamBack1GameVenueOffset','homeTeamBack1GameOutcome','homeTeamBack1GameHomeAway',
    'homeTeamBack1GameFinalPeriod','homeTeamBack1GameScore','homeTeamBack1GameShots','homeTeamBack1GameFaceoffWinPercentage','homeTeamBack1GamePowerPlays',
    'homeTeamBack1GamePowerPlayPercentage','homeTeamBack1GamePIM','homeTeamBack1GameHits','homeTeamBack1GameBlocks','homeTeamBack1GameOpponent',
    'homeTeamBack1GameOpponentScore','homeTeamBack1GameOpponentShots','homeTeamBack1GameOpponentFaceoffWinPercentage','homeTeamBack1GameOpponentPowerPlays',
    'homeTeamBack1GameOpponentPowerPlayPercentage','homeTeamBack1GameOpponentPIM','homeTeamBack1GameOpponentHits','homeTeamBack1GameOpponentBlocks',
    'homeTeamBack2GameId','homeTeamBack2GameDate','homeTeamBack2GameType','homeTeamBack2GameVenue','homeTeamBack2GameStartTime','homeTeamBack2GameEasternOffset',
    'homeTeamBack2GameVenueOffset','homeTeamBack2GameOutcome','homeTeamBack2GameHomeAway','homeTeamBack2GameFinalPeriod','homeTeamBack2GameScore',
    'homeTeamBack2GameShots','homeTeamBack2GameFaceoffWinPercentage','homeTeamBack2GamePowerPlays','homeTeamBack2GamePowerPlayPercentage','homeTeamBack2GamePIM',
    'homeTeamBack2GameHits','homeTeamBack2GameBlocks','homeTeamBack2GameOpponent','homeTeamBack2GameOpponentScore','homeTeamBack2GameOpponentShots',
    'homeTeamBack2GameOpponentFaceoffWinPercentage','homeTeamBack2GameOpponentPowerPlays','homeTeamBack2GameOpponentPowerPlayPercentage','homeTeamBack2GameOpponentPIM',
    'homeTeamBack2GameOpponentHits','homeTeamBack2GameOpponentBlocks','homeTeamBack3GameId','homeTeamBack3GameDate','homeTeamBack3GameType','homeTeamBack3GameVenue',
    'homeTeamBack3GameStartTime','homeTeamBack3GameEasternOffset','homeTeamBack3GameVenueOffset','homeTeamBack3GameOutcome','homeTeamBack3GameHomeAway',
    'homeTeamBack3GameFinalPeriod','homeTeamBack3GameScore','homeTeamBack3GameShots','homeTeamBack3GameFaceoffWinPercentage','homeTeamBack3GamePowerPlays',
    'homeTeamBack3GamePowerPlayPercentage','homeTeamBack3GamePIM','homeTeamBack3GameHits','homeTeamBack3GameBlocks','homeTeamBack3GameOpponent','homeTeamBack3GameOpponentScore',
    'homeTeamBack3GameOpponentShots','homeTeamBack3GameOpponentFaceoffWinPercentage','homeTeamBack3GameOpponentPowerPlays','homeTeamBack3GameOpponentPowerPlayPercentage',
    'homeTeamBack3GameOpponentPIM','homeTeamBack3GameOpponentHits','homeTeamBack3GameOpponentBlocks','homeTeamBack4GameId','homeTeamBack4GameDate','homeTeamBack4GameType',
    'homeTeamBack4GameVenue','homeTeamBack4GameStartTime','homeTeamBack4GameEasternOffset','homeTeamBack4GameVenueOffset','homeTeamBack4GameOutcome','homeTeamBack4GameHomeAway',
    'homeTeamBack4GameFinalPeriod','homeTeamBack4GameScore','homeTeamBack4GameShots','homeTeamBack4GameFaceoffWinPercentage','homeTeamBack4GamePowerPlays',
    'homeTeamBack4GamePowerPlayPercentage','homeTeamBack4GamePIM','homeTeamBack4GameHits','homeTeamBack4GameBlocks','homeTeamBack4GameOpponent','homeTeamBack4GameOpponentScore',
    'homeTeamBack4GameOpponentShots','homeTeamBack4GameOpponentFaceoffWinPercentage','homeTeamBack4GameOpponentPowerPlays','homeTeamBack4GameOpponentPowerPlayPercentage',
    'homeTeamBack4GameOpponentPIM','homeTeamBack4GameOpponentHits','homeTeamBack4GameOpponentBlocks','homeTeamBack5GameId','homeTeamBack5GameDate','homeTeamBack5GameType',
    'homeTeamBack5GameVenue','homeTeamBack5GameStartTime','homeTeamBack5GameEasternOffset','homeTeamBack5GameVenueOffset','homeTeamBack5GameOutcome','homeTeamBack5GameHomeAway',
    'homeTeamBack5GameFinalPeriod','homeTeamBack5GameScore','homeTeamBack5GameShots','homeTeamBack5GameFaceoffWinPercentage','homeTeamBack5GamePowerPlays',
    'homeTeamBack5GamePowerPlayPercentage','homeTeamBack5GamePIM','homeTeamBack5GameHits','homeTeamBack5GameBlocks','homeTeamBack5GameOpponent','homeTeamBack5GameOpponentScore',
    'homeTeamBack5GameOpponentShots','homeTeamBack5GameOpponentFaceoffWinPercentage','homeTeamBack5GameOpponentPowerPlays','homeTeamBack5GameOpponentPowerPlayPercentage',
    'homeTeamBack5GameOpponentPIM','homeTeamBack5GameOpponentHits','homeTeamBack5GameOpponentBlocks','awayTeam','awayTeamBack1GameId','awayTeamBack1GameDate','awayTeamBack1GameType',
    'awayTeamBack1GameVenue','awayTeamBack1GameStartTime','awayTeamBack1GameEasternOffset','awayTeamBack1GameVenueOffset','awayTeamBack1GameOutcome','awayTeamBack1GameHomeAway',
    'awayTeamBack1GameFinalPeriod','awayTeamBack1GameScore','awayTeamBack1GameShots','awayTeamBack1GameFaceoffWinPercentage','awayTeamBack1GamePowerPlays',
    'awayTeamBack1GamePowerPlayPercentage','awayTeamBack1GamePIM','awayTeamBack1GameHits','awayTeamBack1GameBlocks','awayTeamBack1GameOpponent','awayTeamBack1GameOpponentScore',
    'awayTeamBack1GameOpponentShots','awayTeamBack1GameOpponentFaceoffWinPercentage','awayTeamBack1GameOpponentPowerPlays','awayTeamBack1GameOpponentPowerPlayPercentage',
    'awayTeamBack1GameOpponentPIM','awayTeamBack1GameOpponentHits','awayTeamBack1GameOpponentBlocks','awayTeamBack2GameId','awayTeamBack2GameDate','awayTeamBack2GameType',
    'awayTeamBack2GameVenue','awayTeamBack2GameStartTime','awayTeamBack2GameEasternOffset','awayTeamBack2GameVenueOffset','awayTeamBack2GameOutcome','awayTeamBack2GameHomeAway',
    'awayTeamBack2GameFinalPeriod','awayTeamBack2GameScore','awayTeamBack2GameShots','awayTeamBack2GameFaceoffWinPercentage','awayTeamBack2GamePowerPlays',
    'awayTeamBack2GamePowerPlayPercentage','awayTeamBack2GamePIM','awayTeamBack2GameHits','awayTeamBack2GameBlocks','awayTeamBack2GameOpponent','awayTeamBack2GameOpponentScore',
    'awayTeamBack2GameOpponentShots','awayTeamBack2GameOpponentFaceoffWinPercentage','awayTeamBack2GameOpponentPowerPlays','awayTeamBack2GameOpponentPowerPlayPercentage',
    'awayTeamBack2GameOpponentPIM','awayTeamBack2GameOpponentHits','awayTeamBack2GameOpponentBlocks','awayTeamBack3GameId','awayTeamBack3GameDate','awayTeamBack3GameType',
    'awayTeamBack3GameVenue','awayTeamBack3GameStartTime','awayTeamBack3GameEasternOffset','awayTeamBack3GameVenueOffset','awayTeamBack3GameOutcome','awayTeamBack3GameHomeAway',
    'awayTeamBack3GameFinalPeriod','awayTeamBack3GameScore','awayTeamBack3GameShots','awayTeamBack3GameFaceoffWinPercentage','awayTeamBack3GamePowerPlays',
    'awayTeamBack3GamePowerPlayPercentage','awayTeamBack3GamePIM','awayTeamBack3GameHits','awayTeamBack3GameBlocks','awayTeamBack3GameOpponent','awayTeamBack3GameOpponentScore',
    'awayTeamBack3GameOpponentShots','awayTeamBack3GameOpponentFaceoffWinPercentage','awayTeamBack3GameOpponentPowerPlays','awayTeamBack3GameOpponentPowerPlayPercentage',
    'awayTeamBack3GameOpponentPIM','awayTeamBack3GameOpponentHits','awayTeamBack3GameOpponentBlocks','awayTeamBack4GameId','awayTeamBack4GameDate','awayTeamBack4GameType',
    'awayTeamBack4GameVenue','awayTeamBack4GameStartTime','awayTeamBack4GameEasternOffset','awayTeamBack4GameVenueOffset','awayTeamBack4GameOutcome','awayTeamBack4GameHomeAway',
    'awayTeamBack4GameFinalPeriod','awayTeamBack4GameScore','awayTeamBack4GameShots','awayTeamBack4GameFaceoffWinPercentage','awayTeamBack4GamePowerPlays',
    'awayTeamBack4GamePowerPlayPercentage','awayTeamBack4GamePIM','awayTeamBack4GameHits','awayTeamBack4GameBlocks','awayTeamBack4GameOpponent','awayTeamBack4GameOpponentScore',
    'awayTeamBack4GameOpponentShots','awayTeamBack4GameOpponentFaceoffWinPercentage','awayTeamBack4GameOpponentPowerPlays','awayTeamBack4GameOpponentPowerPlayPercentage',
    'awayTeamBack4GameOpponentPIM','awayTeamBack4GameOpponentHits','awayTeamBack4GameOpponentBlocks','awayTeamBack5GameId','awayTeamBack5GameDate','awayTeamBack5GameType',
    'awayTeamBack5GameVenue','awayTeamBack5GameStartTime','awayTeamBack5GameEasternOffset','awayTeamBack5GameVenueOffset','awayTeamBack5GameOutcome','awayTeamBack5GameHomeAway',
    'awayTeamBack5GameFinalPeriod','awayTeamBack5GameScore','awayTeamBack5GameShots','awayTeamBack5GameFaceoffWinPercentage','awayTeamBack5GamePowerPlays',
    'awayTeamBack5GamePowerPlayPercentage','awayTeamBack5GamePIM','awayTeamBack5GameHits','awayTeamBack5GameBlocks','awayTeamBack5GameOpponent','awayTeamBack5GameOpponentScore',
    'awayTeamBack5GameOpponentShots','awayTeamBack5GameOpponentFaceoffWinPercentage','awayTeamBack5GameOpponentPowerPlays','awayTeamBack5GameOpponentPowerPlayPercentage',
    'awayTeamBack5GameOpponentPIM','awayTeamBack5GameOpponentHits','awayTeamBack5GameOpponentBlocks','startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1',
    'linesman1','awayForward1','awayForward1Position','awayForward1Age','awayForward1Shoots','awayForward2','awayForward2Position','awayForward2Age','awayForward2Shoots',
    'awayForward3','awayForward3Position','awayForward3Age','awayForward3Shoots','awayForward4','awayForward4Position','awayForward4Age','awayForward4Shoots','awayForward5',
    'awayForward5Position','awayForward5Age','awayForward5Shoots','awayForward6','awayForward6Position','awayForward6Age','awayForward6Shoots','awayForward7','awayForward7Position',
    'awayForward7Age','awayForward7Shoots','awayForward8','awayForward8Position','awayForward8Age','awayForward8Shoots','awayForward9','awayForward9Position','awayForward9Age',
    'awayForward9Shoots','awayForward10','awayForward10Position','awayForward10Age','awayForward10Shoots','awayForward11','awayForward11Position','awayForward11Age','awayForward11Shoots',
    'awayForward12','awayForward12Position','awayForward12Age','awayForward12Shoots','awayForward13','awayForward13Position','awayForward13Age','awayForward13Shoots','awayDefenseman1',
    'awayDefenseman1Position','awayDefenseman1Age','awayDefenseman1Shoots','awayDefenseman2','awayDefenseman2Position','awayDefenseman2Age','awayDefenseman2Shoots','awayDefenseman3',
    'awayDefenseman3Position','awayDefenseman3Age','awayDefenseman3Shoots','awayDefenseman4','awayDefenseman4Position','awayDefenseman4Age','awayDefenseman4Shoots','awayDefenseman5',
    'awayDefenseman5Position','awayDefenseman5Age','awayDefenseman5Shoots','awayDefenseman6','awayDefenseman6Position','awayDefenseman6Age','awayDefenseman6Shoots','awayDefenseman7',
    'awayDefenseman7Position','awayDefenseman7Age','awayDefenseman7Shoots','awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieAge','awayStartingGoalieHeight',
    'awayStartingGoalieWeight','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieAge','awayBackupGoalieHeight','awayBackupGoalieWeight','homeForward1','homeForward1Position',
    'homeForward1Age','homeForward1Shoots','homeForward2','homeForward2Position','homeForward2Age','homeForward2Shoots','homeForward3','homeForward3Position','homeForward3Age',
    'homeForward3Shoots','homeForward4','homeForward4Position','homeForward4Age','homeForward4Shoots','homeForward5','homeForward5Position','homeForward5Age','homeForward5Shoots',
    'homeForward6','homeForward6Position','homeForward6Age','homeForward6Shoots','homeForward7','homeForward7Position','homeForward7Age','homeForward7Shoots','homeForward8',
    'homeForward8Position','homeForward8Age','homeForward8Shoots','homeForward9','homeForward9Position','homeForward9Age','homeForward9Shoots','homeForward10','homeForward10Position',
    'homeForward10Age','homeForward10Shoots','homeForward11','homeForward11Position','homeForward11Age','homeForward11Shoots','homeForward12','homeForward12Position','homeForward12Age',
    'homeForward12Shoots','homeForward13','homeForward13Position','homeForward13Age','homeForward13Shoots','homeDefenseman1','homeDefenseman1Position','homeDefenseman1Age',
    'homeDefenseman1Shoots','homeDefenseman2','homeDefenseman2Position','homeDefenseman2Age','homeDefenseman2Shoots','homeDefenseman3','homeDefenseman3Position','homeDefenseman3Age',
    'homeDefenseman3Shoots','homeDefenseman4','homeDefenseman4Position','homeDefenseman4Age','homeDefenseman4Shoots','homeDefenseman5','homeDefenseman5Position','homeDefenseman5Age',
    'homeDefenseman5Shoots','homeDefenseman6','homeDefenseman6Position','homeDefenseman6Age','homeDefenseman6Shoots','homeDefenseman7','homeDefenseman7Position','homeDefenseman7Age',
    'homeDefenseman7Shoots','homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieAge','homeStartingGoalieHeight','homeStartingGoalieWeight','homeBackupGoalie',
    'homeBackupGoalieCatches','homeBackupGoalieAge','homeBackupGoalieHeight','homeBackupGoalieWeight'
  ]].values

  y = data['winner']
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
  model = LogisticRegression()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  print("Accuracy:", accuracy_score(y_test, predictions))
  dump(model,f'models/nhl_ai_v{VERSION}_slr.joblib')


training_data = load(f'training_data/training_data_v{TRAINING_DATA_VERSION}.joblib')
predict(training_data)