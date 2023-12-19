import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

import requests
import json
from pymongo import MongoClient
import math
from datetime import datetime
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from joblib import dump, load
import pandas as pd
from multiprocessing import Pool
from util.training_data import season_training_data, game_training_data

RE_PULL = False

VERSION = 5

def xPlayerData(homeAway,position,index,isGoalie=False,gamesBack=-1):
  if isGoalie:
    playerTitle = f'{homeAway}{index}{position}'
    playerKeys = [
      f'{playerTitle}',
      f'{playerTitle}Catches',
      f'{playerTitle}Age',
      f'{playerTitle}Height',
      f'{playerTitle}Weight',
    ]
  else:
    playerTitle = f'{homeAway}{position}{index}'
    playerKeys = [
      f'{playerTitle}',
      f'{playerTitle}Position',
      f'{playerTitle}Age',
      f'{playerTitle}Shoots',
    ]
  
  if gamesBack > -1:
    for i in range(0,gamesBack):
      playerKeys.append(f'{playerTitle}Back{i+1}GameId')
      playerKeys.append(f'{playerTitle}Back{i+1}GameDate')
      playerKeys.append(f'{playerTitle}Back{i+1}GameTeam')
      playerKeys.append(f'{playerTitle}Back{i+1}GameHomeAway')
      playerKeys.append(f'{playerTitle}Back{i+1}GamePlayer')
      playerKeys.append(f'{playerTitle}Back{i+1}GamePosition')
      playerKeys.append(f'{playerTitle}Back{i+1}GamePIM')
      playerKeys.append(f'{playerTitle}Back{i+1}GameTOI')
      if isGoalie:
        playerKeys.append(f'{playerTitle}Back{i+1}GameEvenStrengthShotsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePowerPlayShotsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShorthandedShotsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GameSaves')
        playerKeys.append(f'{playerTitle}Back{i+1}GameSavePercentage')
        playerKeys.append(f'{playerTitle}Back{i+1}GameEvenStrengthGoalsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePowerPlayGoalsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShorthandedGoalsAgainst')
        playerKeys.append(f'{playerTitle}Back{i+1}GameGoalsAgainst')
      else:
        playerKeys.append(f'{playerTitle}Back{i+1}GameGoals')
        playerKeys.append(f'{playerTitle}Back{i+1}GameAssists')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePoints')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePlusMinus')
        playerKeys.append(f'{playerTitle}Back{i+1}GameHits')
        playerKeys.append(f'{playerTitle}Back{i+1}GameBlockedShots')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePowerPlayGoals')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePowerPlayPoints')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShorthandedGoals')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShorthandedPoints')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShots')
        playerKeys.append(f'{playerTitle}Back{i+1}GameFaceoffs')
        playerKeys.append(f'{playerTitle}Back{i+1}GameFaceoffWinPercentage')
        playerKeys.append(f'{playerTitle}Back{i+1}GamePowerPlayTOI')
        playerKeys.append(f'{playerTitle}Back{i+1}GameShorthandedTOI')
  return playerKeys

def xTeamData(homeAway,gamesBack=-1):
  teamKeys = [f'{homeAway}Team']
  for i in range(0,gamesBack):
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameId')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameDate')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameType')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameVenue')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameStartTime')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameEasternOffset')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameVenueOffset')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOutcome')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameHomeAway')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameFinalPeriod')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameScore')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameShots')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameFaceoffWinPercentage')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GamePowerPlays')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GamePowerPlayPercentage')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GamePIM')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameHits')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameBlocks')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponent')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentScore')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentShots')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentFaceoffWinPercentage')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentPowerPlays')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentPowerPlayPercentage')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentPIM')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentHits')
    teamKeys.append(f'{homeAway}TeamBack{i+1}GameOpponentBlocks')
  return teamKeys

def train(inData):
  imputer = SimpleImputer(strategy='constant', fill_value=-1)
  data = pd.DataFrame(inData)
  # x = data [['id','season','gameType','venue','neutralSite','homeTeam','awayTeam','awaySplitSquad','homeSplitSquad','startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1','awayForward1','awayForward1Position','awayForward1Age','awayForward1Shoots','awayForward2','awayForward2Position','awayForward2Age','awayForward2Shoots','awayForward3','awayForward3Position','awayForward3Age','awayForward3Shoots','awayForward4','awayForward4Position','awayForward4Age','awayForward4Shoots','awayForward5','awayForward5Position','awayForward5Age','awayForward5Shoots','awayForward6','awayForward6Position','awayForward6Age','awayForward6Shoots','awayForward7','awayForward7Position','awayForward7Age','awayForward7Shoots','awayForward8','awayForward8Position','awayForward8Age','awayForward8Shoots','awayForward9','awayForward9Position','awayForward9Age','awayForward9Shoots','awayForward10','awayForward10Position','awayForward10Age','awayForward10Shoots','awayForward11','awayForward11Position','awayForward11Age','awayForward11Shoots','awayForward12','awayForward12Position','awayForward12Age','awayForward12Shoots','awayForward13','awayForward13Position','awayForward13Age','awayForward13Shoots','awayDefenseman1','awayDefenseman1Position','awayDefenseman1Age','awayDefenseman1Shoots','awayDefenseman2','awayDefenseman2Position','awayDefenseman2Age','awayDefenseman2Shoots','awayDefenseman3','awayDefenseman3Position','awayDefenseman3Age','awayDefenseman3Shoots','awayDefenseman4','awayDefenseman4Position','awayDefenseman4Age','awayDefenseman4Shoots','awayDefenseman5','awayDefenseman5Position','awayDefenseman5Age','awayDefenseman5Shoots','awayDefenseman6','awayDefenseman6Position','awayDefenseman6Age','awayDefenseman6Shoots','awayDefenseman7','awayDefenseman7Position','awayDefenseman7Age','awayDefenseman7Shoots','awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieAge','awayStartingGoalieHeight','awayStartingGoalieWeight','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieAge','awayBackupGoalieHeight','awayBackupGoalieWeight','awayThirdGoalie','awayThirdGoalieCatches','awayThirdGoalieAge','awayThirdGoalieHeight','awayThirdGoalieWeight','homeForward1','homeForward1Position','homeForward1Age','homeForward1Shoots','homeForward2','homeForward2Position','homeForward2Age','homeForward2Shoots','homeForward3','homeForward3Position','homeForward3Age','homeForward3Shoots','homeForward4','homeForward4Position','homeForward4Age','homeForward4Shoots','homeForward5','homeForward5Position','homeForward5Age','homeForward5Shoots','homeForward6','homeForward6Position','homeForward6Age','homeForward6Shoots','homeForward7','homeForward7Position','homeForward7Age','homeForward7Shoots','homeForward8','homeForward8Position','homeForward8Age','homeForward8Shoots','homeForward9','homeForward9Position','homeForward9Age','homeForward9Shoots','homeForward10','homeForward10Position','homeForward10Age','homeForward10Shoots','homeForward11','homeForward11Position','homeForward11Age','homeForward11Shoots','homeForward12','homeForward12Position','homeForward12Age','homeForward12Shoots','homeForward13','homeForward13Position','homeForward13Age','homeForward13Shoots','homeDefenseman1','homeDefenseman1Position','homeDefenseman1Age','homeDefenseman1Shoots','homeDefenseman2','homeDefenseman2Position','homeDefenseman2Age','homeDefenseman2Shoots','homeDefenseman3','homeDefenseman3Position','homeDefenseman3Age','homeDefenseman3Shoots','homeDefenseman4','homeDefenseman4Position','homeDefenseman4Age','homeDefenseman4Shoots','homeDefenseman5','homeDefenseman5Position','homeDefenseman5Age','homeDefenseman5Shoots','homeDefenseman6','homeDefenseman6Position','homeDefenseman6Age','homeDefenseman6Shoots','homeDefenseman7','homeDefenseman7Position','homeDefenseman7Age','homeDefenseman7Shoots','homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieAge','homeStartingGoalieHeight','homeStartingGoalieWeight','homeBackupGoalie','homeBackupGoalieCatches','homeBackupGoalieAge','homeBackupGoalieHeight','homeBackupGoalieWeight','homeThirdGoalie','homeThirdGoalieCatches','homeThirdGoalieAge','homeThirdGoalieHeight','homeThirdGoalieWeight']].values
  # x = data [['id','season','gameType','venue','neutralSite','homeTeam','awayTeam','startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1','awayForward1','awayForward1Position','awayForward1Age','awayForward1Shoots','awayForward2','awayForward2Position','awayForward2Age','awayForward2Shoots','awayForward3','awayForward3Position','awayForward3Age','awayForward3Shoots','awayForward4','awayForward4Position','awayForward4Age','awayForward4Shoots','awayForward5','awayForward5Position','awayForward5Age','awayForward5Shoots','awayForward6','awayForward6Position','awayForward6Age','awayForward6Shoots','awayForward7','awayForward7Position','awayForward7Age','awayForward7Shoots','awayForward8','awayForward8Position','awayForward8Age','awayForward8Shoots','awayForward9','awayForward9Position','awayForward9Age','awayForward9Shoots','awayForward10','awayForward10Position','awayForward10Age','awayForward10Shoots','awayForward11','awayForward11Position','awayForward11Age','awayForward11Shoots','awayForward12','awayForward12Position','awayForward12Age','awayForward12Shoots','awayForward13','awayForward13Position','awayForward13Age','awayForward13Shoots','awayDefenseman1','awayDefenseman1Position','awayDefenseman1Age','awayDefenseman1Shoots','awayDefenseman2','awayDefenseman2Position','awayDefenseman2Age','awayDefenseman2Shoots','awayDefenseman3','awayDefenseman3Position','awayDefenseman3Age','awayDefenseman3Shoots','awayDefenseman4','awayDefenseman4Position','awayDefenseman4Age','awayDefenseman4Shoots','awayDefenseman5','awayDefenseman5Position','awayDefenseman5Age','awayDefenseman5Shoots','awayDefenseman6','awayDefenseman6Position','awayDefenseman6Age','awayDefenseman6Shoots','awayDefenseman7','awayDefenseman7Position','awayDefenseman7Age','awayDefenseman7Shoots','awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieAge','awayStartingGoalieHeight','awayStartingGoalieWeight','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieAge','awayBackupGoalieHeight','awayBackupGoalieWeight','awayThirdGoalie','awayThirdGoalieCatches','awayThirdGoalieAge','awayThirdGoalieHeight','awayThirdGoalieWeight','homeForward1','homeForward1Position','homeForward1Age','homeForward1Shoots','homeForward2','homeForward2Position','homeForward2Age','homeForward2Shoots','homeForward3','homeForward3Position','homeForward3Age','homeForward3Shoots','homeForward4','homeForward4Position','homeForward4Age','homeForward4Shoots','homeForward5','homeForward5Position','homeForward5Age','homeForward5Shoots','homeForward6','homeForward6Position','homeForward6Age','homeForward6Shoots','homeForward7','homeForward7Position','homeForward7Age','homeForward7Shoots','homeForward8','homeForward8Position','homeForward8Age','homeForward8Shoots','homeForward9','homeForward9Position','homeForward9Age','homeForward9Shoots','homeForward10','homeForward10Position','homeForward10Age','homeForward10Shoots','homeForward11','homeForward11Position','homeForward11Age','homeForward11Shoots','homeForward12','homeForward12Position','homeForward12Age','homeForward12Shoots','homeForward13','homeForward13Position','homeForward13Age','homeForward13Shoots','homeDefenseman1','homeDefenseman1Position','homeDefenseman1Age','homeDefenseman1Shoots','homeDefenseman2','homeDefenseman2Position','homeDefenseman2Age','homeDefenseman2Shoots','homeDefenseman3','homeDefenseman3Position','homeDefenseman3Age','homeDefenseman3Shoots','homeDefenseman4','homeDefenseman4Position','homeDefenseman4Age','homeDefenseman4Shoots','homeDefenseman5','homeDefenseman5Position','homeDefenseman5Age','homeDefenseman5Shoots','homeDefenseman6','homeDefenseman6Position','homeDefenseman6Age','homeDefenseman6Shoots','homeDefenseman7','homeDefenseman7Position','homeDefenseman7Age','homeDefenseman7Shoots','homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieAge','homeStartingGoalieHeight','homeStartingGoalieWeight','homeBackupGoalie','homeBackupGoalieCatches','homeBackupGoalieAge','homeBackupGoalieHeight','homeBackupGoalieWeight','homeThirdGoalie','homeThirdGoalieCatches','homeThirdGoalieAge','homeThirdGoalieHeight','homeThirdGoalieWeight']].values
  # x = data [
  #   ['id','season','gameType','venue','neutralSite'] +
  #   xTeamData('home',5) +
  #   xTeamData('away',5) +
  #   ['startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1'] +
  #   xPlayerData('away','Forward',1) +
  #   xPlayerData('away','Forward',2) +
  #   xPlayerData('away','Forward',3) +
  #   xPlayerData('away','Forward',4) +
  #   xPlayerData('away','Forward',5) +
  #   xPlayerData('away','Forward',6) +
  #   xPlayerData('away','Forward',7) +
  #   xPlayerData('away','Forward',8) +
  #   xPlayerData('away','Forward',9) +
  #   xPlayerData('away','Forward',10) +
  #   xPlayerData('away','Forward',11) +
  #   xPlayerData('away','Forward',12) +
  #   xPlayerData('away','Forward',13) +
  #   xPlayerData('away','Defenseman',1) +
  #   xPlayerData('away','Defenseman',2) +
  #   xPlayerData('away','Defenseman',3) +
  #   xPlayerData('away','Defenseman',4) +
  #   xPlayerData('away','Defenseman',5) +
  #   xPlayerData('away','Defenseman',6) +
  #   xPlayerData('away','Defenseman',7) +
  #   xPlayerData('away','Goalie','Starting',True) +
  #   xPlayerData('away','Goalie','Backup',True) +
  #   xPlayerData('home','Forward',1) +
  #   xPlayerData('home','Forward',2) +
  #   xPlayerData('home','Forward',3) +
  #   xPlayerData('home','Forward',4) +
  #   xPlayerData('home','Forward',5) +
  #   xPlayerData('home','Forward',6) +
  #   xPlayerData('home','Forward',7) +
  #   xPlayerData('home','Forward',8) +
  #   xPlayerData('home','Forward',9) +
  #   xPlayerData('home','Forward',10) +
  #   xPlayerData('home','Forward',11) +
  #   xPlayerData('home','Forward',12) +
  #   xPlayerData('home','Forward',13) +
  #   xPlayerData('home','Defenseman',1) +
  #   xPlayerData('home','Defenseman',2) +
  #   xPlayerData('home','Defenseman',3) +
  #   xPlayerData('home','Defenseman',4) +
  #   xPlayerData('home','Defenseman',5) +
  #   xPlayerData('home','Defenseman',6) +
  #   xPlayerData('home','Defenseman',7) +
  #   xPlayerData('home','Goalie','Starting',True) +
  #   xPlayerData('home','Goalie','Backup',True)
  # ].values
  x = data [[
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
  y = data [['homeScore','awayScore','winner']].values

  imputer.fit(x)
  x = imputer.transform(x)

  clf = RandomForestClassifier(random_state=12)

  clf.fit(x,y)

  dump(clf, f'models/nhl_ai_v{VERSION}.joblib')

tdList = os.listdir(f'training_data/v{VERSION}')

USE_SEASONS = True
SKIP_SEASONS = [int(td.replace(f'training_data_v{VERSION}_','').replace('.joblib','')) for td in tdList] if len(tdList) > 0 and not f'training_data_v{VERSION}.joblib' in os.listdir('training_data') else []
START_SEASON = 20052006

if __name__ == '__main__':
  if RE_PULL:
    # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
    # client = MongoClient(db_url)
    client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
    db = client["hockey"]
    if USE_SEASONS:
      seasons = list(db["dev_seasons"].find(
        {'seasonId': {'$gte': START_SEASON}},
        {'_id':0,'seasonId': 1}
      ))
      seasons = [int(season['seasonId']) for season in seasons]
      print(seasons)
      if (len(SKIP_SEASONS) > 0):
        for season in SKIP_SEASONS:
          seasons.remove(season)
        print(seasons)
    else:
      startID = 1924030112
      endID = 1924030114
      games = db["dev_games"].find(
        {'id':{'$gte':startID,'$lt':endID+1}},
        # {'id':{'$lt':endID+1}},
        {'id': 1, '_id': 0}
      )

    pool = Pool(processes=4)
    if USE_SEASONS:
      result = pool.map(season_training_data,seasons)
    else:
      result = pool.map(game_training_data,games)
    if len(SKIP_SEASONS) > 0:
      for skip_season in SKIP_SEASONS:
        season_data = load(f'training_data/v{VERSION}/training_data_v{VERSION}_{skip_season}.joblib')
        result.append(season_data)
    result = np.concatenate(result).tolist()
    pool.close()
    dump(result,f'training_data/training_data_v{VERSION}.joblib')
    # f = open('training_data/training_data_text.txt','w')
    # f.write(json.dumps(result[60000:60500]))
  else:
    training_data_path = f'training_data/training_data_v{VERSION}.joblib'
    print(training_data_path)
    result = load(training_data_path)
    # f = open('training_data/training_data_text.txt', 'w')
    # f.write(json.dumps(result[60000:60500]))
  print('Games Collected')
  train(result)