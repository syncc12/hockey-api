import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from process_team import nhl_data_team
from util.team_models import PREDICT_SCORE_H2H, PREDICT_H2H, PREDICT_SPREAD, PREDICT_SCORE_SPREAD, PREDICT_COVERS, PREDICT_SCORE_COVERS, PREDICT_LGBM_H2H, PREDICT_LGBM_SCORE_H2H

def ai_teams(db, games, projectedLineups, models, useModel, projectedRosters, simple=False, receipt=False, vote='hard'):
  if useModel == 'xgb':
    wModels = models['wModels']
    lModels = models['lModels']
  elif useModel == 'lgbm':
    wModels = models['wModelsLGBM']
    lModels = None
  # cModels = models['cModels']
  # sModels = models['sModels']
  all_games = []
  data, game_data, extra_data = nhl_data_team(db=db, games=games, useProjectedLineups=projectedLineups, useProjectedRosters=projectedRosters, no_df=True)
  if vote == 'soft':
    if useModel == 'xgb':
      predictions,confidences = PREDICT_H2H(data, wModels, lModels, simple_return=True)
    elif useModel == 'lgbm':
      predictions,confidences = PREDICT_LGBM_H2H(data, wModels, simple_return=True)
    # spread_predictions,spread_confidences = PREDICT_SPREAD(data, sModels,simple_return=True)
    # covers_predictions,covers_confidences = PREDICT_COVERS(data, cModels,simple_return=True)
  else:
    if useModel == 'xgb':
      predictions,confidences = PREDICT_SCORE_H2H(data, wModels, lModels, simple_return=True)
    elif useModel == 'lgbm':
      predictions,confidences = PREDICT_LGBM_SCORE_H2H(data, wModels, simple_return=True)
    # spread_predictions,spread_confidences = PREDICT_SCORE_SPREAD(data, sModels,simple_return=True)
    # covers_predictions,covers_confidences = PREDICT_SCORE_COVERS(data, cModels,simple_return=True)
  if simple:
    for i, prediction in enumerate(predictions):
      awayTeam = game_data[i]["away_team"]["name"]
      homeTeam = game_data[i]["home_team"]["name"]
      winner = homeTeam if prediction == 0 else awayTeam
      all_games.append({
        'awayTeam': awayTeam,
        'homeTeam': homeTeam,
        'winningTeamB': f"{winner} - {(confidences[i]*100):.2f}%",
        # 'spread': f'{spread_predictions[i]} - {(spread_confidences[i]*100):.2f}%',
        # 'covers': f'{covers_predictions[i]} - {(covers_confidences[i]*100):.2f}%',
        # 'crosscheck': {
        #   'awayWin': f"{w_predictions_away[i]} - {(w_probabilities_away[i]*100):.2f}%",
        #   'awayLoss': f"{l_predictions_away[i]} - {(l_probabilities_away[i]*100):.2f}%",
        #   'homeWin': f"{w_predictions_home[i]} - {(w_probabilities_home[i]*100):.2f}%",
        #   'homeLoss': f"{l_predictions_home[i]} - {(l_probabilities_home[i]*100):.2f}%",
        # },
        # 'message': extra_data['message'],
        # 'id': game_data['game_id'],
        # 'live': simple_live_data,
      })
    return all_games
  elif receipt:
    for i in range(len(predictions)):
      # p_covers = f' | C - {round(covers_confidences[i]*100)}%' if covers_predictions[i] == 1 else ''
      # all_games.append(f'{"p-" if extra_data[i]["isProjectedLineup"] else ""}{game_data[i]["home_team"]["abbreviation"] if predictions[i] == 0 else game_data[i]["away_team"]["abbreviation"]} {round(confidences[i]*100)}%{p_covers} | {spread_predictions[i]} - {round(spread_confidences[i]*100)}%')
      all_games.append(f'{"p-" if extra_data[i]["isProjectedLineup"] else ""}{game_data[i]["home_team"]["abbreviation"] if predictions[i] == 0 else game_data[i]["away_team"]["abbreviation"]} {round(confidences[i]*100)}%')
    return all_games
  else:
    return {
      'predictions':predictions,
      'confidences':confidences,
    }
