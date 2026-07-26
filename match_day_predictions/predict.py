import csv
import json
import os
from dataclasses import asdict

import pandas as pd
import yaml

from .fixture_data import FixtureData
from .team import Team
from .league import build_league_from_fixtures, DEFAULT_MIN_GAMES_PLAYED
from .train_model import load_model_bundle

OUTCOME_CLASSES = ('H', 'D', 'A')


class Predict:
    """Scores a hypothetical/upcoming fixture using the trained model, by
    replaying a division's played-so-far fixtures to reconstruct current
    team/league state (via the same League used at training time), then
    building a prediction-time feature row for the requested pairing.
    """

    def __init__(self, config_filename, verbose_flag=False):
        self.verbose = verbose_flag
        with open(config_filename, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        model_path = self.config['trained_model']['model_path']
        bundle = load_model_bundle(model_path)
        self.model = bundle['model']
        self.feature_columns = bundle['feature_columns']
        self.classes = bundle['classes']

        self._league_cache = {}

    def predict_fixture(self, div, season, home_team_name, away_team_name, odds=None):
        league = self._get_league(season, div)
        home_team = self._resolve_team(league, season, div, home_team_name)
        away_team = self._resolve_team(league, season, div, away_team_name)

        feature_row = league.get_prediction_features(home_team, away_team, odds=odds)
        return self._score(div, season, home_team_name, away_team_name, feature_row, odds)

    def predict_fixtures_csv(self, csv_path):
        """Batch mode: CSV with columns div,season,home_team,away_team and
        optional home_odds,draw_odds,away_odds (single current-price quotes,
        not the min/max/avg/range/std_dev spread the model was trained on --
        see note in _parse_row_odds).
        """
        results = []
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                odds = self._parse_row_odds(row)
                results.append(self.predict_fixture(
                    div=row['div'],
                    season=row['season'],
                    home_team_name=row['home_team'],
                    away_team_name=row['away_team'],
                    odds=odds,
                ))
        return results

    def _parse_row_odds(self, row):
        def _float(key):
            value = row.get(key, '')
            return float(value) if value not in (None, '') else 0.0

        home_odds, draw_odds, away_odds = _float('home_odds'), _float('draw_odds'), _float('away_odds')
        if home_odds == 0.0 and draw_odds == 0.0 and away_odds == 0.0:
            return None
        return {
            'home_win_odds_avg': home_odds,
            'draw_odds_avg': draw_odds,
            'away_win_odds_avg': away_odds,
        }

    def _get_league(self, season, div):
        cache_key = (season, div)
        if cache_key in self._league_cache:
            return self._league_cache[cache_key]

        fixtures_folder = self.config['fixtures']['fixtures_folder']
        fixtures_path = os.path.join(fixtures_folder, f"{season}_{div}.json")
        if not os.path.exists(fixtures_path):
            raise FileNotFoundError(
                f"No fixtures found at {fixtures_path}. Run the 'import' and 'features' "
                f"steps for this season/division first."
            )

        with open(fixtures_path, 'r') as f:
            fixture_list = [FixtureData(**fixture_dict) for fixture_dict in json.load(f)]

        min_games_played = self.config.get('league', {}).get('min_games_played', DEFAULT_MIN_GAMES_PLAYED)
        league, _ = build_league_from_fixtures(fixture_list, self.verbose, min_games_played)
        league.update_final_standings()

        self._league_cache[cache_key] = league
        return league

    def _resolve_team(self, league, season, div, team_name):
        team = league.get_team(team_name)
        if team is None:
            if self.verbose:
                print(f"Warning: '{team_name}' has no fixtures in {div}/{season} yet; "
                      f"predicting from a cold start (no history).")
            team = Team(div=div, season=season, team_name=team_name)
        return team

    def _score(self, div, season, home_team_name, away_team_name, feature_row, odds):
        row = pd.DataFrame([asdict(feature_row)])[self.feature_columns]
        probabilities = dict(zip(self.classes, self.model.predict_proba(row)[0]))

        result = {
            'div': div,
            'season': season,
            'home_team': home_team_name,
            'away_team': away_team_name,
            'prob_home_win': probabilities.get('H', 0.0),
            'prob_draw': probabilities.get('D', 0.0),
            'prob_away_win': probabilities.get('A', 0.0),
        }

        odds_market = {
            'H': odds.get('home_win_odds_avg', 0.0) if odds else 0.0,
            'D': odds.get('draw_odds_avg', 0.0) if odds else 0.0,
            'A': odds.get('away_win_odds_avg', 0.0) if odds else 0.0,
        }
        if all(odds_market[cls] > 0 for cls in OUTCOME_CLASSES):
            implied = {cls: 1.0 / odds_market[cls] for cls in OUTCOME_CLASSES}
            overround = sum(implied.values())
            fair_prob = {cls: implied[cls] / overround for cls in OUTCOME_CLASSES}
            edges = {cls: probabilities.get(cls, 0.0) - fair_prob[cls] for cls in OUTCOME_CLASSES}
            best_class = max(edges, key=edges.get)

            result['fair_prob_home_win'] = fair_prob['H']
            result['fair_prob_draw'] = fair_prob['D']
            result['fair_prob_away_win'] = fair_prob['A']
            result['best_value_bet'] = best_class
            result['edge'] = edges[best_class]

        return result


def format_prediction(result):
    lines = [f"{result['home_team']} (H) vs {result['away_team']} (A) — {result['div']}/{result['season']}"]
    lines.append(
        f"  P(home win)={result['prob_home_win']:.3f}  "
        f"P(draw)={result['prob_draw']:.3f}  "
        f"P(away win)={result['prob_away_win']:.3f}"
    )
    if 'best_value_bet' in result:
        outcome_name = {'H': 'home win', 'D': 'draw', 'A': 'away win'}[result['best_value_bet']]
        lines.append(f"  Best value bet: {outcome_name} (edge {result['edge']:+.3f} vs. de-vigged market)")
    return "\n".join(lines)
