from dataclasses import fields

from match_day_predictions.team import Team
from match_day_predictions.league import League
from match_day_predictions.feature_data import FeatureData
from match_day_predictions.import_fixtures import ImportFixtures

TEAM_TAIL_FIELDS = [
    f.name for f in fields(FeatureData)
    if f.name.startswith('home_team_') or f.name.startswith('away_team_')
]


class TestPredictionFeatures:

    def test_prediction_features_have_no_leaked_outcome(self):
        league = League('2324', 'E0', False)
        home_team = Team(div='E0', season='2324', team_name='Home FC')
        away_team = Team(div='E0', season='2324', team_name='Away FC')

        features = league.get_prediction_features(home_team, away_team)

        assert features.season == '2324'
        assert features.div == 'E0'
        assert features.home_team == 'Home FC'
        assert features.away_team == 'Away FC'
        assert features.home_ft is None
        assert features.away_ft is None
        assert features.is_home_win is None
        assert features.is_away_win is None
        assert features.is_draw is None

    def test_prediction_features_use_supplied_odds_and_default_the_rest(self):
        league = League('2324', 'E0', False)
        home_team = Team(div='E0', season='2324', team_name='Home FC')
        away_team = Team(div='E0', season='2324', team_name='Away FC')

        features = league.get_prediction_features(
            home_team, away_team,
            odds={'home_win_odds_avg': 1.8, 'draw_odds_avg': 3.4, 'away_win_odds_avg': 4.5},
        )

        assert features.home_win_odds_avg == 1.8
        assert features.draw_odds_avg == 3.4
        assert features.away_win_odds_avg == 4.5
        # Fields not supplied default to 0.0 rather than being left unset.
        assert features.home_win_odds_min == 0.0
        assert features.over_two_and_a_half_goals_avg == 0.0

    def test_prediction_features_match_training_features_pre_match(self):
        """The team-derived part of a prediction-time feature row must be
        identical to what get_features() would have computed right before
        that same fixture during training -- this is what makes train/predict
        consistent.
        """
        import_fixtures = ImportFixtures(None, False)
        fixture_list = import_fixtures.get_csv_as_fixture_data_list('./data/fixtures_raw/2324/E0.csv')
        fixture_list.sort(key=lambda fixture: fixture.date)

        league = League(fixture_list[0].season, fixture_list[0].div, False)

        target = None
        for fixture in fixture_list:
            home_team = league.get_team(fixture.home_team)
            away_team = league.get_team(fixture.away_team)
            if home_team and away_team and home_team.get_played() >= 5 and away_team.get_played() >= 5:
                target = fixture
                break
            league.update(fixture)

        assert target is not None, "expected to find a fixture with both teams already at 5+ games played"

        home_team = league.get_team(target.home_team)
        away_team = league.get_team(target.away_team)
        league.update_final_standings()

        predicted = league.get_prediction_features(home_team, away_team)
        trained = league.get_features(target, home_team, away_team)

        for name in TEAM_TAIL_FIELDS:
            assert getattr(predicted, name) == getattr(trained, name), name
