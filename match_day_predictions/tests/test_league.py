import pytest

from match_day_predictions.team import Team
from match_day_predictions.league import League
from match_day_predictions.import_fixtures import ImportFixtures


class TestLeague:

    @pytest.fixture(autouse=True)
    def setup_league(self):
        import_fixtures = ImportFixtures(None, False)
        fixture_list = import_fixtures.get_csv_as_fixture_data_list('./data/fixtures_raw/2324/E0.csv')
        self.league = League(season=fixture_list[0].season, div=fixture_list[0].div, verbose_flag=False)
        for fixture_data in fixture_list:
            self.league.update(fixture_data)

    def test_fixture_load(self):
        import_fixtures = ImportFixtures(None, False)
        fixture_list = import_fixtures.get_csv_as_fixture_data_list('./data/fixtures_raw/2324/E0.csv')
        assert len(fixture_list) == 380

    def test_num_teams_in_league(self):
        assert self.league.get_num_teams() == 20

    def test_top_team(self):
        assert self.league.get_team_at_league_position(0).get_name() == 'Man City'

    def test_second_team(self):
        assert self.league.get_team_at_league_position(1).get_name() == 'Arsenal'

    def test_second_bottom_team(self):
        assert self.league.get_team_at_league_position(18).get_name() == 'Burnley'

    def test_top_team_points(self):
        assert self.league.get_team_at_league_position(0).get_points() == 91

    def test_top_team_goal_diff(self):
        assert self.league.get_team_at_league_position(0).get_goal_diff() == 62

    def test_top_team_l5_points(self):
        assert self.league.get_team_at_league_position(0).get_points_l5() == 3.0

    def test_bottom_team_l5_points(self):
        assert self.league.get_team_at_league_position(18).get_points_l5() == 0.8

    def test_second_bottom_team_l5_results(self):
        assert self.league.get_team_at_league_position(18).get_results_l5() == 'W,D,L,L,L'

    def test_bottom_team(self):
        assert self.league.get_team_at_league_position(19).get_name() == 'Sheffield United'

    def test_bottom_team_div(self):
        assert self.league.get_team_at_league_position(19).get_div() == 'E0'

    def test_bottom_team_season(self):
        assert self.league.get_team_at_league_position(19).get_season() == '2324'

    def test_bottom_team_l5_results(self):
        assert self.league.get_team_at_league_position(19).get_results_l5() == 'L,L,L,L,L'

    def test_bottom_team_l5_points(self):
        assert self.league.get_team_at_league_position(19).get_points_l5() == 0.0

    def test_bottom_team_l5_wins(self):
        assert self.league.get_team_at_league_position(19).get_wins_l5() == 0.0

    def test_bottom_team_l5_draws(self):
        assert self.league.get_team_at_league_position(19).get_draws_l5() == 0.0

    def test_bottom_team_l5_losses(self):
        assert self.league.get_team_at_league_position(19).get_losses_l5() == 1.0

    def test_league_positions_mid(self):
        team_obj = self.league.get_team_at_league_position(5)
        top_pos, _, _ = self.league.get_league_position(team_obj)
        assert team_obj.get_name() == "Tottenham"
        assert top_pos == (5.0 / 20.0)

    def test_verify_league_rankings(self):
        self.league.update_final_standings()
        expected = {
            0: "Man City", 1: "Arsenal", 2: "Liverpool", 3: "Aston Villa",
            4: "Tottenham", 5: "Chelsea", 6: "Newcastle", 7: "Man United",
            17: "Luton", 18: "Burnley", 19: "Sheffield United",
        }
        for position, name in expected.items():
            assert self.league.get_team_at_league_position(position).get_name() == name

    def test_league_positions_final_rankings(self):
        self.league.update_final_standings()
        expected_positions = {
            0: 0.0, 1: 1 / 20, 2: 2 / 20, 3: 3 / 20, 4: 4 / 20, 5: 5 / 20, 6: 6 / 20, 7: 7 / 20,
            17: 17 / 20, 18: 18 / 20, 19: 19 / 20,
        }
        for position, expected in expected_positions.items():
            team = self.league.get_team_at_league_position(position)
            league_pos, _, _ = self.league.get_league_position(team)
            assert league_pos == expected

        team_7 = self.league.get_team_at_league_position(7)
        _, _, away_pos_7 = self.league.get_league_position(team_7)
        assert away_pos_7 == (5.0 / 20.0)

        team_17 = self.league.get_team_at_league_position(17)
        _, _, away_pos_17 = self.league.get_league_position(team_17)
        assert away_pos_17 == (18.0 / 20.0)

    def test_get_features(self):
        home_team = self.league.get_team("Man City")
        away_team = self.league.get_team("Burnley")
        fixture = self.league.get_fixture("Man City", "Burnley")
        features = self.league.get_features(fixture, home_team, away_team)
        assert features.div == 'E0'
        assert features.season == '2324'
        assert features.home_team == 'Man City'
        assert features.away_team == 'Burnley'

    def test_pagerank_is_positive_for_top_team(self):
        self.league.update_final_standings()
        assert self.league.get_team_at_league_position(0).get_pagerank() > 0.0


class TestTeamLeaguePositionBugFix:
    """Regression tests for two bugs in the original V2 implementation:
    Team.update_league_position() wrote the away-table position into
    home_league_position (and never set away_league_position at all), and
    get_away_league_position() returned home_league_position instead of
    away_league_position -- together they silently made both accessors
    report the same (wrong) value.
    """

    def test_home_and_away_league_position_are_stored_independently(self):
        team = Team(div='E0', season='2324', team_name='Test FC')
        team.update_league_position(0.5, 0.25, 0.75)
        assert team.get_league_position() == 0.5
        assert team.get_home_league_position() == 0.25
        assert team.get_away_league_position() == 0.75
