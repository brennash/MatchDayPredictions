import types

import networkx as nx

from .team import Team
from .feature_data import FeatureData

DEFAULT_MIN_GAMES_PLAYED = 5

# Order matters: this is the order odds fields are appended to feature rows,
# and it lines up 1:1 with the tail of FixtureData / FeatureData.
ODDS_FIELDS = [
    'home_win_odds_min', 'home_win_odds_max', 'home_win_odds_avg', 'home_win_odds_range', 'home_win_odds_std_dev',
    'draw_odds_min', 'draw_odds_max', 'draw_odds_avg', 'draw_odds_range', 'draw_odds_std_dev',
    'away_win_odds_min', 'away_win_odds_max', 'away_win_odds_avg', 'away_win_odds_range', 'away_win_odds_std_dev',
    'over_two_and_a_half_goals_min', 'over_two_and_a_half_goals_max', 'over_two_and_a_half_goals_avg',
    'over_two_and_a_half_goals_range', 'over_two_and_a_half_goals_std_dev',
    'under_two_and_a_half_goals_min', 'under_two_and_a_half_goals_max', 'under_two_and_a_half_goals_avg',
    'under_two_and_a_half_goals_range', 'under_two_and_a_half_goals_std_dev',
]


class League:

    def __init__(self, season, div, verbose_flag=False):
        self.season = season
        self.div = div
        self.verbose = verbose_flag

        self.league = []
        self.home_league = []
        self.away_league = []

        self.league_fixtures = []

        self.graph = nx.DiGraph()

    def get_div(self):
        return self.div

    def get_season(self):
        return self.season

    def get_num_teams(self):
        return len(self.league)

    def get_team_at_league_position(self, index):
        return self.league[index]

    def update(self, fixture_data):
        """Processes one chronologically-ordered fixture: snapshots pre-match
        features for both teams, then applies the result. Returns the
        pre-match FeatureData for this fixture.
        """
        self.league_fixtures.append(fixture_data)

        home_team = self.get_home_team(fixture_data)
        away_team = self.get_away_team(fixture_data)

        # Refresh standings/positions/pagerank to reflect every fixture played
        # so far (including other teams' matches processed earlier in this
        # chronological walk) BEFORE snapshotting features, so the features
        # reflect the true state of the table walking into this fixture.
        self._refresh_standings(home_team, away_team)

        features = self.get_features(fixture_data, home_team, away_team)

        home_team.update(fixture_data)
        away_team.update(fixture_data)

        self.update_league_teams_list(home_team)
        self.update_league_teams_list(away_team)
        self.update_graph(fixture_data)
        return features

    def _refresh_standings(self, home_team, away_team):
        self._resort_tables()

        home_pos, home_home_pos, home_away_pos = self.get_league_position(home_team)
        away_pos, away_home_pos, away_away_pos = self.get_league_position(away_team)
        home_team.update_league_position(home_pos, home_home_pos, home_away_pos)
        away_team.update_league_position(away_pos, away_home_pos, away_away_pos)

        if home_team.get_played() >= DEFAULT_MIN_GAMES_PLAYED and away_team.get_played() >= DEFAULT_MIN_GAMES_PLAYED:
            pagerank = self._compute_pagerank()
            home_team.update_page_rank(pagerank.get(home_team.get_name(), 0.0))
            away_team.update_page_rank(pagerank.get(away_team.get_name(), 0.0))

    def _resort_tables(self):
        self.league.sort(key=lambda team: (team.get_points(), team.get_goal_diff()), reverse=True)
        self.home_league = self.league.copy()
        self.away_league = self.league.copy()
        self.home_league.sort(key=lambda team: (team.get_home_points(), team.get_home_goal_diff()), reverse=True)
        self.away_league.sort(key=lambda team: (team.get_away_points(), team.get_away_goal_diff()), reverse=True)

    def update_final_standings(self):
        """Refreshes standings/league positions for every team using the full
        set of fixtures processed so far. Useful after replaying a season to
        get each team's true up-to-date table position (each update() call
        only refreshes standings as of just before that fixture), and as a
        helper for tests.
        """
        self._resort_tables()
        for team in self.league:
            league_pos, home_league_pos, away_league_pos = self.get_league_position(team)
            team.update_league_position(league_pos, home_league_pos, away_league_pos)

        if len(self.league) > 0:
            pagerank = self._compute_pagerank()
            for team in self.league:
                team.update_page_rank(pagerank.get(team.get_name(), 0.0))

    def get_league_position(self, team_obj):
        """Returns (league position, home-table position, away-table
        position) for a team, each normalised to [0, 1) by table size, or
        zero if the team isn't found in the corresponding table yet.
        """
        team_pos = 0.0
        home_team_pos = 0.0
        away_team_pos = 0.0

        total_teams = float(len(self.league))
        home_total_teams = float(len(self.home_league))
        away_total_teams = float(len(self.away_league))

        for index, team in enumerate(self.league):
            if team.get_name() == team_obj.get_name():
                team_pos = float(index) / total_teams

        for index, team in enumerate(self.home_league):
            if team.get_name() == team_obj.get_name():
                home_team_pos = float(index) / home_total_teams

        for index, team in enumerate(self.away_league):
            if team.get_name() == team_obj.get_name():
                away_team_pos = float(index) / away_total_teams

        return team_pos, home_team_pos, away_team_pos

    def get_home_team(self, fixture_data):
        return self._get_or_create_team(fixture_data.home_team)

    def get_away_team(self, fixture_data):
        return self._get_or_create_team(fixture_data.away_team)

    def _get_or_create_team(self, team_name):
        for team in self.league:
            if team.get_name() == team_name:
                return team
        new_team = Team(div=self.div, season=self.season, team_name=team_name)
        self.league.append(new_team)
        return new_team

    def update_league_teams_list(self, team):
        """Writes the team's updated state back into the league list."""
        team_name = team.get_name()
        for index, league_team in enumerate(self.league):
            if league_team.get_name() == team_name:
                self.league[index] = team
                return

    def get_fixture(self, home_team_name, away_team_name):
        for fixture_data in self.league_fixtures:
            if fixture_data.home_team == home_team_name and fixture_data.away_team == away_team_name:
                return fixture_data
        return None

    def get_team(self, team_name):
        for team in self.league:
            if team.get_name() == team_name:
                return team
        return None

    def update_graph(self, fixture_data):
        if fixture_data.home_team not in self.graph.nodes:
            self.graph.add_node(fixture_data.home_team)
        if fixture_data.away_team not in self.graph.nodes:
            self.graph.add_node(fixture_data.away_team)

        if fixture_data.home_ft > fixture_data.away_ft:
            self.graph.add_edge(fixture_data.away_team, fixture_data.home_team)
        elif fixture_data.away_ft > fixture_data.home_ft:
            self.graph.add_edge(fixture_data.home_team, fixture_data.away_team)

    def _compute_pagerank(self):
        """Computes pagerank for the whole graph once per fixture, rather
        than once per team as the original implementation did.
        """
        if len(self.graph.nodes) == 0:
            return {}
        return nx.pagerank(self.graph)

    def get_features(self, fixture_data, home_team, away_team):
        """Training-time feature row: includes the actual result and odds
        taken from the historical fixture record.
        """
        if fixture_data.result == 'H':
            is_home_win, is_away_win, is_draw = 1, 0, 0
        elif fixture_data.result == 'A':
            is_home_win, is_away_win, is_draw = 0, 1, 0
        else:
            is_home_win, is_away_win, is_draw = 0, 0, 1

        head = [
            fixture_data.season, fixture_data.div, fixture_data.home_team, fixture_data.away_team,
            fixture_data.date,
            fixture_data.home_ft, fixture_data.away_ft,
            is_home_win, is_away_win, is_draw,
        ]
        head.extend(self._odds_head(fixture_data))
        return FeatureData(*head, *self._team_derived_features(home_team, away_team))

    def get_prediction_features(self, home_team, away_team, odds=None, date=None):
        """Inference-time feature row for a fixture that hasn't been played
        (or a hypothetical pairing): same team-derived tail as get_features,
        but the outcome is unknown and odds are optional/supplied by the
        caller instead of read off a historical FixtureData.
        """
        odds_source = types.SimpleNamespace(**{field: 0.0 for field in ODDS_FIELDS})
        if odds:
            for key, value in odds.items():
                setattr(odds_source, key, value)

        head = [
            self.season, self.div, home_team.get_name(), away_team.get_name(),
            date or "",
            None, None,
            None, None, None,
        ]
        head.extend(self._odds_head(odds_source))
        return FeatureData(*head, *self._team_derived_features(home_team, away_team))

    def _odds_head(self, odds_source):
        return [getattr(odds_source, field) for field in ODDS_FIELDS]

    def _team_derived_features(self, home_team, away_team):
        """The long list of home/away cumulative, home-split, away-split and
        last-5 rolling stats shared by both get_features and
        get_prediction_features.
        """
        features = []
        for team in (home_team, away_team):
            features.append(team.get_pagerank())
            features.append(team.get_played())
            features.append(team.get_league_position())
            features.append(self.safe_divide(team.get_points(), team.get_played()))
            features.append(self.safe_divide(team.get_wins(), team.get_played()))
            features.append(self.safe_divide(team.get_draws(), team.get_played()))
            features.append(self.safe_divide(team.get_losses(), team.get_played()))
            features.append(self.safe_divide(team.get_goals_for(), team.get_played()))
            features.append(self.safe_divide(team.get_goals_against(), team.get_played()))
            features.append(self.safe_divide(team.get_shots(), team.get_played()))
            features.append(self.safe_divide(team.get_shots_on_target(), team.get_played()))
            features.append(self.safe_divide(team.get_corners(), team.get_played()))
            features.append(self.safe_divide(team.get_fouls_committed(), team.get_played()))
            features.append(self.safe_divide(team.get_free_kicks(), team.get_played()))
            features.append(self.safe_divide(team.get_offsides(), team.get_played()))
            features.append(self.safe_divide(team.get_yellow_cards(), team.get_played()))
            features.append(self.safe_divide(team.get_red_cards(), team.get_played()))

            features.append(team.get_home_played())
            features.append(team.get_home_league_position())
            features.append(self.safe_divide(team.get_home_points(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_wins(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_draws(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_losses(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_goals_for(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_goals_against(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_shots(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_shots_on_target(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_corners(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_fouls_committed(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_free_kicks(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_offsides(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_yellow_cards(), team.get_home_played()))
            features.append(self.safe_divide(team.get_home_red_cards(), team.get_home_played()))

            features.append(team.get_away_played())
            features.append(team.get_away_league_position())
            features.append(self.safe_divide(team.get_away_points(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_wins(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_draws(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_losses(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_goals_for(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_goals_against(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_shots(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_shots_on_target(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_corners(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_fouls_committed(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_free_kicks(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_offsides(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_yellow_cards(), team.get_away_played()))
            features.append(self.safe_divide(team.get_away_red_cards(), team.get_away_played()))

            features.append(team.get_league_position_l5())
            features.append(team.get_points_l5())
            features.append(team.get_wins_l5())
            features.append(team.get_draws_l5())
            features.append(team.get_losses_l5())
            features.append(team.get_goals_for_l5())
            features.append(team.get_goals_against_l5())
            features.append(team.get_shots_l5())
            features.append(team.get_shots_on_target_l5())
            features.append(team.get_corners_l5())
            features.append(team.get_fouls_committed_l5())
            features.append(team.get_free_kicks_l5())
            features.append(team.get_offsides_l5())
            features.append(team.get_yellow_cards_l5())
            features.append(team.get_red_cards_l5())
        return features

    def safe_divide(self, a, b):
        try:
            return a / b
        except ZeroDivisionError:
            return 0.0


def build_league_from_fixtures(fixture_list, verbose_flag=False, min_games_played=DEFAULT_MIN_GAMES_PLAYED):
    """Replays a season/division's fixtures (in chronological order) through
    a fresh League. Returns (league, features_list) where features_list only
    contains rows for fixtures where both teams had already played at least
    min_games_played games — early-season rows are too noisy to train or
    evaluate on.
    """
    ordered_fixtures = sorted(fixture_list, key=lambda fixture: fixture.date)
    season_id = ordered_fixtures[0].season
    div_id = ordered_fixtures[0].div
    league = League(season_id, div_id, verbose_flag)

    features_list = []
    for fixture in ordered_fixtures:
        features = league.update(fixture)
        if features.home_team_played >= min_games_played and features.away_team_played >= min_games_played:
            features_list.append(features)

    return league, features_list
