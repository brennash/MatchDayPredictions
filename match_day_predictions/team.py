class Team:
    """Tracks a team's cumulative, home/away-split, and last-5-match rolling
    statistics across a single season/division. Updated fixture-by-fixture in
    chronological order by League.
    """

    def __init__(self, div, season, team_name):
        self.div = div
        self.season = season
        self.team_name = team_name
        self.league_position = 0.0
        self.page_rank = 0.0

        self.home_played = 0.0
        self.home_league_position = 0.0
        self.home_wins = 0.0
        self.home_draws = 0.0
        self.home_losses = 0.0
        self.home_points = 0.0
        self.home_goals_for = 0.0
        self.home_goals_against = 0.0
        self.home_shots = 0.0
        self.home_shots_on_target = 0.0
        self.home_corners = 0.0
        self.home_fouls_committed = 0.0
        self.home_free_kicks = 0.0
        self.home_offsides = 0.0
        self.home_yellow_cards = 0.0
        self.home_red_cards = 0.0

        self.away_played = 0.0
        self.away_league_position = 0.0
        self.away_wins = 0.0
        self.away_draws = 0.0
        self.away_losses = 0.0
        self.away_points = 0.0
        self.away_goals_for = 0.0
        self.away_goals_against = 0.0
        self.away_shots = 0.0
        self.away_shots_on_target = 0.0
        self.away_corners = 0.0
        self.away_fouls_committed = 0.0
        self.away_free_kicks = 0.0
        self.away_offsides = 0.0
        self.away_yellow_cards = 0.0
        self.away_red_cards = 0.0

        self.l5_results = [''] * 5
        self.l5_league_position = [0.0] * 5
        self.l5_wins = [0.0] * 5
        self.l5_draws = [0.0] * 5
        self.l5_losses = [0.0] * 5
        self.l5_points = [0.0] * 5
        self.l5_goals_for = [0.0] * 5
        self.l5_goals_against = [0.0] * 5
        self.l5_shots = [0.0] * 5
        self.l5_shots_on_target = [0.0] * 5
        self.l5_corners = [0.0] * 5
        self.l5_fouls_committed = [0.0] * 5
        self.l5_free_kicks = [0.0] * 5
        self.l5_offsides = [0.0] * 5
        self.l5_yellow_cards = [0.0] * 5
        self.l5_red_cards = [0.0] * 5

    def get_name(self):
        return self.team_name

    def update_league_position(self, league_position, home_league_position, away_league_position):
        self.league_position = league_position
        self.home_league_position = home_league_position
        self.away_league_position = away_league_position
        self.l5_league_position.pop(0)
        self.l5_league_position.append(league_position)

    def update_page_rank(self, page_rank):
        self.page_rank = page_rank

    def update(self, fixture_data):
        if fixture_data.home_team == self.team_name and fixture_data.div == self.div and fixture_data.season == self.season:
            self.home_update(fixture_data)
        elif fixture_data.away_team == self.team_name and fixture_data.div == self.div and fixture_data.season == self.season:
            self.away_update(fixture_data)

    def home_update(self, fixture_data):
        self.home_played += 1.0

        if fixture_data.home_ft > fixture_data.away_ft:
            self.home_wins += 1.0
            self.home_points += 3.0
            self._push_l5_result('W', wins=1.0, draws=0.0, losses=0.0, points=3.0)
        elif fixture_data.home_ft < fixture_data.away_ft:
            self.home_losses += 1.0
            self._push_l5_result('L', wins=0.0, draws=0.0, losses=1.0, points=0.0)
        else:
            self.home_draws += 1.0
            self.home_points += 1.0
            self._push_l5_result('D', wins=0.0, draws=1.0, losses=0.0, points=1.0)

        self.home_goals_for += fixture_data.home_ft
        self.home_goals_against += fixture_data.away_ft
        self.home_shots += fixture_data.home_shots
        self.home_shots_on_target += fixture_data.home_shots_on_target
        self.home_corners += fixture_data.home_corners
        self.home_fouls_committed += fixture_data.home_fouls_committed
        self.home_free_kicks += fixture_data.home_free_kicks
        self.home_offsides += fixture_data.home_offsides
        self.home_yellow_cards += fixture_data.home_yellow_cards
        self.home_red_cards += fixture_data.home_red_cards

        self._push_l5_stats(
            goals_for=fixture_data.home_ft,
            goals_against=fixture_data.away_ft,
            shots=fixture_data.home_shots,
            shots_on_target=fixture_data.home_shots_on_target,
            corners=fixture_data.home_corners,
            fouls_committed=fixture_data.home_fouls_committed,
            free_kicks=fixture_data.home_free_kicks,
            offsides=fixture_data.home_offsides,
            yellow_cards=fixture_data.home_yellow_cards,
            red_cards=fixture_data.home_red_cards,
        )

    def away_update(self, fixture_data):
        self.away_played += 1.0

        if fixture_data.home_ft < fixture_data.away_ft:
            self.away_wins += 1.0
            self.away_points += 3.0
            self._push_l5_result('W', wins=1.0, draws=0.0, losses=0.0, points=3.0)
        elif fixture_data.home_ft > fixture_data.away_ft:
            self.away_losses += 1.0
            self._push_l5_result('L', wins=0.0, draws=0.0, losses=1.0, points=0.0)
        else:
            self.away_draws += 1.0
            self.away_points += 1.0
            self._push_l5_result('D', wins=0.0, draws=1.0, losses=0.0, points=1.0)

        self.away_goals_for += fixture_data.away_ft
        self.away_goals_against += fixture_data.home_ft
        self.away_shots += fixture_data.away_shots
        self.away_shots_on_target += fixture_data.away_shots_on_target
        self.away_corners += fixture_data.away_corners
        self.away_fouls_committed += fixture_data.away_fouls_committed
        self.away_free_kicks += fixture_data.away_free_kicks
        self.away_offsides += fixture_data.away_offsides
        self.away_yellow_cards += fixture_data.away_yellow_cards
        self.away_red_cards += fixture_data.away_red_cards

        self._push_l5_stats(
            goals_for=fixture_data.away_ft,
            goals_against=fixture_data.home_ft,
            shots=fixture_data.away_shots,
            shots_on_target=fixture_data.away_shots_on_target,
            corners=fixture_data.away_corners,
            fouls_committed=fixture_data.away_fouls_committed,
            free_kicks=fixture_data.away_free_kicks,
            offsides=fixture_data.away_offsides,
            yellow_cards=fixture_data.away_yellow_cards,
            red_cards=fixture_data.away_red_cards,
        )

    def _push_l5_result(self, result, wins, draws, losses, points):
        self.l5_wins.pop(0)
        self.l5_wins.append(wins)
        self.l5_draws.pop(0)
        self.l5_draws.append(draws)
        self.l5_losses.pop(0)
        self.l5_losses.append(losses)
        self.l5_results.pop(0)
        self.l5_results.append(result)
        self.l5_points.pop(0)
        self.l5_points.append(points)

    def _push_l5_stats(self, goals_for, goals_against, shots, shots_on_target, corners,
                        fouls_committed, free_kicks, offsides, yellow_cards, red_cards):
        for l5_list, value in (
            (self.l5_goals_for, goals_for),
            (self.l5_goals_against, goals_against),
            (self.l5_shots, shots),
            (self.l5_shots_on_target, shots_on_target),
            (self.l5_corners, corners),
            (self.l5_fouls_committed, fouls_committed),
            (self.l5_free_kicks, free_kicks),
            (self.l5_offsides, offsides),
            (self.l5_yellow_cards, yellow_cards),
            (self.l5_red_cards, red_cards),
        ):
            l5_list.pop(0)
            l5_list.append(value)

    def get_div(self):
        return self.div

    def get_season(self):
        return self.season

    def get_pagerank(self):
        return self.page_rank

    def get_played(self):
        return self.home_played + self.away_played

    def get_league_position(self):
        return self.league_position

    def get_home_played(self):
        return self.home_played

    def get_away_played(self):
        return self.away_played

    def get_home_league_position(self):
        return self.home_league_position

    def get_away_league_position(self):
        return self.away_league_position

    def get_points(self):
        return self.home_points + self.away_points

    def get_home_points(self):
        return self.home_points

    def get_away_points(self):
        return self.away_points

    def get_goal_diff(self):
        return self.home_goals_for + self.away_goals_for - self.home_goals_against - self.away_goals_against

    def get_home_goal_diff(self):
        return self.home_goals_for - self.home_goals_against

    def get_away_goal_diff(self):
        return self.away_goals_for - self.away_goals_against

    def get_wins(self):
        return self.home_wins + self.away_wins

    def get_home_wins(self):
        return self.home_wins

    def get_away_wins(self):
        return self.away_wins

    def get_draws(self):
        return self.home_draws + self.away_draws

    def get_home_draws(self):
        return self.home_draws

    def get_away_draws(self):
        return self.away_draws

    def get_losses(self):
        return self.home_losses + self.away_losses

    def get_home_losses(self):
        return self.home_losses

    def get_away_losses(self):
        return self.away_losses

    def get_goals_for(self):
        return self.home_goals_for + self.away_goals_for

    def get_home_goals_for(self):
        return self.home_goals_for

    def get_away_goals_for(self):
        return self.away_goals_for

    def get_goals_against(self):
        return self.home_goals_against + self.away_goals_against

    def get_home_goals_against(self):
        return self.home_goals_against

    def get_away_goals_against(self):
        return self.away_goals_against

    def get_shots(self):
        return self.home_shots + self.away_shots

    def get_home_shots(self):
        return self.home_shots

    def get_away_shots(self):
        return self.away_shots

    def get_shots_on_target(self):
        return self.home_shots_on_target + self.away_shots_on_target

    def get_home_shots_on_target(self):
        return self.home_shots_on_target

    def get_away_shots_on_target(self):
        return self.away_shots_on_target

    def get_free_kicks(self):
        return self.home_free_kicks + self.away_free_kicks

    def get_home_free_kicks(self):
        return self.home_free_kicks

    def get_away_free_kicks(self):
        return self.away_free_kicks

    def get_fouls_committed(self):
        return self.home_fouls_committed + self.away_fouls_committed

    def get_home_fouls_committed(self):
        return self.home_fouls_committed

    def get_away_fouls_committed(self):
        return self.away_fouls_committed

    def get_offsides(self):
        return self.home_offsides + self.away_offsides

    def get_home_offsides(self):
        return self.home_offsides

    def get_away_offsides(self):
        return self.away_offsides

    def get_corners(self):
        return self.home_corners + self.away_corners

    def get_home_corners(self):
        return self.home_corners

    def get_away_corners(self):
        return self.away_corners

    def get_yellow_cards(self):
        return self.home_yellow_cards + self.away_yellow_cards

    def get_home_yellow_cards(self):
        return self.home_yellow_cards

    def get_away_yellow_cards(self):
        return self.away_yellow_cards

    def get_red_cards(self):
        return self.home_red_cards + self.away_red_cards

    def get_home_red_cards(self):
        return self.home_red_cards

    def get_away_red_cards(self):
        return self.away_red_cards

    def get_points_l5(self):
        return sum(self.l5_points) / 5.0

    def get_wins_l5(self):
        return sum(self.l5_wins) / 5.0

    def get_draws_l5(self):
        return sum(self.l5_draws) / 5.0

    def get_losses_l5(self):
        return sum(self.l5_losses) / 5.0

    def get_league_position_l5(self):
        return sum(self.l5_league_position) / 5.0

    def get_goals_for_l5(self):
        return sum(self.l5_goals_for) / 5.0

    def get_goals_against_l5(self):
        return sum(self.l5_goals_against) / 5.0

    def get_shots_l5(self):
        return sum(self.l5_shots) / 5.0

    def get_shots_on_target_l5(self):
        return sum(self.l5_shots_on_target) / 5.0

    def get_corners_l5(self):
        return sum(self.l5_corners) / 5.0

    def get_free_kicks_l5(self):
        return sum(self.l5_free_kicks) / 5.0

    def get_offsides_l5(self):
        return sum(self.l5_offsides) / 5.0

    def get_fouls_committed_l5(self):
        return sum(self.l5_fouls_committed) / 5.0

    def get_yellow_cards_l5(self):
        return sum(self.l5_yellow_cards) / 5.0

    def get_red_cards_l5(self):
        return sum(self.l5_red_cards) / 5.0

    def get_results_l5(self):
        return ','.join(self.l5_results)
