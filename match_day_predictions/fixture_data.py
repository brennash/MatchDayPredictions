from dataclasses import dataclass

@dataclass(frozen=True)
class FixtureData:
    season: str = ""
    div: str = ""
    date: str = ""
    time: str = ""
    home_team: str = ""
    away_team: str = ""
    home_ht: float = 0.0
    home_ft: float = 0.0
    home_shots: float = 0.0
    home_shots_on_target: float = 0.0
    home_corners: float = 0.0
    home_fouls_committed: float = 0.0
    home_free_kicks: float = 0.0
    home_offsides: float = 0.0
    home_yellow_cards: float = 0.0
    home_red_cards: float = 0.0
    away_ht: float = 0.0
    away_ft: float = 0.0
    away_shots: float = 0.0
    away_shots_on_target: float = 0.0
    away_corners: float = 0.0
    away_fouls_committed: float = 0.0
    away_free_kicks: float = 0.0
    away_offsides: float = 0.0
    away_yellow_cards: float = 0.0
    away_red_cards: float = 0.0
    home_win_odds_min: float = 0.0
    home_win_odds_max: float = 0.0
    home_win_odds_avg: float = 0.0
    home_win_odds_range: float = 0.0
    home_win_odds_std_dev: float = 0.0
    draw_odds_min: float = 0.0
    draw_odds_max: float = 0.0
    draw_odds_avg: float = 0.0
    draw_odds_range: float = 0.0
    draw_odds_std_dev: float = 0.0
    away_win_odds_min: float = 0.0
    away_win_odds_max: float = 0.0
    away_win_odds_avg: float = 0.0
    away_win_odds_range: float = 0.0
    away_win_odds_std_dev: float = 0.0
    over_two_and_a_half_goals_min: float = 0.0
    over_two_and_a_half_goals_max: float = 0.0
    over_two_and_a_half_goals_avg: float = 0.0
    over_two_and_a_half_goals_range: float = 0.0
    over_two_and_a_half_goals_std_dev: float = 0.0
    under_two_and_a_half_goals_min: float = 0.0
    under_two_and_a_half_goals_max: float = 0.0
    under_two_and_a_half_goals_avg: float = 0.0
    under_two_and_a_half_goals_range: float = 0.0
    under_two_and_a_half_goals_std_dev: float = 0.0
    result: str = ""

    def is_home_win(self) -> bool:
        return self.result == "H"

    def is_away_win(self) -> bool:
        return self.result == "A"

    def is_draw(self) -> bool:
        return self.result == "D"
