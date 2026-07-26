import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

import pandas as pd
import yaml

from .train_model import load_model_bundle

OUTCOME_CLASSES = ('H', 'D', 'A')
OUTCOME_ODDS_COLUMNS = {'H': 'home_win_odds_avg', 'D': 'draw_odds_avg', 'A': 'away_win_odds_avg'}


@dataclass
class Bet:
    date: str
    div: str
    season: str
    home_team: str
    away_team: str
    predicted_class: str
    model_prob: float
    fair_prob: float
    edge: float
    odds: float
    stake: float
    actual_result: str
    won: bool
    profit: float
    bankroll_after: float


class ScenarioModel:
    """Simulates betting a bankroll on the trained model's value bets across
    a season: walks that season's already-played fixtures in chronological
    order (across one or more divisions sharing a single bankroll), and for
    every fixture where the model's probability beats the de-vigged market
    price by more than edge_threshold, places a whole-euro stake sized by
    the Kelly criterion -- the fraction of the *current* bankroll that
    maximises expected log growth given the model's probability and the
    market's offered odds -- then resolves it against the fixture's actual
    result.

    This reuses the same edge/value-bet logic as Predict and the training
    backtest, but as a standalone tool that can be re-run against different
    seasons/divisions/staking settings without retraining, and produces a
    full bet-by-bet ledger with a running bankroll rather than just an
    aggregate.
    """

    def __init__(self, config_filename, verbose_flag=False, model_bundle=None):
        self.verbose = verbose_flag
        with open(config_filename, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        bundle = model_bundle or load_model_bundle(self.config['trained_model']['model_path'])
        self.model = bundle['model']
        self.feature_columns = bundle['feature_columns']
        self.classes = bundle['classes']

    def run(self, season, divisions=None, starting_bankroll=None, edge_threshold=None,
            kelly_fraction=None, bet_classes=None, take_profit_bankroll=None,
            stake_menu=None, kelly_shrinkage=None, max_stake_fraction=None,
            flat_stake=None, output_csv=None):
        cfg = self.config.get('scenario', {})
        starting_bankroll = self._coalesce(starting_bankroll, cfg.get('starting_bankroll', 1000.0))
        edge_threshold = self._coalesce(edge_threshold, cfg.get('edge_threshold', 0.05))
        kelly_fraction = self._coalesce(kelly_fraction, cfg.get('kelly_fraction', 1.0))
        bet_classes = self._coalesce(bet_classes, cfg.get('bet_classes', None)) or list(OUTCOME_CLASSES)
        take_profit_bankroll = self._coalesce(take_profit_bankroll, cfg.get('take_profit_bankroll', None))
        stake_menu = self._coalesce(stake_menu, cfg.get('stake_menu', None))
        stake_menu = sorted(stake_menu) if stake_menu else None
        kelly_shrinkage = self._coalesce(kelly_shrinkage, cfg.get('kelly_shrinkage', 0.0))
        max_stake_fraction = self._coalesce(max_stake_fraction, cfg.get('max_stake_fraction', None))
        flat_stake = self._coalesce(flat_stake, cfg.get('flat_stake', None))

        df = self.load_season_features(season, divisions)
        if self.verbose:
            print(f"Loaded {len(df)} played fixtures for season {season} "
                  f"across divisions {sorted(df['div'].unique())}, betting on {bet_classes} only")

        bankroll = starting_bankroll
        peak = starting_bankroll
        max_drawdown = 0.0
        ledger: List[Bet] = []
        bankrupt_at = None
        stopped_at_take_profit = None

        candidates = self._score_candidates(df, edge_threshold, bet_classes)

        for _, row in candidates.iterrows():
            if bankroll <= 0:
                bankrupt_at = row['date']
                break

            odds = {cls: row[OUTCOME_ODDS_COLUMNS[cls]] for cls in OUTCOME_CLASSES}
            best_class = row['_best_class']
            best_edge = row['_best_edge']
            probabilities = {cls: row[f'_proba_{cls}'] for cls in OUTCOME_CLASSES}
            fair_prob = {cls: row[f'_fair_{cls}'] for cls in OUTCOME_CLASSES}

            if flat_stake is not None:
                stake = min(flat_stake, bankroll)
            elif stake_menu:
                stake = self._menu_stake(probabilities[best_class], fair_prob[best_class], odds[best_class],
                                          bankroll, kelly_fraction, stake_menu, kelly_shrinkage, max_stake_fraction)
            else:
                stake = self._kelly_stake(probabilities[best_class], fair_prob[best_class], odds[best_class],
                                           bankroll, kelly_fraction, kelly_shrinkage, max_stake_fraction)
            if stake < 1.0:
                continue  # rounds/snaps to less than a whole euro, or the bankroll can't afford any menu stake

            actual_result = self._actual_result(row)
            won = actual_result == best_class
            profit = stake * (odds[best_class] - 1.0) if won else -stake
            bankroll += profit
            peak = max(peak, bankroll)
            max_drawdown = max(max_drawdown, peak - bankroll)

            ledger.append(Bet(
                date=row['date'], div=row['div'], season=row['season'],
                home_team=row['home_team'], away_team=row['away_team'],
                predicted_class=best_class, model_prob=probabilities[best_class],
                fair_prob=fair_prob[best_class], edge=best_edge, odds=odds[best_class],
                stake=stake, actual_result=actual_result, won=won, profit=profit,
                bankroll_after=bankroll,
            ))

            if take_profit_bankroll is not None and peak > take_profit_bankroll:
                stopped_at_take_profit = row['date']
                break

        summary = self._summarize(starting_bankroll, bankroll, peak, max_drawdown, ledger,
                                   bankrupt_at, stopped_at_take_profit)
        self._print_summary(summary)

        if output_csv:
            self._write_csv(output_csv, ledger)
            if self.verbose:
                print(f"Wrote {len(ledger)}-row bet ledger to {output_csv}")

        return summary, ledger

    def load_season_features(self, season, divisions=None):
        features_folder = self.config['features']['features_folder']
        rows = []
        for file_name in sorted(os.listdir(features_folder)):
            if not file_name.endswith('.json'):
                continue
            file_season, _, file_div = file_name[:-len('.json')].partition('_')
            if file_season != str(season):
                continue
            if divisions and file_div not in divisions:
                continue
            with open(os.path.join(features_folder, file_name), 'r') as f:
                rows.extend(json.load(f))

        if len(rows) == 0:
            raise ValueError(
                f"No feature rows found for season {season}"
                + (f" divisions {divisions}" if divisions else "")
            )

        df = pd.DataFrame(rows)
        return df.sort_values(['date', 'div', 'home_team'], kind='stable').reset_index(drop=True)

    def _score_candidates(self, df, edge_threshold, bet_classes):
        """Batch-scores every row with the model in one call (much faster
        than one predict_proba per fixture), computes the de-vigged market
        price and each row's best edge -- considering only bet_classes, e.g.
        ['H'] to only ever back the home team -- and returns the rows with
        odds coverage and edge > edge_threshold, preserving df's
        chronological order for the sequential bankroll walk in run().
        """
        proba = pd.DataFrame(
            self.model.predict_proba(df[self.feature_columns]),
            columns=self.classes, index=df.index,
        )
        for cls in OUTCOME_CLASSES:
            df[f'_proba_{cls}'] = proba[cls]

        odds = df[[OUTCOME_ODDS_COLUMNS[cls] for cls in OUTCOME_CLASSES]].copy()
        odds.columns = list(OUTCOME_CLASSES)
        has_odds = (odds > 0).all(axis=1)

        implied = 1.0 / odds[has_odds]
        overround = implied.sum(axis=1)
        fair_prob = implied.div(overround, axis=0)
        for cls in OUTCOME_CLASSES:
            df.loc[has_odds, f'_fair_{cls}'] = fair_prob[cls]

        edge = proba.loc[has_odds, bet_classes] - fair_prob[bet_classes]
        df.loc[has_odds, '_best_class'] = edge.idxmax(axis=1)
        df.loc[has_odds, '_best_edge'] = edge.max(axis=1)

        candidates = df[has_odds & (df['_best_edge'] > edge_threshold)]
        return candidates.sort_values(['date', 'div', 'home_team'], kind='stable')

    def _actual_result(self, row):
        if row['is_home_win'] == 1:
            return 'H'
        if row['is_away_win'] == 1:
            return 'A'
        return 'D'

    def _raw_kelly_stake(self, prob, fair_prob, odds, bankroll, kelly_fraction, kelly_shrinkage=0.0):
        """Unconstrained Kelly criterion stake for a single bet: f* = p -
        (1-p)/b, where b is the net odds (decimal odds - 1). f* is the
        bankroll fraction that maximises expected log growth; kelly_fraction
        scales it down (e.g. 0.5 for "half Kelly") to trade growth for lower
        variance -- 1.0 is full Kelly.

        p is not used at face value: it's shrunk toward the de-vigged
        market probability fair_prob by kelly_shrinkage (0 = trust the
        model's probability entirely, 1 = size as if there's no edge at
        all). The model's biggest edge estimates are also its least
        reliable ones (isotonic calibration overfits in exactly the sparse,
        high-confidence region that produces them -- see the calibration
        comparison), and raw Kelly stakes in direct proportion to the edge,
        so without shrinkage it systematically puts the most money on the
        estimates most likely to be wrong. bet *selection* still uses the
        model's raw probability/edge -- this only affects how much gets
        staked once a bet has already been chosen.
        """
        b = odds - 1.0
        if b <= 0:
            return 0.0
        p_stake = (1.0 - kelly_shrinkage) * prob + kelly_shrinkage * fair_prob
        f_star = max(p_stake - (1.0 - p_stake) / b, 0.0)
        return kelly_fraction * f_star * bankroll

    def _cap_stake(self, raw_stake, bankroll, max_stake_fraction):
        if max_stake_fraction is not None:
            raw_stake = min(raw_stake, max_stake_fraction * bankroll)
        return raw_stake

    def _kelly_stake(self, prob, fair_prob, odds, bankroll, kelly_fraction,
                      kelly_shrinkage=0.0, max_stake_fraction=None):
        """Kelly stake rounded to a whole euro and capped at the current
        bankroll (and, if set, at max_stake_fraction of it).
        """
        raw_stake = self._raw_kelly_stake(prob, fair_prob, odds, bankroll, kelly_fraction, kelly_shrinkage)
        raw_stake = self._cap_stake(raw_stake, bankroll, max_stake_fraction)
        whole_stake = round(raw_stake)
        return float(max(0.0, min(whole_stake, bankroll)))

    def _menu_stake(self, prob, fair_prob, odds, bankroll, kelly_fraction, stake_menu,
                     kelly_shrinkage=0.0, max_stake_fraction=None):
        """Kelly stake snapped to the closest value in stake_menu that the
        current bankroll can afford, rather than any whole euro -- e.g. to
        keep stakes looking like typical round amounts a casual bettor might
        place, instead of precise algorithmically-sized figures.

        Whether to bet at all is decided the same way as the plain
        (unrestricted) Kelly stake -- round to the nearest whole euro and
        require >= 1 -- so the menu only changes what a placed bet's stake
        looks like, not which fixtures get bet on. Without this, a fixture
        whose true Kelly stake is a small fraction of a euro (and would
        normally round down to 0 and be skipped) would still get snapped up
        to the menu's smallest value, forcing a flood of near-threshold,
        low-quality bets that were never supposed to be placed.
        """
        raw_stake = self._raw_kelly_stake(prob, fair_prob, odds, bankroll, kelly_fraction, kelly_shrinkage)
        raw_stake = self._cap_stake(raw_stake, bankroll, max_stake_fraction)
        whole_stake = round(raw_stake)
        if whole_stake < 1.0:
            return 0.0

        affordable = [value for value in stake_menu if value <= bankroll]
        if not affordable:
            return 0.0

        return float(min(affordable, key=lambda value: abs(value - whole_stake)))

    def _summarize(self, starting_bankroll, ending_bankroll, peak, max_drawdown, ledger,
                    bankrupt_at, stopped_at_take_profit=None):
        total_bets = len(ledger)
        wins = sum(1 for bet in ledger if bet.won)
        total_staked = sum(bet.stake for bet in ledger)
        total_profit = ending_bankroll - starting_bankroll
        return {
            'starting_bankroll': starting_bankroll,
            'ending_bankroll': ending_bankroll,
            'total_bets': total_bets,
            'win_rate': (wins / total_bets) if total_bets > 0 else 0.0,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': (total_profit / total_staked) if total_staked > 0 else 0.0,
            'peak_bankroll': peak,
            'max_drawdown': max_drawdown,
            'bankrupt_at': bankrupt_at,
            'stopped_at_take_profit': stopped_at_take_profit,
        }

    def _print_summary(self, summary):
        print("\n=== Season scenario summary ===")
        print(f"Starting bankroll: {summary['starting_bankroll']:.2f}")
        print(f"Ending bankroll:   {summary['ending_bankroll']:.2f}")
        print(f"Bets placed: {summary['total_bets']}  Win rate: {summary['win_rate']:.3f}")
        print(f"Total staked: {summary['total_staked']:.2f}  "
              f"Total profit: {summary['total_profit']:.2f}  ROI: {summary['roi']:.4f}")
        print(f"Peak bankroll: {summary['peak_bankroll']:.2f}  Max drawdown: {summary['max_drawdown']:.2f}")
        if summary['bankrupt_at']:
            print(f"Ran out of bankroll on {summary['bankrupt_at']} -- stopped simulation early.")
        if summary['stopped_at_take_profit']:
            print(f"Peak bankroll target hit on {summary['stopped_at_take_profit']} -- stopped simulation early.")

    def _write_csv(self, output_csv, ledger):
        fieldnames = list(Bet.__dataclass_fields__.keys())
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for bet in ledger:
                writer.writerow(asdict(bet))

    @staticmethod
    def _coalesce(value, default):
        return default if value is None else value
