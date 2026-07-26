# MatchDayPredictions
Predicting value bets for European League football.

Given a division and season, `match_day_predictions` replays every match played so far to build each team's current form (points, goals, shots, cards, home/away splits, last-5-match rolling stats, league position and a graph-pagerank "who beat whom" score), then trains a calibrated 3-way (home win / draw / away win) classifier on ~15 seasons of history across 22 European divisions. The same team-state computation is used at prediction time, so a forecast for an unplayed fixture is built from exactly the same features the model was trained on.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Pipeline

Each step reads/writes the folders configured in `conf/config.yaml`.

```
# 1. Raw football-data.co.uk CSVs (data/fixtures_raw/<season>/<div>.csv) -> per season/division JSON
.venv/bin/python main.py -v import

# 2. Fixture JSON -> one feature row per match (data/features/<season>_<div>.json)
.venv/bin/python main.py -v features

# 3. Train the model, evaluated on the most recent season held out entirely,
#    including a value-bet backtest against the odds already in the features
.venv/bin/python main.py -v train

# 4. Predict a fixture using each team's current-season form
.venv/bin/python main.py predict --div E0 --season 2526 --home "Man City" --away "Arsenal" \
    --home-odds 1.9 --draw-odds 3.8 --away-odds 4.0
```

`predict` also supports batch mode over a CSV:

```
.venv/bin/python main.py predict --fixtures-file upcoming.csv
```

with columns `div,season,home_team,away_team` and optional `home_odds,draw_odds,away_odds`. When odds are supplied (single-shot or batch), the output includes the de-vigged market-implied probability and flags the outcome with the largest positive edge over that market price.

## Scenario modeller

Simulates betting a bankroll on the model's value bets across an already-played season, to see how the strategy would have done. Walks the season's fixtures in chronological order (across one division or all of them, sharing a single bankroll), and for every fixture where the model's probability beats the de-vigged bookmaker price by more than `edge_threshold`, places a whole-euro stake sized by the **Kelly criterion** -- the fraction of the *current* bankroll that maximises expected log growth given the model's probability and the odds on offer -- then resolves it against the actual result:

```
.venv/bin/python main.py -v scenario --season 2526 --div E0 \
    --starting-bankroll 1000 --edge-threshold 0.05 --kelly-fraction 0.5 --output ledger.csv
```

Omit `--div` to run across every division with data for that season, sharing one bankroll. `--kelly-fraction` scales the stake down from full Kelly (1.0, the default) -- e.g. 0.5 for "half Kelly", a standard way to trade growth rate for lower variance/drawdown. Prints a summary (bets placed, win rate, ROI, peak bankroll, max drawdown) and, with `--output`, writes the full bet-by-bet ledger (date, teams, odds, stake, result, running bankroll) as CSV. Defaults live under `scenario:` in `conf/config.yaml`.

## Running tests

```
.venv/bin/python -m pytest
```

## Notes

- A team's rolling stats are considered too noisy to use until it has played `league.min_games_played` (default 5) matches; earlier fixtures are excluded from training data and from pagerank/league-position updates.
- The model is evaluated on `train.holdout_seasons` (default 1) most-recent season(s) that never appear in training or hyperparameter tuning, so the reported accuracy/log-loss reflect forecasting a season the model hasn't seen rather than a random shuffle of historical rows.
- The value-bet backtest bets a flat stake whenever the model's probability exceeds the bookmaker's de-vigged implied probability by more than `train.edge_threshold` (default 0.05) — see `conf/config.yaml`.
- `train.calibration_method` is `isotonic`. Both `isotonic` and `sigmoid` give near-identical accuracy/log-loss and both show the same failure mode at edge > 0.08 (calibration overfits in sparse high-confidence regions, producing "big edge" bets that lose). At edge > 0.05 -- the threshold with enough bet volume in both cases to mean anything -- a walk-forward comparison across the **last five seasons** (2021-22..2025-26, each trained only on data strictly before it) found isotonic positive in 3/5 seasons with a ~2x larger sample than sigmoid (905 vs 466 bets total) and a much smaller worst-case season (essentially flat, -0.2%, vs sigmoid's real -15.2% miss in 2022-23). An earlier pass using only the most recent 2 seasons had pointed to sigmoid instead -- that was an artifact of a too-small sample happening to land on sigmoid's two best seasons, and doesn't hold up against the full 5. This is the general risk with tuning a betting strategy against backtest results: with ~100-200 bets in a season, any one season (or even two) is still mostly noise, and optimizing against a small holdout easily finds a configuration that "works" there by chance and nowhere else. Even the 5-season read here isn't a statistically airtight edge -- it's a more defensible one than a 1- or 2-season read, not a proven one.
