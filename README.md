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

## Running tests

```
.venv/bin/python -m pytest
```

## Notes

- A team's rolling stats are considered too noisy to use until it has played `league.min_games_played` (default 5) matches; earlier fixtures are excluded from training data and from pagerank/league-position updates.
- The model is evaluated on `train.holdout_seasons` (default 1) most-recent season(s) that never appear in training or hyperparameter tuning, so the reported accuracy/log-loss reflect forecasting a season the model hasn't seen rather than a random shuffle of historical rows.
- The value-bet backtest bets a flat stake whenever the model's probability exceeds the bookmaker's de-vigged implied probability by more than `train.edge_threshold` (default 0.05) — see `conf/config.yaml`.
