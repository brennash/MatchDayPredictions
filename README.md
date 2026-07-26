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

Simulates betting a bankroll on the model's value bets across an already-played season, to see how the strategy would have done. Walks the season's fixtures in chronological order (across one division or all of them, sharing a single bankroll), and for every fixture where the model's probability beats the de-vigged bookmaker price by more than `edge_threshold`, places a stake and resolves it against the actual result:

```
.venv/bin/python main.py -v scenario --season 2526 --div E0 \
    --starting-bankroll 1000 --edge-threshold 0.05 --flat-stake 10 --output ledger.csv
```

Omit `--div` to run across every division with data for that season, sharing one bankroll. Prints a summary (bets placed, win rate, ROI, peak bankroll, max drawdown) and, with `--output`, writes the full bet-by-bet ledger (date, teams, odds, stake, result, running bankroll) as CSV. Defaults live under `scenario:` in `conf/config.yaml`.

**Staking: `--flat-stake` is the recommended mode** -- the same fixed amount on every qualifying bet. A whole family of Kelly-criterion variants was tried (`--kelly-fraction` for full/half/quarter Kelly, `--stake-menu` to snap stakes to round "human-looking" amounts, `--kelly-shrinkage` to blend the model's probability toward the market's before sizing, `--max-stake-fraction` to cap a stake as a % of bankroll, `--take-profit` to stop once ahead by X) and all of them are still available, but every one underperformed flat staking once evaluated on a large enough, untruncated sample. The reason is structural, not a tuning mistake: Kelly-family sizing scales the stake with the model's own edge estimate, and that estimate is *least* reliable exactly where it's largest (see the calibration note below) -- so proportional staking systematically overweights the model's least trustworthy signals. Flat staking doesn't look at the edge size at all, so it can't make that mistake. Full details are in `conf/config.yaml`'s `scenario.flat_stake` comment; the walk-forward numbers behind it are below.

### Validated strategy performance

Walk-forward backtest: for each season, train on every season strictly before it (so the model never sees the season it's betting on), then run the scenario modeller with a flat EUR10 stake, home-win-only, edge_threshold 0.05, isotonic calibration, EUR1000 starting bankroll reset every season (no compounding across seasons). 2020-21 is excluded throughout -- see below.

| Season | Bets | Win rate | Staked | Profit | Ending | Peak | Max drawdown | ROI |
|---|---|---|---|---|---|---|---|---|
| 2017-18 | 324 | 61.7% | €3,240 | −€0.90 | €999.10 | €1,158.86 | €195.38 | −0.03% |
| 2018-19 | 186 | 42.5% | €1,860 | −€110.76 | €889.24 | €1,051.50 | €166.97 | −5.96% |
| 2019-20 | 239 | 50.6% | €2,390 | +€174.22 | €1,174.22 | €1,224.22 | €166.70 | +7.29% |
| **2020-21** | **299** | **46.5%** | **€2,990** | **−€463.64** | -- | -- | **€523.94*** | **−15.5%** |
| 2021-22 | 127 | 62.2% | €1,270 | +€37.69 | €1,037.69 | €1,100.05 | €93.01 | +2.97% |
| 2022-23 | 118 | 70.3% | €1,180 | +€17.89 | €1,017.89 | €1,044.26 | €58.49 | +1.52% |
| 2023-24 | 181 | 63.5% | €1,810 | +€86.60 | €1,086.60 | €1,127.28 | €86.21 | +4.78% |
| 2024-25 | 129 | 54.3% | €1,290 | +€28.03 | €1,028.03 | €1,092.53 | €156.88 | +2.17% |
| 2025-26 | 83 | 59.0% | €830 | +€1.53 | €1,001.53 | €1,100.48 | €103.88 | +0.18% |

*2020-21's ending/peak/drawdown are quoted from the EUR500-bankroll run that first surfaced it; the pattern (near-wipeout) doesn't depend on the bankroll size.

**8 seasons excluding 2020-21: 6 of 8 positive, 1,387 bets, +€234.31 total profit, +1.69% edge on the bookies.** This supersedes an earlier 6-season read (2019-20..2025-26 only) that showed +3.95%: adding 2017-18 and 2018-19 weakens the aggregate, mainly because 2018-19 is a real loss on a decent sample (186 bets, -6%). Plausibly those two seasons' models were trained on less history (7-8 prior seasons vs. 11-15 for the later ones) and so were noisier, but that's not confirmed -- treat it as one more data point that the edge is real-looking but thin, not proof either way.

**2020-21 is excluded** for a specific, pre-existing reason, not because it lost: matches were played without crowds that season, a well-documented cause of reduced real home advantage that a model trained only on normal-crowd seasons has no way to anticipate. It's not small-sample noise either -- 299 bets is a large sample, and the bankroll came within a few euro of being wiped out (96% drawdown) even at flat stakes.

A EUR1000-bankroll, stop-at-+EUR250-profit variant was also tested: the take-profit never actually triggered in any of the 8 seasons (peak got as high as EUR1,224 in 2019-20 but never crossed EUR1,250) -- at these flat stakes and this edge size, a season just doesn't generate enough cumulative swing to hit a EUR250 target before it ends. So that variant's numbers are identical to running with no take-profit at all; the table above already reflects that.

Even in the better seasons, expect real drawdowns -- up to €195.38 in 2017-18, on top of two outright losing seasons. This is a modest, noisy edge on a small sample by any statistical standard, not a proven system; treat it accordingly.

## Running tests

```
.venv/bin/python -m pytest
```

## Notes

- A team's rolling stats are considered too noisy to use until it has played `league.min_games_played` (default 5) matches; earlier fixtures are excluded from training data and from pagerank/league-position updates.
- The model is evaluated on `train.holdout_seasons` (default 1) most-recent season(s) that never appear in training or hyperparameter tuning, so the reported accuracy/log-loss reflect forecasting a season the model hasn't seen rather than a random shuffle of historical rows.
- The value-bet backtest bets a flat stake whenever the model's probability exceeds the bookmaker's de-vigged implied probability by more than `train.edge_threshold` (default 0.05) — see `conf/config.yaml`.
- `train.calibration_method` is `isotonic`. Both `isotonic` and `sigmoid` give near-identical accuracy/log-loss and both show the same failure mode at edge > 0.08 (calibration overfits in sparse high-confidence regions, producing "big edge" bets that lose). At edge > 0.05 -- the threshold with enough bet volume in both cases to mean anything -- a walk-forward comparison across the **last five seasons** (2021-22..2025-26, each trained only on data strictly before it) found isotonic positive in 3/5 seasons with a ~2x larger sample than sigmoid (905 vs 466 bets total) and a much smaller worst-case season (essentially flat, -0.2%, vs sigmoid's real -15.2% miss in 2022-23). An earlier pass using only the most recent 2 seasons had pointed to sigmoid instead -- that was an artifact of a too-small sample happening to land on sigmoid's two best seasons, and doesn't hold up against the full 5. This is the general risk with tuning a betting strategy against backtest results: with ~100-200 bets in a season, any one season (or even two) is still mostly noise, and optimizing against a small holdout easily finds a configuration that "works" there by chance and nowhere else. Even the 5-season read here isn't a statistically airtight edge -- it's a more defensible one than a 1- or 2-season read, not a proven one.
- Extending the walk-forward window back to 2019-20 (7 seasons total) surfaced 2020-21 as a genuine outlier: matches played without crowds that season, a well-documented reduction in real home advantage that a model trained only on normal-crowd seasons has no way to anticipate. It's excluded from the headline `scenario.flat_stake` numbers in `conf/config.yaml` for that specific, pre-existing reason -- not because it happened to lose. Left in, it drags the 7-season aggregate to roughly breakeven entirely on its own.
