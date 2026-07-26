import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

INFORMATIONAL_COLUMNS = [
    'season', 'div', 'home_team', 'away_team', 'home_ft', 'away_ft',
    'is_home_win', 'is_away_win', 'is_draw',
]

MODEL_BUNDLE_VERSION = 1


class TrainModel:
    """Trains a single 3-way (home/draw/away) calibrated classifier on the
    generated feature data, evaluated on the most recent season(s) held out
    entirely from training/tuning, and backtests a simple value-bet strategy
    against the bookmaker odds already present as features.
    """

    def __init__(self, config_filename, verbose_flag=False):
        self.verbose = verbose_flag
        with open(config_filename, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def run(self):
        df = self.load_features(self.config['features']['features_folder'])
        if self.verbose:
            print(f"Loaded {len(df)} feature rows across {df['season'].nunique()} seasons")

        train_cfg = self.config.get('train', {})
        holdout_seasons = int(train_cfg.get('holdout_seasons', 1))
        edge_threshold = float(train_cfg.get('edge_threshold', 0.05))
        stake = float(train_cfg.get('stake', 10.0))

        df_train, df_holdout = self.time_based_split(df, holdout_seasons)
        if self.verbose:
            print(f"Training on seasons {sorted(df_train['season'].unique())}")
            print(f"Holding out seasons {sorted(df_holdout['season'].unique())} ({len(df_holdout)} rows)")

        X_train, y_train, groups_train = self.split_features_target(df_train)
        X_holdout, y_holdout, _ = self.split_features_target(df_holdout)

        best_params = self.search_hyperparameters(X_train, y_train, groups_train)
        if self.verbose:
            print(f"Best hyperparameters: {best_params}")

        model = self.fit_calibrated_model(X_train, y_train, groups_train, best_params)

        self.evaluate(model, X_holdout, y_holdout)
        self.backtest_value_bets(model, df_holdout, X_holdout, edge_threshold, stake)

        self.save(model, list(X_train.columns), best_params, df_train, df_holdout)

    def load_features(self, features_folder):
        rows = []
        for file_name in sorted(os.listdir(features_folder)):
            if file_name.endswith('.json'):
                with open(os.path.join(features_folder, file_name), 'r') as f:
                    rows.extend(json.load(f))
        df = pd.DataFrame(rows)
        df['result'] = np.select(
            [df['is_home_win'] == 1, df['is_away_win'] == 1],
            ['H', 'A'],
            default='D',
        )
        return df

    def time_based_split(self, df, holdout_seasons):
        """Splits chronologically by season string (e.g. '2425' sorts after
        '2324'), holding out the most recent `holdout_seasons` seasons
        entirely, so evaluation reflects forecasting a season the model has
        never seen rather than a random shuffle of already-historical rows.
        """
        seasons = sorted(df['season'].unique())
        if len(seasons) <= holdout_seasons:
            raise ValueError(
                f"Only {len(seasons)} season(s) of features available; need more than "
                f"holdout_seasons={holdout_seasons} to leave any training data."
            )
        holdout_set = set(seasons[-holdout_seasons:])
        df_holdout = df[df['season'].isin(holdout_set)]
        df_train = df[~df['season'].isin(holdout_set)]
        return df_train, df_holdout

    def split_features_target(self, df):
        X = df.drop(columns=INFORMATIONAL_COLUMNS + ['result'], errors='ignore')
        y = df['result']
        groups = df['season']
        return X, y, groups

    def search_hyperparameters(self, X_train, y_train, groups_train, n_splits=4, n_iter=15):
        """Small randomized search over HistGradientBoostingClassifier
        hyperparameters, scored with season-grouped CV (GroupKFold) so no
        fold mixes rows from the same season across train/validation.
        """
        n_splits = min(n_splits, groups_train.nunique())
        cv = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups=groups_train))

        param_distributions = {
            'max_iter': randint(100, 400),
            'max_depth': randint(2, 8),
            'learning_rate': uniform(0.02, 0.18),
            'l2_regularization': uniform(0.0, 1.0),
        }

        search = RandomizedSearchCV(
            estimator=HistGradientBoostingClassifier(random_state=42),
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='neg_log_loss',
            cv=cv,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        return search.best_params_

    def fit_calibrated_model(self, X_train, y_train, groups_train, best_params, n_splits=4):
        """Fits the tuned model wrapped in probability calibration, using the
        same season-grouped folds so calibration is also evaluated on unseen
        seasons rather than a random split.
        """
        n_splits = min(n_splits, groups_train.nunique())
        cv = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups=groups_train))
        base_estimator = HistGradientBoostingClassifier(random_state=42, **best_params)
        model = CalibratedClassifierCV(base_estimator, method='isotonic', cv=cv)
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X_holdout, y_holdout):
        classes = list(model.classes_)
        proba = model.predict_proba(X_holdout)
        predictions = model.predict(X_holdout)

        accuracy = accuracy_score(y_holdout, predictions)
        loss = log_loss(y_holdout, proba, labels=classes)

        one_hot = pd.get_dummies(y_holdout)[classes].to_numpy()
        brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

        majority_class = y_holdout.value_counts().idxmax()
        baseline_accuracy = (y_holdout == majority_class).mean()
        naive_uniform_loss = float(np.log(len(classes)))

        print("\n=== Holdout evaluation ===")
        print(f"Rows: {len(y_holdout)}, classes: {classes}")
        print(f"Accuracy: {accuracy:.4f} (always predict '{majority_class}': {baseline_accuracy:.4f})")
        print(f"Log loss: {loss:.4f} (uniform-random baseline: {naive_uniform_loss:.4f})")
        print(f"Multiclass Brier score: {brier:.4f}")
        print(f"Class distribution (holdout): {y_holdout.value_counts(normalize=True).to_dict()}")

    def backtest_value_bets(self, model, df_holdout, X_holdout, edge_threshold, stake):
        """Backtests: bet a flat stake on whichever outcome has the largest
        (model probability - de-vigged bookmaker implied probability), when
        that edge exceeds edge_threshold. Skips rows with no odds coverage.
        """
        classes = list(model.classes_)
        proba = pd.DataFrame(model.predict_proba(X_holdout), columns=classes, index=df_holdout.index)

        odds = df_holdout[['home_win_odds_avg', 'draw_odds_avg', 'away_win_odds_avg']].rename(
            columns={'home_win_odds_avg': 'H', 'draw_odds_avg': 'D', 'away_win_odds_avg': 'A'}
        )
        has_odds = (odds > 0).all(axis=1)

        implied = 1.0 / odds[has_odds]
        overround = implied.sum(axis=1)
        fair_prob = implied.div(overround, axis=0)

        edge = proba.loc[has_odds, classes] - fair_prob[classes]
        best_class = edge.idxmax(axis=1)
        best_edge = edge.max(axis=1)

        bets = best_edge[best_edge > edge_threshold]
        if len(bets) == 0:
            print("\n=== Value-bet backtest ===")
            print(f"No bets cleared the edge threshold ({edge_threshold}).")
            return

        bet_rows = df_holdout.loc[bets.index]
        bet_class = best_class.loc[bets.index]
        bet_odds = odds.loc[bets.index].to_numpy()
        chosen_odds = np.array([odds.loc[idx, cls] for idx, cls in bet_class.items()])
        actual = bet_rows['result'].to_numpy()
        won = actual == bet_class.to_numpy()

        profit = np.where(won, stake * (chosen_odds - 1.0), -stake)
        total_staked = stake * len(bets)
        total_profit = float(profit.sum())
        roi = total_profit / total_staked if total_staked > 0 else 0.0

        print("\n=== Value-bet backtest ===")
        print(f"Bets placed: {len(bets)} / {has_odds.sum()} odds-covered holdout rows (edge > {edge_threshold})")
        print(f"Win rate: {won.mean():.4f}")
        print(f"Total staked: {total_staked:.2f}, total profit: {total_profit:.2f}, ROI: {roi:.4f}")

    def save(self, model, feature_columns, best_params, df_train, df_holdout):
        model_path = self.config['trained_model']['model_path']
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        bundle = {
            'version': MODEL_BUNDLE_VERSION,
            'model': model,
            'feature_columns': feature_columns,
            'classes': list(model.classes_),
            'best_params': best_params,
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'train_seasons': sorted(df_train['season'].unique().tolist()),
            'holdout_seasons': sorted(df_holdout['season'].unique().tolist()),
        }
        joblib.dump(bundle, model_path)
        if self.verbose:
            print(f"\nSaved model bundle to {model_path}")


def load_model_bundle(model_path):
    return joblib.load(model_path)
