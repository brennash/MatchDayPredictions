import argparse
import sys

from match_day_predictions.import_fixtures import ImportFixtures
from match_day_predictions.create_features import CreateFeatures
from match_day_predictions.train_model import TrainModel
from match_day_predictions.predict import Predict, format_prediction


def build_parser():
    parser = argparse.ArgumentParser(prog='main.py', description='MatchDayPredictions pipeline')
    parser.add_argument('-v', '--verbose', action='store_true', help='Set verbose mode')
    parser.add_argument('-c', '--config', default='conf/config.yaml', help='Path to the YAML config file')

    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('import', help='Import raw fixture CSVs into per season/division JSON')
    subparsers.add_parser('features', help='Generate feature data from imported fixtures')
    subparsers.add_parser('train', help='Train the prediction model on generated features')

    predict_parser = subparsers.add_parser('predict', help='Predict the outcome of a fixture')
    predict_parser.add_argument('--home', help='Home team name')
    predict_parser.add_argument('--away', help='Away team name')
    predict_parser.add_argument('--div', help='Division code, e.g. E0')
    predict_parser.add_argument('--season', help='Season code, e.g. 2526')
    predict_parser.add_argument('--home-odds', type=float, default=None, help='Average home win odds')
    predict_parser.add_argument('--draw-odds', type=float, default=None, help='Average draw odds')
    predict_parser.add_argument('--away-odds', type=float, default=None, help='Average away win odds')
    predict_parser.add_argument('--fixtures-file', help='CSV of fixtures to predict in batch: '
                                                          'div,season,home_team,away_team[,home_odds,draw_odds,away_odds]')

    return parser


def run_predict(args):
    predictor = Predict(args.config, args.verbose)

    if args.fixtures_file:
        results = predictor.predict_fixtures_csv(args.fixtures_file)
        results.sort(key=lambda result: result.get('edge', float('-inf')), reverse=True)
        for result in results:
            print(format_prediction(result))
            print()
        return

    if not (args.home and args.away and args.div and args.season):
        print("predict requires either --fixtures-file, or all of --home/--away/--div/--season", file=sys.stderr)
        sys.exit(1)

    odds = None
    if args.home_odds and args.draw_odds and args.away_odds:
        odds = {
            'home_win_odds_avg': args.home_odds,
            'draw_odds_avg': args.draw_odds,
            'away_win_odds_avg': args.away_odds,
        }

    result = predictor.predict_fixture(args.div, args.season, args.home, args.away, odds=odds)
    print(format_prediction(result))


def main(argv):
    parser = build_parser()
    args = parser.parse_args(argv[1:])

    if args.command == 'import':
        ImportFixtures(args.config, args.verbose).run()
    elif args.command == 'features':
        CreateFeatures(args.config, args.verbose).run()
    elif args.command == 'train':
        TrainModel(args.config, args.verbose).run()
    elif args.command == 'predict':
        run_predict(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
