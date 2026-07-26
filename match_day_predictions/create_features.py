import json
import os
from dataclasses import asdict

import yaml

from .fixture_data import FixtureData
from .league import build_league_from_fixtures, DEFAULT_MIN_GAMES_PLAYED


class CreateFeatures:
    """Reads each season/division's fixtures JSON (produced by
    ImportFixtures) and replays it through a League to produce one
    feature-row-per-fixture JSON file per season/division.
    """

    def __init__(self, config_filename, verbose_flag=False):
        self.verbose = verbose_flag
        with open(config_filename, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def run(self):
        fixture_file_list = self.get_fixture_file_list()

        if self.verbose:
            print(f'Processing {len(fixture_file_list)} season/division fixture files')

        min_games_played = self.config.get('league', {}).get('min_games_played', DEFAULT_MIN_GAMES_PLAYED)
        features_folder = self.config['features']['features_folder']

        total_features = 0
        for fixture_filename in fixture_file_list:
            fixture_list = self.deserialize_fixtures(fixture_filename)
            if len(fixture_list) == 0:
                continue

            season_id = fixture_list[0].season
            div_id = fixture_list[0].div

            if self.verbose:
                print(f"Processing {div_id}/{season_id} - {len(fixture_list)} fixtures")

            _, features_list = build_league_from_fixtures(fixture_list, self.verbose, min_games_played)

            output_json_file = f"{features_folder}/{season_id}_{div_id}.json"
            with open(output_json_file, "w") as file:
                json.dump([asdict(feature) for feature in features_list], file, indent=4)
            total_features += len(features_list)

        if self.verbose:
            print(f'{total_features} feature rows written to {features_folder}')

    def get_fixture_file_list(self):
        """Returns the list of season/division fixture JSON files produced by
        ImportFixtures.
        """
        fixtures_folder = self.config['fixtures']['fixtures_folder']
        fixture_files = []
        for root, dirs, files in os.walk(fixtures_folder):
            for file in files:
                if file.lower().endswith('.json'):
                    fixture_files.append(os.path.join(root, file))
        return fixture_files

    def deserialize_fixtures(self, json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return [FixtureData(**fixture_dict) for fixture_dict in data]
