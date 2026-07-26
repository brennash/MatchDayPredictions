import csv
import json
import os
import sys

import yaml
from dataclasses import asdict

from .fixture_data import FixtureData
from . import odds_utils


class ImportFixtures:
    """Converts the raw football-data.co.uk CSVs under fixtures_raw_folder into
    one JSON file per season/division under fixtures_folder, each a list of
    FixtureData records.
    """

    def __init__(self, config_filename, verbose_flag=False):
        self.verbose = verbose_flag
        self.config = None
        if config_filename is not None:
            with open(config_filename, 'r') as config_file:
                self.config = yaml.safe_load(config_file)

    def run(self):
        fixture_list = self.get_fixture_list()

        if self.verbose:
            print(f"Starting to process {len(fixture_list)} fixture files..")

        fixtures_folder = self.config['fixtures']['fixtures_folder']
        fixture_count = 0
        for fixture_file in fixture_list:
            fixture_data_list = self.get_csv_as_fixture_data_list(fixture_file)
            if len(fixture_data_list) == 0:
                continue

            if self.verbose:
                print(f"Processing file {fixture_file} ({len(fixture_data_list)} fixtures).")

            season_value = fixture_data_list[0].season
            div_value = fixture_data_list[0].div
            output_json_file = f"{fixtures_folder}/{season_value}_{div_value}.json"
            with open(output_json_file, "w") as output_file:
                json.dump([asdict(fixture_data) for fixture_data in fixture_data_list], output_file)

            fixture_count += len(fixture_data_list)

        if self.verbose:
            print(f'{fixture_count} fixtures processed and written to {fixtures_folder}')

    def get_csv_as_fixture_data_list(self, fixture_file):
        """Loads a single raw CSV into a list of FixtureData objects. Also
        used directly by tests to avoid needing a config file.
        """
        fixture_data_list = []
        with open(fixture_file, newline='', encoding='ISO-8859-1') as csvfile:
            fixture_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            header = None
            for row_num, row in enumerate(fixture_reader):
                if row_num == 0:
                    header = row
                elif len(row) > 0 and row[0] not in (None, ''):
                    fixture_row = self.get_row_as_list(fixture_file, header, row)
                    if fixture_row is not None:
                        fixture_data_list.append(FixtureData(*fixture_row))
        return fixture_data_list

    def get_row_as_list(self, fixture_file, header, row):
        season = fixture_file.split('/')[-2]
        div = odds_utils.get_element(header, row, 'Div')
        date = odds_utils.convert_date(header, row, 'Date')
        time = odds_utils.get_element(header, row, 'Time')
        home_team = odds_utils.get_element(header, row, 'HomeTeam')
        away_team = odds_utils.get_element(header, row, 'AwayTeam')

        if home_team in (None, '') or away_team in (None, ''):
            return None

        home_ht = odds_utils.get_element_float(header, row, 'HTHG')
        home_ft = odds_utils.get_element_float(header, row, 'FTHG')
        home_shots = odds_utils.get_element_float(header, row, 'HS')
        home_shots_on_target = odds_utils.get_element_float(header, row, 'HST')
        home_corners = odds_utils.get_element_float(header, row, 'HC')
        home_fouls_committed = odds_utils.get_element_float(header, row, 'HF')
        home_free_kicks = odds_utils.get_element_float(header, row, 'HFKC')
        home_offsides = odds_utils.get_element_float(header, row, 'HO')
        home_yellow_cards = odds_utils.get_element_float(header, row, 'HY')
        home_red_cards = odds_utils.get_element_float(header, row, 'HR')
        away_ht = odds_utils.get_element_float(header, row, 'HTAG')
        away_ft = odds_utils.get_element_float(header, row, 'FTAG')
        away_shots = odds_utils.get_element_float(header, row, 'AS')
        away_shots_on_target = odds_utils.get_element_float(header, row, 'AST')
        away_corners = odds_utils.get_element_float(header, row, 'AC')
        away_fouls_committed = odds_utils.get_element_float(header, row, 'AF')
        away_free_kicks = odds_utils.get_element_float(header, row, 'AFKC')
        away_offsides = odds_utils.get_element_float(header, row, 'AO')
        away_yellow_cards = odds_utils.get_element_float(header, row, 'AY')
        away_red_cards = odds_utils.get_element_float(header, row, 'AR')
        result = odds_utils.get_element(header, row, 'FTR')

        home_odds, draw_odds, away_odds, over_25, under_25 = odds_utils.get_match_odds_stats(header, row)

        return [
            season, div, date, time, home_team, away_team,
            home_ht, home_ft, home_shots, home_shots_on_target, home_corners,
            home_fouls_committed, home_free_kicks, home_offsides, home_yellow_cards, home_red_cards,
            away_ht, away_ft, away_shots, away_shots_on_target, away_corners,
            away_fouls_committed, away_free_kicks, away_offsides, away_yellow_cards, away_red_cards,
            *home_odds, *draw_odds, *away_odds, *over_25, *under_25,
            result,
        ]

    def get_fixture_list(self):
        fixtures_folder = self.config['fixtures_raw']['fixtures_raw_folder']
        fixture_files = []
        for root, dirs, files in os.walk(fixtures_folder):
            for file in files:
                if file.lower().endswith('.csv'):
                    fixture_file_path = os.path.join(root, file)
                    fixture_files.append(fixture_file_path)
                    if self.verbose:
                        print(f'Adding fixture file - {fixture_file_path}')
        return fixture_files
