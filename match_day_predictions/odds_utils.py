import statistics
import sys

HOME_ODDS_COLUMNS = ['B365H', 'BFH', 'BSH', 'BWH', 'GBH', 'IWH', 'LBH', 'PSH', 'PH', 'SOH', 'SBH', 'SJH', 'SYH', 'VCH', 'WHH']
DRAW_ODDS_COLUMNS = ['B365D', 'BFD', 'BSD', 'BWD', 'GBD', 'IWD', 'LBD', 'PSD', 'PD', 'SOD', 'SBD', 'SJD', 'SYD', 'VCD', 'WHD']
AWAY_ODDS_COLUMNS = ['B365A', 'BFA', 'BSA', 'BWA', 'GBA', 'IWA', 'LBA', 'PSA', 'PA', 'SOA', 'SBA', 'SJA', 'SYA', 'VCA', 'WHA']
OVER_25_COLUMNS = ['GB>2.5', 'B365>2.5', 'P>2.5', 'Max>2.5', 'Min>2.5']
UNDER_25_COLUMNS = ['GB<2.5', 'B365<2.5', 'P<2.5', 'Max<2.5', 'Min<2.5']


def get_element(header, row, element_name):
    """Looks up a named column in a CSV header/row pair, returning '' if missing."""
    try:
        if element_name == 'Div':
            return row[0]
        index = header.index(element_name)
        return row[index].strip()
    except ValueError:
        return ''


def get_element_float(header, row, element_name):
    """Looks up a named column and parses it as a float, defaulting to 0.0."""
    try:
        if element_name == 'Div':
            return row[0]
        index = header.index(element_name)
        try:
            return float(row[index].strip())
        except ValueError:
            return 0.0
    except ValueError:
        return 0.0


def convert_date(header, row, header_key):
    """Converts a football-data.co.uk DD/MM/YY(YY) date into ISO YYYY-MM-DD."""
    date_str = get_element(header, row, header_key)
    try:
        day, month, year = date_str.split('/')
        if len(year) == 2:
            year = f"20{year}"
        return f"{year}-{month}-{day}"
    except (ValueError, IndexError):
        print(f"Error tokenizing {date_str}")
        print(header)
        print(row)
        sys.exit(1)


def get_odds_stats(header, row, column_names):
    """Returns (min, max, avg, range, std_dev) across whichever of the given
    bookmaker columns are present for this row.
    """
    values = []
    for column_name in column_names:
        value = get_element(header, row, column_name)
        if value is not None and value != '':
            values.append(float(value))

    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0], 0.0, 0.0

    min_odds = min(values)
    max_odds = max(values)
    avg = statistics.mean(values)
    range_val = max_odds - min_odds
    std_dev = statistics.stdev(values)
    return min_odds, max_odds, avg, range_val, std_dev


def get_match_odds_stats(header, row):
    """Convenience wrapper returning all five odds-stat tuples (home, draw,
    away, over 2.5, under 2.5) for a single fixture row.
    """
    home = get_odds_stats(header, row, HOME_ODDS_COLUMNS)
    draw = get_odds_stats(header, row, DRAW_ODDS_COLUMNS)
    away = get_odds_stats(header, row, AWAY_ODDS_COLUMNS)
    over_25 = get_odds_stats(header, row, OVER_25_COLUMNS)
    under_25 = get_odds_stats(header, row, UNDER_25_COLUMNS)
    return home, draw, away, over_25, under_25
