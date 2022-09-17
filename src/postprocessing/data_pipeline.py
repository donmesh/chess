import pandas as pd

from configs.constants import BLACK_WIN, DRAW, WHITE_WIN

def _map_result(series: pd.Series) -> pd.Series:
    result_map = {BLACK_WIN: 0.0, DRAW: 0.5, WHITE_WIN: 1.0}
    series = series.map(result_map)
    return series

def process_data(series_orig: pd.Series) -> pd.Series:
    series = series_orig.copy()
    series = _map_result(series)
    return series