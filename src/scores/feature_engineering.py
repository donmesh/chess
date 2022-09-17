import pandas as pd
from utils.name_mapping import map_name_to_players, get_name_mapping

def _map_names(df: pd.DataFrame, name_mapping: dict[str, str]) -> pd.DataFrame:
    df.index = map_name_to_players(df.index.tolist(), name_mapping)['mapped']
    return df

def create_features_scores(df_orig: pd.DataFrame) -> pd.DataFrame:
    df = df_orig.copy()
    name_mapping = get_name_mapping()
    df = _map_names(df, name_mapping)
    return df