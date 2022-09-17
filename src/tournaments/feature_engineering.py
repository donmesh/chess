import numpy as np
import pandas as pd
from datetime import timedelta

from utils.name_mapping import map_name_to_players, get_name_mapping
from configs.constants import BLACK_WIN, DRAW, WHITE_WIN

def _get_number_of_unique_players(df: pd.DataFrame) -> pd.DataFrame:
    number = df.reindex(columns=['white','black']).stack().unique().shape[0]
    return number

def _add_number_of_players(df: pd.DataFrame) -> pd.DataFrame:
    if 'num_players' in df.columns:
        df = df.drop(columns=['num_players'])
    df = df.merge(df.groupby(['tournament_id', 'tour_id']).apply(_get_number_of_unique_players).rename('num_players'), 
             on=['tournament_id', 'tour_id'],
             how='outer'
    )
    return df

def _convert_time_control(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df['time_control'] == 'classic', 'is_classic'] = 1
    df.loc[df['time_control'] == 'rapid', 'is_classic'] = 0
    df['is_classic'] = df['is_classic'].astype(int)
    df = df.drop(columns=['time_control'])
    return df

def _get_tournament_type(df: pd.DataFrame) -> str:
    players = df.reindex(columns=['tour_id', 'num_players']).drop_duplicates()

    value = np.nan
    if players['num_players'].unique().shape[0] == 1:
        starting_number = players[players['tour_id'] == 1]['num_players'].iloc[0]
        if starting_number * (starting_number - 1) / 2 == players.shape[0]:
            value = 'round_robin'
        else:
            value = 'swiss'
    elif (players['num_players'].diff(1).fillna(-np.inf) < 0).all():
        value = 'knockout'
    return value

def _get_all_tournament_types(df: pd.DataFrame) -> pd.DataFrame:
    res = df.groupby('tournament_id').apply(_get_tournament_type).rename('tournament_type')
    df = df.merge(res, on='tournament_id', how='outer')
    return df

def _one_hot_tournament_type(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df['tournament_type'] == 'knockout', 'is_knockout'] = 1
    df.loc[df['tournament_type'] == 'swiss', 'is_swiss'] = 1
    df.loc[df['tournament_type'] == 'round_robin', 'is_round_robin'] = 1
    df[['is_knockout', 'is_swiss', 'is_round_robin']] = df[['is_knockout', 'is_swiss', 'is_round_robin']].fillna(0).astype(int)
    df = df.drop(columns=['tournament_type'])
    return df

def _add_tournament_type(df: pd.DataFrame) -> pd.DataFrame:
    df = _get_all_tournament_types(df)
    df = _one_hot_tournament_type(df)
    return df

def _get_tournament_duration(df: pd.DataFrame) -> int:
    duration = (df['end_date'] - df['start_date'] + timedelta(days=1)).dt.days
    return duration

def _add_tournament_duration(df: pd.DataFrame) -> pd.DataFrame:
    duration = df.groupby('tournament_id').apply(_get_tournament_duration).reset_index(drop=True).rename('tournament_duration_days')
    df = df.merge(duration, left_index=True, right_index=True, how='outer')
    return df

def _get_days_since_tournament_start(df: pd.DataFrame) -> pd.DataFrame:
    days = (df['date'] - df['start_date']).dt.days
    return days

def _add_days_since_tournament_start(df: pd.DataFrame) -> pd.DataFrame:
    days = df.groupby('tournament_id').apply(_get_days_since_tournament_start).reset_index(drop=True).rename('days_since_start')
    df = df.merge(days, left_index=True, right_index=True, how='outer')
    return df

def add_scores(df: pd.DataFrame, df_scores: pd.DataFrame) -> pd.DataFrame:
    df_scores = df_scores
    df = df.merge(df_scores.rename(columns={'score_2014': 'white_score_2014', 'score_2020': 'white_score_2020'}),
                   left_on='white',
                   right_index=True,
                   how='left')
    df = df.merge(df_scores.rename(columns={'score_2014': 'black_score_2014', 'score_2020': 'black_score_2020'}),
                   left_on='black',
                   right_index=True,
                   how='left')
    df = df.sort_index()
    score_columns = ['white_score_2014', 'white_score_2020', 'black_score_2014', 'black_score_2020']
    df[score_columns] = df[score_columns].fillna(0).astype(int)
    return df

def _convert_tour_as_completion_percentage(df: pd.DataFrame) -> pd.DataFrame:
    df['tour_completion'] = df['tour_id'] / df['num_tours']
    df = df.drop(columns=['tour_id', 'num_tours'])
    return df

def _convert_tournament_as_completion_percentage(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_tournament_duration(df)
    df = _add_days_since_tournament_start(df)
    df['tournament_completion'] = df['days_since_start'] / df['tournament_duration_days']
    df = df.drop(columns=['days_since_start', 'tournament_duration_days'])
    return df

def _map_result(df: pd.DataFrame) -> pd.DataFrame:
    result_map = {0.0: BLACK_WIN, 0.5: DRAW, 1.0: WHITE_WIN}
    df['result'] = df['result'].map(result_map)
    return df

def _map_names(df: pd.DataFrame, name_mapping: dict[str, str]) -> pd.DataFrame:
    df['white'] = df['white'].map(map_name_to_players(df['white'].tolist(), name_mapping).set_index('orig_name')['mapped'].to_dict())
    df['black'] = df['black'].map(map_name_to_players(df['black'].tolist(), name_mapping).set_index('orig_name')['mapped'].to_dict())
    return df

def create_features_tournament(df_orig: pd.DataFrame) -> pd.DataFrame:
    df = df_orig.copy()
    df = _add_number_of_players(df)
    df = _convert_time_control(df)
    df = _add_tournament_type(df)
    df = _convert_tour_as_completion_percentage(df)
    df = _convert_tournament_as_completion_percentage(df)
    name_mapping = get_name_mapping()
    df = _map_names(df, name_mapping)
    df = _map_result(df)
    return df