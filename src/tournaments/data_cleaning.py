import os
import pandas as pd

def _extract_games_df(nested_games: pd.Series) -> pd.DataFrame:
    df_games_raw = nested_games.explode().reset_index()
    df_games = pd.concat([
        df_games_raw['index'],
        pd.DataFrame(df_games_raw['games'].tolist())
    ], axis=1)
    return df_games

def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'index':'tour_id',
        'name': 'tournament_id',
        'tours': 'num_tours',
        'id': 'game_id'
    })
    return df

def _reindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=[
        'tournament_id', 'tour_id', 'game_id',
        'start_date', 'end_date', 'date',
        'time_control',
        'white', 'black', 'result',
        'num_tours'
    ])
    return df

def _clean_ids(df: pd.DataFrame) -> pd.DataFrame:
    df['tournament_id'] = df['tournament_id'].str.strip('tournament_').str.strip('test_').astype(int)
    df['tour_id'] = df['tour_id'].str.strip('tour_').astype(int)
    df['game_id'] = df['game_id'].str.strip('tournament_').str.rsplit('_', 1).str[1].astype(int)
    return df

def _convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    df['start_date'] = df['start_date'].map(pd.to_datetime)
    df['end_date'] = df['end_date'].map(pd.to_datetime)
    df['date'] = df['date'].map(pd.to_datetime)
    return df

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_columns(df)
    df = _reindex_columns(df)
    df = _clean_ids(df)
    df = _convert_dates(df)
    return df

def _get_tournament_data(path: str, clean_data: bool) -> pd.DataFrame:
    df_tournament = pd.read_json(path)
    df_games = _extract_games_df(df_tournament['games'])
    df = df_tournament.reset_index().merge(df_games, on='index')
    if clean_data:
        df = _clean_data(df)
    return df

def _sort_values(df: pd.DataFrame) -> pd.DataFrame:
    columns = ['tournament_id', 'tour_id', 'game_id', 'start_date', 'end_date', 'date', 'white', 'black']
    df = df.sort_values(columns).reset_index(drop=True)
    return df

def get_all_tournament_data(data_path: str, clean_data: bool) -> pd.DataFrame:
    data = []
    for file in os.listdir(data_path):
        df_tournament = _get_tournament_data(f'{data_path}/{file}', clean_data)
        data.append(df_tournament)
    df = pd.concat(data, axis=0) if len(data) > 1 else data
    if clean_data:
        df = _sort_values(df)
    return df