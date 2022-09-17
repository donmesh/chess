import os
import json
import pandas as pd
from datetime import datetime as dt

from configs.paths import MODEL_DIR, OUTPUT_DIR
from models.base_model import BaseModel
from tournaments.data_cleaning import get_all_tournament_data

def save_models(models: list[BaseModel]) -> None:
    for model in models:
        timestamp = dt.now().strftime('%Y-%m-%d_%H%M%S')
        path = f'{MODEL_DIR}/{timestamp}'
        os.makedirs(path, exist_ok=True)
        model.save(f'{path}/{model.name}.pickle')

def save_evaluations_to_excel(evaluations: pd.DataFrame, path: str) -> None:
    evaluations.to_excel(path.rsplit('.',1)[0] + '.xlsx')

def _group_games_groupby(df: pd.DataFrame) -> dict:
    return df.drop(columns=['name','index']).to_dict(orient='records')

def _group_games(df: pd.DataFrame) -> pd.DataFrame:
    df_games = df.reindex(columns=['index','name','white','black','date','id','result']) \
                 .groupby(['name','index']) \
                 .apply(_group_games_groupby) \
                 .rename('games') \
                 .reset_index()
    return df_games

def _assign_games(df: pd.DataFrame, df_games: pd.DataFrame) -> pd.DataFrame:
    df_merged = df.drop(columns=['games','white','black','date','id','result']).drop_duplicates().merge(df_games, on=['name','index'])
    df_merged['temp1'] = df_merged['name'].str.rsplit('_',1).str[1].astype(int)
    df_merged['temp2'] = df_merged['index'].str.rsplit('_',1).str[1].astype(int)
    df_merged = df_merged.sort_values(['temp1','temp2']).reset_index(drop=True).drop(columns=['temp1','temp2'])
    return df_merged

def _assign_tours_to_games_groupby(df: pd.DataFrame) -> dict:
    return df.drop(columns=['name']).set_index('index')['games'].to_dict()

def _clean_final_df(df: pd.DataFrame):
    df_final = df.groupby(['name','start_date','end_date','tours','time_control']) \
                 .apply(_assign_tours_to_games_groupby) \
                 .rename('games') \
                 .reset_index() \
                 .reindex(columns=['name','start_date','end_date','games','tours','time_control'])
    return df_final

def _prepare_output_df(df: pd.DataFrame) -> pd.DataFrame:
    df_games = _group_games(df)
    df_merged = _assign_games(df, df_games)
    df_final = _clean_final_df(df_merged)
    return df_final

def _save_jsons(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for _, row in df.iterrows():
        name = row['name']
        dc = row.to_dict()
        path = f'{OUTPUT_DIR}/{name}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dc, f, indent=4, ensure_ascii=False)

def save_predictions_to_jsons(predictions: pd.Series, data_path: str) -> None:
    df = get_all_tournament_data(data_path, False).reset_index(drop=True)
    df['result'] = predictions
    df = _prepare_output_df(df)
    _save_jsons(df)
