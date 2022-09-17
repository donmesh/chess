import pickle
import pinyin
import unicodedata
import pandas as pd
from itertools import permutations

from configs.paths import MODEL_DIR

def is_english(text: str) -> bool:
    return text.isascii()

def translate_chinese(text: str) -> str:
    translated = pinyin.get(text, format="strip", delimiter=" ")
    return translated

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def _get_clean_name(df: pd.DataFrame) -> pd.DataFrame:
    df['clean_name'] = df['orig_name'].str.strip().str.lower()
    df['clean_name'] = df['clean_name'].map(strip_accents)
    df['is_english'] = df['clean_name'].map(is_english)
    df.loc[~df['is_english'], 'clean_name'] = df['clean_name'].map(translate_chinese)
    df['clean_name'] = df['clean_name'].str.replace(',', '')
    df = df.drop(columns=['is_english'])
    return df

def _get_clean_name_permutations(names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(names, columns=['orig_name'])
    df = _get_clean_name(df)
    df['clean_name_list'] = df['clean_name'].str.split(' ')
    df['clean_name_list_permutations'] = df['clean_name_list'].map(permutations).map(list)
    df['clean_name_permutations'] = df['clean_name_list_permutations'].map(lambda x: [''.join(name) for name in x])
    df = df.reindex(columns=['orig_name', 'clean_name_permutations'])
    return df

def _get_permutation_index_map(df: pd.DataFrame) -> dict[str, int]:
    permutation_index_map = df.reset_index().set_index('clean_name_permutations')['index'].to_dict()
    return permutation_index_map

def _rename_name_mapping(name_mapping: dict[str, int]) -> dict[str, int]:
    rename_dict = {number: f'player_{i}' for i, number in enumerate(sorted(set(name_mapping.values())))}
    renamed_name_mapping = {name: rename_dict[idx] for name, idx in name_mapping.items()}
    return renamed_name_mapping

def _add_index_to_connect_permutations(df: pd.DataFrame, permutation_index_map: dict[str, int]) -> pd.DataFrame:
    df['index'] = df['clean_name_permutations'].map(permutation_index_map)
    return df

def _convert_df_to_name_mapping(df: pd.DataFrame) -> dict[str, int]:
    name_mapping = df.set_index('clean_name_permutations')['index'].to_dict()
    return name_mapping

def create_name_mapping(names: list[str]) -> dict[str, str]:
    df_permutations = _get_clean_name_permutations(names)
    df_permutations_exploded = df_permutations.explode('clean_name_permutations').drop_duplicates()
    permutation_index_map = _get_permutation_index_map(df_permutations_exploded)
    df_names_grouped = df_permutations_exploded.groupby(['clean_name_permutations']).agg(tuple).reset_index()
    df_names_grouped = _add_index_to_connect_permutations(df_names_grouped, permutation_index_map)
    df_names_exploded = df_names_grouped.explode('orig_name')
    name_mapping = _convert_df_to_name_mapping(df_names_exploded)
    renamed_name_mapping = _rename_name_mapping(name_mapping)
    return renamed_name_mapping

def get_name_mapping() -> dict[str, str]:
    with open(f'{MODEL_DIR}/name_mapping.pickle', 'rb') as f:
        name_mapping = pickle.load(f)
        return name_mapping

def map_name_to_players(names: list[str], name_mapping: dict[str, str]) -> dict[str, str]:
    df = pd.Series(names).rename('orig_name').to_frame()
    df_permutations = _get_clean_name_permutations(df)
    df_exploded_permutations = df_permutations.explode('clean_name_permutations')
    df_exploded_permutations['mapped'] = df_exploded_permutations['clean_name_permutations'].map(name_mapping)
    df_map = df_exploded_permutations.reindex(columns=['orig_name', 'mapped']).drop_duplicates()
    return df_map