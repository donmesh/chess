import pandas as pd

from configs.paths import SCORES_2014, SCORES_2020

from tournaments.data_cleaning import get_all_tournament_data
from tournaments.feature_engineering import create_features_tournament, add_scores

from scores.data_cleaning import get_combined_scores
from scores.feature_engineering import create_features_scores

def _keep_features_and_dependent_columns(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        'white', 'black',
        'is_classic', 'is_knockout',
        'tour_completion', 'tournament_completion',
        'white_score_2014', 'white_score_2020',
        'black_score_2014', 'black_score_2020'
    ]
    dependent_column = 'result'
    columns = feature_columns + [dependent_column]
    df_filtered = df.reindex(columns=columns)
    return df_filtered

def process_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df_scores_raw = get_combined_scores(SCORES_2014, SCORES_2020)
    df_tournament_raw = get_all_tournament_data(data_path, True)

    df_scores = create_features_scores(df_scores_raw)
    df_tournament = create_features_tournament(df_tournament_raw)
    df = add_scores(df_tournament, df_scores)

    df_features = _keep_features_and_dependent_columns(df)

    df_dummies = pd.get_dummies(df_features)
    X = df_dummies.drop(columns=['result'])
    y = df_dummies['result']
    return X, y