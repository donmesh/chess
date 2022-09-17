import pandas as pd

def read_scores(path: str) -> pd.Series:
    df = pd.read_csv(path, sep='\t', header=None) \
           .rename(columns={0:'name', 1:'score'})
    df = df.sort_values('score', ascending=False)
    df = df.reset_index(drop=True)
    df = df.groupby('name')['score'].max()
    return df

def get_combined_scores(path_2014: str, path_2020: str) -> pd.DataFrame:
    scores_2014 = read_scores(path_2014)
    scores_2020 = read_scores(path_2020)

    df_scores = pd.concat([
        scores_2014.rename('score_2014'),
        scores_2020.rename('score_2020')
    ], axis=1)
    return df_scores
