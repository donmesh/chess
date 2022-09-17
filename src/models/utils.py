import numpy as np
import pandas as pd

def format_evaluation_scores(model_name: str, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(scores).T.reset_index()
    df[['score_type','metric']] = df['index'].str.split('_', 1, expand=True)
    df.loc[~df['index'].str.contains('train') & ~df['index'].str.contains('test'), 'score_type'] = ''
    df.loc[~df['index'].str.contains('train') & ~df['index'].str.contains('test'), 'metric'] = df['index']
    df['model'] = model_name
    df = df.drop(columns=['index']).set_index(['model', 'score_type', 'metric']).sort_index()
    df.columns = [f'fold_{i}' for i in df.columns]
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    return df