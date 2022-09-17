import pandas as pd
import matplotlib.pyplot as plt

def plot_scores(df_scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1,2, figsize=[15,5], sharey=True)
    df_scores.hist(ax=axes)

    for ax in axes:
        ax.set_title(ax.get_title(), fontsize=15)
        ax.set_xlabel('score', fontsize=15)
        ax.set_ylabel('count', fontsize=15)