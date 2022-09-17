import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlogram(df: pd.DataFrame) -> None:
    sns.pairplot(df)
    plt.show()

def plot_winner_distribution(series: pd.Series) -> None:
    series.hist()
    plt.title('Winner distribution')
    plt.show()