import pandas as pd

from models.base_model import BaseModel

def train_models(models: list[BaseModel], X: pd.DataFrame, y: pd.Series) -> None:
    for model in models:
        model.fit(X, y)