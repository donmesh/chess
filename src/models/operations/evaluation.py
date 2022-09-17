import numpy as np
import pandas as pd

from models.base_model import BaseModel

def evaluate_models(models: list[BaseModel], X: np.ndarray, y: np.ndarray, metrics: list[str]) -> pd.DataFrame:
    results = [model.evaluate(X, y, metrics) for model in models]
    df = pd.concat(results, axis=0)
    return df