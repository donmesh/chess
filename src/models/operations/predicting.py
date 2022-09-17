import numpy as np
import pandas as pd

from models.base_model import BaseModel
from postprocessing.data_pipeline import process_data

def predict_models(models: list[BaseModel], X: np.ndarray) -> pd.DataFrame:
    results = [process_data(model.predict(X).rename(model.name)) for model in models]
    df = pd.concat(results, axis=1) if len(results) > 1 else results[0].to_frame()
    return df