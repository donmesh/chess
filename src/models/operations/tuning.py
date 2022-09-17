import numpy as np

from models.base_model import BaseModel

def tune_models(models: list[BaseModel], X: np.ndarray, y: np.ndarray) -> list[BaseModel]:
    tuned_models_and_params = [model.tune_hyperparameters(X, y) for model in models]
    tuned_models = [model for model, _ in tuned_models_and_params if model is not None]
    return tuned_models