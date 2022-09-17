import pickle
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

from configs.constants import CROSS_VALIDATION_SPLITS, RANDOM_SEED, TEST_SAMPLE_SIZE
from models.utils import format_evaluation_scores

class BaseModel(BaseEstimator):
    def __init__(self) -> None:
        self.name = self.__class__.__name__
        self.sampler = StratifiedShuffleSplit(n_splits=CROSS_VALIDATION_SPLITS, test_size=TEST_SAMPLE_SIZE, random_state=RANDOM_SEED)
        self.model = DummyClassifier()

    def __str__(self) -> str:
        return self.name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        return self.model.fit(X.values, y.values)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        y_hat = self.model.predict(X.values)
        y_hat = pd.Series(y_hat)
        return y_hat

    def evaluate(self, X: np.ndarray, y: np.ndarray, metrics: list[str]) -> pd.DataFrame:
        scores = cross_validate(self, X, y, cv=self.sampler, scoring=metrics, return_train_score=True)
        df_scores = format_evaluation_scores(self.name, scores)
        return df_scores

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> tuple[None, None]:
        return None, None

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        path_hyperparams = path.replace('.pickle','_hyperparams.pickle')
        with open(path_hyperparams, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            return pickle.load(f)

