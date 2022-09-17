from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import Optional
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from models.base_model import BaseModel
from configs.constants import RANDOM_SEED, TUNE_BATCH_SPLIT, TUNE_BATCH_N_ITERATIONS, TUNE_N_ITERATIONS

class Criterion(Enum):
    GINI = 'gini'
    ENTROPY = 'entropy'
    LOG_LOSS = 'log_loss'

class MaxFeatures(Enum):
    SQRT = 'sqrt'
    LOG2 = 'log2'

class ClassWeights(Enum):
    BALANCED = 'balanced'
    BALANCED_SUBSAMPLE = 'balanced_subsample'

class RandomForestModel(BaseModel, ClassifierMixin):
    def __init__(self,
            n_epochs: int,
            batch_size: int,
            bootstrap: Optional[bool] = True,
            criterion: Optional[Criterion] = 'gini',
            max_depth: Optional[int] = None,
            max_features: Optional[MaxFeatures | int | float | None] = 'sqrt',
            max_leaf_nodes: Optional[int] = None,
            min_impurity_decrease: Optional[float] = 0.0,
            min_samples_leaf: Optional[int | float] = 1,
            min_samples_split: Optional[int | float] = 2,
            min_weight_fraction_leaf: Optional[float] = 0.0,
            n_estimators: Optional[int] = 1,
            n_jobs: Optional[int] | None = None,
            oob_score: Optional[bool] = False,
            random_state: Optional[int | None] = RANDOM_SEED,
            verbose: Optional[int] = 0,
            warm_start: Optional[bool] = True,
            class_weight: Optional[ClassWeights | dict | list[dict] | None] = None,
            ccp_alpha: Optional[float] = 0.0,
            max_samples: Optional[int | float] = None) -> None:
        super().__init__()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

        self.model = RandomForestClassifier(**{k: v for k,v in self.get_params().items() if k not in ['n_epochs', 'batch_size']}
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_one_hot = pd.get_dummies(X)
        self.feature_names = X_one_hot.columns.tolist()
        data = np.concatenate([X_one_hot.values, y.values.reshape([-1,1])], axis=1)
        n_batches = int(np.ceil(data.shape[0] / self.batch_size))
        for _ in tqdm(range(self.n_epochs)):
            for data_batch in np.array_split(data, n_batches):
                X_batch, y_batch = data_batch[:, :-1], data_batch[:, -1]
                self.model.fit(X_batch, y_batch)
                self.model.n_estimators += 1

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_one_hot = pd.get_dummies(X).reindex(columns=self.feature_names).fillna(0)
        y_hat = self.model.predict(X_one_hot.values)
        y_hat = pd.Series(y_hat)
        return y_hat

    def _get_initial_search_space(self) -> dict[str, str]:
        max_features = ['log2', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] + [None]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        search_space = {
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}
        return search_space

    def _batch_tune(self, X: np.ndarray, y: np.ndarray) -> tuple[RandomForestModel, dict[str, str | int]]:
        search_space_initial = self._get_initial_search_space()
        params = []
        data = np.concatenate([X.values, y.values.reshape([-1,1])], axis=1)
        for data_batch in np.array_split(data, TUNE_BATCH_SPLIT):
            tuning_initial = RandomizedSearchCV(estimator = self,
                                            param_distributions = search_space_initial,
                                            n_iter = TUNE_BATCH_N_ITERATIONS,
                                            cv = self.sampler,
                                            verbose = 0,
                                            random_state = RANDOM_SEED,
                                            n_jobs = -1)
            X_batch, y_batch = pd.DataFrame(data_batch[:, :-1]), pd.Series(data_batch[:, -1])
            tuning_initial.fit(X_batch, y_batch)
            params.append(tuning_initial.best_params_)
        return params

    def _get_final_search_space(self, params: list[dict[str, str]]):
        search_space = pd.DataFrame(params).T.apply(set, axis=1).map(list).to_dict()
        return search_space

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> tuple[RandomForestModel, dict[str, str | int]]:
        print(X.shape, y.shape)
        params = self._batch_tune(X, y)
        print(params)
        search_space_final = self._get_final_search_space(params)
        tuning = RandomizedSearchCV(estimator = self,
                                    param_distributions = search_space_final,
                                    n_iter = TUNE_N_ITERATIONS,
                                    cv = self.sampler,
                                    verbose = 2,
                                    random_state = RANDOM_SEED,
                                    n_jobs = -1)

        tuning.fit(X, y)
        best_estimator = tuning.best_estimator_
        best_estimator.name = best_estimator.name + '_tuned'
        return best_estimator, tuning.best_params_