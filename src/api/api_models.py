from enum import Enum
from typing import Optional
from pydantic import BaseModel

from configs.paths import OUTPUT_DIR

class MaxFeatures(Enum):
    sqrt: str = 'sqrt'
    log2: str = 'log2'

class Metric(Enum):
    accuracy: str = 'accuracy'
    recall_macro: str = 'recall_macro'
    f1_macro: str = 'f1_macro'

class Model(Enum):
    WhiteWinsModel: str = 'WhiteWinsModel'
    BlackWinsModel: str = 'BlackWinsModel'
    DrawModel: str = 'DrawModel'
    RandomWinnerModel: str = 'RandomWinnerModel'
    RandomForestModel: str = 'RandomForestModel'
    RandomForestModel_tuned: str = 'RandomForestModel_tuned'

class Params(BaseModel):
    n_epochs: int = 100
    batch_size: int = 1024
    max_features: Optional[MaxFeatures] = 'sqrt'
    max_depth: Optional[int]
    min_samples_split: Optional[int] = 2
    min_samples_leaf: Optional[int] = 1

class TrainModelsRequest(BaseModel):
    data_path: str
    models: list[Model]
    params: Params

class TuneModelsRequest(BaseModel):
    data_path: str
    models: list[Model]

class EvaluateModelsRequest(BaseModel):
    data_path: str
    models: list[Model]
    metrics: list[Metric]
    output_file: Optional[str] = f'{OUTPUT_DIR}/evaluations.xlsx'

class PredictModelsRequest(BaseModel):
    data_path: str
    models: list[Model]
    save_to_files: bool