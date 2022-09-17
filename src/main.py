from configs.paths import TRAIN_DATA_PATH, TEST_DATA_PATH

from preprocessing.data_pipeline import process_data

from models.baseline_models.white_wins import WhiteWinsModel
from models.baseline_models.black_wins import BlackWinsModel
from models.baseline_models.draw import DrawModel
from models.baseline_models.random_winner import RandomWinnerModel

from models.advanced_models.random_forest import RandomForestModel

from models.operations.evaluation import evaluate_models
from models.operations.training import train_models
from models.operations.saving import save_models

metrics = ['accuracy', 'recall_macro', 'f1_macro']

baseline_models = [
    WhiteWinsModel(),
    BlackWinsModel(),
    DrawModel(),
    RandomWinnerModel(),
]

advanced_models = [
    RandomForestModel(n_epochs = 100, batch_size = 1024),
]

models = baseline_models + advanced_models

from datetime import datetime as dt

print(f'<<<<<< Processing data: {dt.now()}')
X, y = process_data(TRAIN_DATA_PATH)

print(f'<<<<<< Training models: {dt.now()}')
train_models(models, X, y)

print(f'<<<<<< Tuning hyperparameters: {dt.now()}')
tuned_models_and_params = [model.tune_hyperparameters(X, y) for model in models]
tuned_models = [model for model, _ in tuned_models_and_params if model is not None]
models += tuned_models

print(f'<<<<<< Saving models: {dt.now()}')
save_models(models)

print(f'<<<<<< Evaluating models: {dt.now()}')
evaluations = evaluate_models(models, X, y, metrics)

print(evaluations)
evaluations.to_excel('./evaluations.xlsx')