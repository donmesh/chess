import os
import json
import uvicorn
from fastapi import FastAPI, Body

from preprocessing.data_pipeline import process_data

from models.baseline_models.white_wins import WhiteWinsModel
from models.baseline_models.black_wins import BlackWinsModel
from models.baseline_models.draw import DrawModel
from models.baseline_models.random_winner import RandomWinnerModel

from models.advanced_models.random_forest import RandomForestModel

from models.operations.saving import save_models, save_evaluations_to_excel, save_predictions_to_jsons
from models.operations.loading import load_models
from models.operations.training import train_models
from models.operations.tuning import tune_models
from models.operations.evaluation import evaluate_models
from models.operations.predicting import predict_models

from api.api_models import TrainModelsRequest, TuneModelsRequest, \
                            EvaluateModelsRequest, PredictModelsRequest

app = FastAPI()

@app.post('/train-models')
def train_models_endpoint(request: TrainModelsRequest = Body(...)):
    request_data = json.loads(request.json())
    baseline_models = []
    advanced_models = []
    if 'WhiteWinsModel' in request_data['models']:
        baseline_models.append(WhiteWinsModel())
    if 'BlackWinsModel' in request_data['models']:
        baseline_models.append(BlackWinsModel())
    if 'DrawModel' in request_data['models']:
        baseline_models.append(DrawModel())
    if 'RandomWinnerModel' in request_data['models']:
        baseline_models.append(RandomWinnerModel())

    if 'RandomForestModel' in request_data['models']:
        baseline_models.append(RandomForestModel(**request_data['params']))

    models = baseline_models + advanced_models

    X, y = process_data(request.data_path)
    train_models(models, X, y)
    save_models(models)

@app.post('/tune-models')
def tune_models_endpoint(request: TuneModelsRequest = Body(...)):
    request_data = json.loads(request.json())
    models = load_models(request_data['models'])
    X, y = process_data(request_data['data_path'])
    tuned_models = tune_models(models, X, y)
    save_models(tuned_models)

@app.post('/evaluate-models', response_model = list[dict])
def evaluate_models_endpoint(request: EvaluateModelsRequest = Body(...)):
    request_data = json.loads(request.json())
    models = load_models(request_data['models'])
    X, y = process_data(request_data['data_path'])
    evaluations = evaluate_models(models, X, y, request_data['metrics'])
    if request_data['output_file'] is not None:
        save_evaluations_to_excel(evaluations, request_data['output_file'])
    evaluations = evaluations.reset_index().to_dict(orient='records')
    return evaluations

@app.post('/predict-models', response_model = list[dict])
def predict_models_endpoint(request: PredictModelsRequest = Body(...)):
    request_data = json.loads(request.json())
    models = load_models(request_data['models'])
    X, _ = process_data(request_data['data_path'])
    predictions = predict_models(models, X)
    if request_data['save_to_files'] and len(models) == 1 and isinstance(models[0], RandomForestModel):
        save_predictions_to_jsons(predictions, request_data['data_path'])
    predictions = predictions.to_dict(orient='records')
    return predictions

if __name__ == "__main__":
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host=host, port=port)