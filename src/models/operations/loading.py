import os
import re
from configs.paths import MODEL_DIR
from models.base_model import BaseModel
from utils.error_codes import NoModelsFoundException

def load_models(model_names: list[str]) -> list[BaseModel]:
    folders = [name for name in os.listdir(MODEL_DIR) if re.match(r'^\d{4}-\d{2}-\d{2}_\d{6}$', name)]
    folders = sorted(folders, reverse=True)

    models = []
    selected_models = []
    for model_name in model_names:
        if model_name in selected_models:
            continue
        for folder_name in folders:
            path = f'{MODEL_DIR}/{folder_name}/{model_name}.pickle'
            if os.path.isfile(path):
                models.append(BaseModel().load(path))
                selected_models.append(model_name)
                break
    if not models:
        raise NoModelsFoundException
    return models