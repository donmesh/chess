import os
DATA_PATH = f"{os.getenv('PROJECT_ROOT')}/data"
TRAIN_DATA_PATH = f'{DATA_PATH}/train'
TEST_DATA_PATH = f'{DATA_PATH}/test'
SCORES_2014 = f'{DATA_PATH}/rating_2014.txt'
SCORES_2020 = f'{DATA_PATH}/rating_2020.txt'
MODEL_DIR = f"{os.getenv('PROJECT_ROOT')}/model_files"
OUTPUT_DIR = f"{os.getenv('PROJECT_ROOT')}/output_files"