from sklearn.dummy import DummyClassifier

from configs.constants import WHITE_WIN
from models.base_model import BaseModel

class WhiteWinsModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.model = DummyClassifier(
            strategy = 'constant',
            constant = WHITE_WIN
        )