from sklearn.dummy import DummyClassifier

from configs.constants import BLACK_WIN
from models.base_model import BaseModel

class BlackWinsModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.model = DummyClassifier(
            strategy = 'constant',
            constant = BLACK_WIN
        )