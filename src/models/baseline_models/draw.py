from sklearn.dummy import DummyClassifier

from configs.constants import DRAW
from models.base_model import BaseModel

class DrawModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.model = DummyClassifier(
            strategy = 'constant',
            constant = DRAW
        )