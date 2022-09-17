from sklearn.dummy import DummyClassifier

from models.base_model import BaseModel

class RandomWinnerModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.model = DummyClassifier(strategy = 'uniform')
