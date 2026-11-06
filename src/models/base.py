from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    def __init__(self):
        self.history = {}

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def predict_proba(self, X):
        raise NotImplementedError(
            f"{self.__class__.__name__} chưa hỗ trợ predict_proba()."
        )