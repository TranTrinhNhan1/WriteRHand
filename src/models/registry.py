"""Name -> class registry (used by ``BaseClassifier.load`` and the experiment runner)."""

from src.models.adaboost import AdaBoostSAMME
from src.models.logistic_regression import SoftmaxRegression
from src.models.mlp import MLP
from src.models.naive_bayes import GaussianNB

MODELS = {
    "GaussianNB": GaussianNB,
    "SoftmaxRegression": SoftmaxRegression,
    "MLP": MLP,
    "AdaBoostSAMME": AdaBoostSAMME,
}
