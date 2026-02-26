"""Shared EnsembleWrapper class for pickle serialization/deserialization.

Both train_all_models.py and run_daily.py import this class so pickle
can locate it regardless of which script is __main__.
"""
import numpy as np


class EnsembleWrapper:
    """Soft-voting ensemble compatible with sklearn predict/predict_proba interface."""

    def __init__(self, models, weights):
        self.models = models  # list of calibrated classifiers
        self.weights = weights
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        for model, w in zip(self.models, self.weights):
            probs += w * model.predict_proba(X)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
