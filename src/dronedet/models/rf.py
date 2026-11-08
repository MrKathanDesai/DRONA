
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


def shape_features(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=2, keepdims=True)


def band_features(X: np.ndarray, n_bands: int = 64) -> np.ndarray:
    parts = np.array_split(X, n_bands, axis=2)
    return np.concatenate([p.mean(axis=2) for p in parts], axis=1)


class TinyCNN(ClassifierMixin, BaseEstimator):


    def __init__(self, epochs: int = 40, lr: float = 1e-3, batch: int = 64, seed: int = 0):
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed

    def _build(self) -> nn.Module:
        layers, ch = [], 2
        for out in (8, 16, 32, 32):
            layers += [nn.Conv1d(ch, out, 7, stride=2, padding=3),
                       nn.BatchNorm1d(out), nn.ReLU()]
            ch = out
        return nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                             nn.Dropout(0.3), nn.Linear(ch, 1))

    def _tensor(self, X) -> torch.Tensor:
        return torch.tensor((np.asarray(X) - self.mean_) / self.std_,
                            dtype=torch.float32)

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.mean_ = np.asarray(X).mean(axis=0, keepdims=True)
        self.std_ = np.asarray(X).std(axis=0, keepdims=True) + 1e-6
        torch.manual_seed(self.seed)
        self.net_ = self._build()
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        weight = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        x_t, y_t = self._tensor(X), torch.as_tensor(y, dtype=torch.float32)
        gen = torch.Generator().manual_seed(self.seed)
        self.net_.train()
        for _ in range(self.epochs):
            for idx in torch.randperm(len(x_t), generator=gen).split(self.batch):
                if len(idx) < 2:          # batch norm needs two samples
                    continue
                opt.zero_grad()
                loss_fn(self.net_(x_t[idx]).squeeze(1), y_t[idx]).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(self._tensor(X)).squeeze(1)).numpy()
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  class_weight="balanced", n_jobs=-1, random_state=0)


def logreg():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))


MODELS = {
    "logreg": (lambda X: band_features(shape_features(X)), logreg),
    "forest": (lambda X: band_features(shape_features(X)), forest),
    "cnn": (shape_features, TinyCNN),
    "forest+level": (band_features, forest),   
}
DEPLOY = "forest"
