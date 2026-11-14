"""Audio models trained from scratch on log-mel spectrograms: the proposal's
CNN vs CNN+GRU vs lightweight transformer comparison.

All three take the cached (128 mel x 63 frame) log-mel of a 1 s clip and share
one training recipe, so differences come from the architecture:

- the clip's mean dB is subtracted first (loudness alone separates the classes
  in this corpus, ROC-AUC 0.79), then each mel band is standardised with
  training statistics;
- SpecAugment (2 frequency and 2 time masks) and a random circular time shift;
- AdamW, cosine schedule, a fixed number of epochs; no early stopping on
  anything that could be a test set;
- class imbalance (or tier 4's source weights) through per-sample loss weights.

`DeepClassifier` follows the scikit-learn estimator interface, so the shared
tiers in `eval.protocol` run it exactly as they run the sklearn models.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _block(cin, cout, pool):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.MaxPool2d(pool))


class CNN(nn.Module):
    """Four conv blocks, global mean+max pooling. ~210k parameters."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(_block(1, 16, 2), _block(16, 32, 2),
                                      _block(32, 64, 2), _block(64, 96, (2, 1)))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * 96, 1))

    def forward(self, x):                       # x (B, mels, frames)
        h = self.features(x[:, None])           # (B, C, mels', frames')
        h = h.mean(dim=2)                       # pool frequency
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1)).squeeze(1)


class CRNN(nn.Module):
    """Conv front end that pools frequency only, then a bidirectional GRU
    over time: the CNN+GRU of the proposal. ~270k parameters."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.features = nn.Sequential(_block(1, 16, (2, 1)), _block(16, 32, (2, 1)),
                                      _block(32, 64, (2, 2)), _block(64, 64, (4, 1)))
        self.gru = nn.GRU(64 * 4, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * hidden, 1))

    def forward(self, x):
        h = self.features(x[:, None])           # (B, 64, 4, T')
        h = h.flatten(1, 2).transpose(1, 2)     # (B, T', 256)
        h, _ = self.gru(h)
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1)).squeeze(1)


class TinyAST(nn.Module):
    """Audio Spectrogram Transformer in miniature (Gong et al. 2021): 16x7
    non-overlapping patches -> 72 tokens + [CLS], learned positions, 4
    pre-norm encoder layers. ~320k parameters, against AST's 87M."""

    def __init__(self, dim: int = 96, depth: int = 4, heads: int = 4,
                 n_mels: int = 128, n_frames: int = 63, patch=(16, 7)):
        super().__init__()
        self.patch = nn.Conv2d(1, dim, patch, stride=patch)
        n_tok = (n_mels // patch[0]) * (n_frames // patch[1])
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n_tok + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim,
                                           dropout=0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        t = self.patch(x[:, None]).flatten(2).transpose(1, 2)   # (B, N, dim)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.encoder(t)[:, 0])).squeeze(1)


ARCHS = {"cnn": CNN, "cnn+gru": CRNN, "transformer": TinyAST}


def n_params(arch: str) -> int:
    return sum(p.numel() for p in ARCHS[arch]().parameters())


def spec_augment(x: torch.Tensor, gen: torch.Generator, f_max=16, t_max=10,
                 n=2) -> torch.Tensor:
    """In-place frequency and time masks, plus a random circular time shift."""
    B, M, T = x.shape
    shift = int(torch.randint(0, T, (1,), generator=gen))
    x = torch.roll(x, shift, dims=2)
    for _ in range(n):
        f = torch.randint(0, f_max + 1, (B,), generator=gen)
        f0 = (torch.rand(B, generator=gen) * (M - f)).long()
        t = torch.randint(0, t_max + 1, (B,), generator=gen)
        t0 = (torch.rand(B, generator=gen) * (T - t)).long()
        m_idx, t_idx = torch.arange(M)[None], torch.arange(T)[None]
        fm = (m_idx >= f0[:, None]) & (m_idx < (f0 + f)[:, None])
        tm = (t_idx >= t0[:, None]) & (t_idx < (t0 + t)[:, None])
        x = x.masked_fill(fm[:, :, None] | tm[:, None, :], 0.0)
    return x


class DeepClassifier(ClassifierMixin, BaseEstimator):
    """Binary drone classifier on (N, mels, frames) log-mel arrays."""

    def __init__(self, arch: str = "cnn", epochs: int = 20, lr: float = 2e-3,
                 batch: int = 128, weight_decay: float = 1e-2, seed: int = 0,
                 device: str | None = None):
        self.arch, self.epochs, self.lr, self.batch = arch, epochs, lr, batch
        self.weight_decay, self.seed, self.device = weight_decay, seed, device

    # level removal + per-band standardisation, statistics from train only
    def _prep(self, X) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32)
        X = X - X.mean(axis=(1, 2), keepdims=True)
        return torch.from_numpy((X - self.mean_) / self.std_)

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])
        Xc = np.asarray(X, dtype=np.float32)
        Xc = Xc - Xc.mean(axis=(1, 2), keepdims=True)
        self.mean_ = Xc.mean(axis=(0, 2), keepdims=True)[0][None]
        self.std_ = Xc.std(axis=(0, 2), keepdims=True)[0][None] + 1e-6
        del Xc
        if sample_weight is None:           # class-balanced, as class_weight="balanced"
            p = y.mean()
            sample_weight = np.where(y == 1, 0.5 / p, 0.5 / (1 - p))
        w = np.asarray(sample_weight, dtype=np.float32)
        w = w / w.mean()

        dev = self.device or pick_device()
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        net = ARCHS[self.arch]().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        x_all, y_all, w_all = self._prep(X), torch.from_numpy(y), torch.from_numpy(w)
        steps = self.epochs * int(np.ceil(len(y) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=steps,
                                                    pct_start=0.1)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.history_ = []
        net.train()
        for _ in range(self.epochs):
            total = 0.0
            for idx in torch.randperm(len(y), generator=gen).split(self.batch):
                if len(idx) < 2:
                    continue
                xb = spec_augment(x_all[idx], gen).to(dev)
                loss = (loss_fn(net(xb), y_all[idx].to(dev)) * w_all[idx].to(dev)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach()) * len(idx)
            self.history_.append(total / len(y))
        self.net_ = net.cpu().eval()
        return self

    def decision_function(self, X) -> np.ndarray:
        dev = self.device or pick_device()
        net = self.net_.to(dev)
        x = self._prep(X)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 512):
                out.append(net(x[i:i + 512].to(dev)).float().cpu().numpy())
        self.net_ = net.cpu()
        return np.concatenate(out)

    def predict_proba(self, X) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.stack([1 - p, p], axis=1)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
