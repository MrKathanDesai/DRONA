"""Grad-CAM for the frozen encoder + linear head.

The head is a StandardScaler + LogisticRegression on the pooled embedding,
so the drone logit is a linear function of the embedding and can be
back-propagated through the encoder. Grad-CAM (Selvaraju et al. 2017)
weights each channel of an intermediate feature map by its mean gradient
and keeps the positive part: the regions of the log-mel spectrogram that
pushed the score *towards* "drone".

Resolution is coarse. A 1 s clip is 101 x 64 (time x mel) at the encoder's
input; block 11 (stride 16) sees it as 6 x 4 cells and block 5 (stride 8) as
12 x 8. Read the maps as "which part of the second and which frequency
region", not as pixel-level evidence.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dronedet.models.pretrained import Encoder

LAYERS = {"coarse (block 11, 6x4)": 11, "fine (block 5, 12x8)": 5}


def linear_head(pipeline) -> tuple[np.ndarray, float]:
    """(w, b) such that logit = w @ embedding + b, for StandardScaler + LR."""
    scaler, lr = pipeline[0], pipeline[-1]
    coef = lr.coef_.ravel()
    w = coef / scaler.scale_
    b = float(lr.intercept_[0] - np.sum(coef * scaler.mean_ / scaler.scale_))
    return w.astype(np.float32), b


def gradcam(enc: Encoder, y16k: np.ndarray, w: np.ndarray, b: float,
            layer: int = 11) -> tuple[np.ndarray, np.ndarray, float]:
    """One 1 s clip -> (log-mel (T, 64), CAM in [0, 1] (T, 64), logit)."""
    M = enc.logmel(y16k)
    net, dev = enc.net, enc.device
    x = torch.from_numpy(M[None, None]).to(dev)
    with torch.enable_grad():
        x = net.bn0(x.transpose(1, 3)).transpose(1, 3)
        acts = None
        for i, block in enumerate(net.features):
            x = block(x)
            if i == layer:
                acts = x
                acts.retain_grad()
        x = x.mean(dim=3)
        emb = x.max(dim=2).values + x.mean(dim=2)
        z = (emb[0] * torch.from_numpy(w).to(dev)).sum() + b
        z.backward()
    alpha = acts.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((alpha * acts).sum(dim=1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=M.shape, mode="bilinear", align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    top = cam.max()
    cam = cam / top if top > 0 else cam
    net.zero_grad(set_to_none=True)
    return M, cam.astype(np.float32), float(z.detach().cpu())


def mel_axis_hz(n_mels: int = 64, fmin: float = 50.0, fmax: float = 14000.0) -> np.ndarray:
    """Centre frequencies of the checkpoint's mel bands (PANNs defaults)."""
    import librosa
    return librosa.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)[1:-1]
