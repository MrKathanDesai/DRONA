from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs[:, None])).squeeze(1)


class CNNRecurrent(nn.Module):
    def __init__(self, recurrent: str) -> None:
        super().__init__()
        if recurrent not in {"gru", "lstm"}:
            raise ValueError(f"unsupported recurrent layer: {recurrent}")
        self.frontend = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        layer = nn.GRU if recurrent == "gru" else nn.LSTM
        self.sequence = layer(
            input_size=64,
            hidden_size=32,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.frontend(inputs).transpose(1, 2)
        sequence, _ = self.sequence(sequence)
        pooled = torch.cat((sequence.mean(dim=1), sequence.amax(dim=1)), dim=1)
        return self.classifier(pooled).squeeze(1)


class LightweightTransformer(nn.Module):
    def __init__(self, n_frames: int = 21) -> None:
        super().__init__()
        width = 64
        self.projection = nn.Linear(64, width)
        self.position = nn.Parameter(torch.zeros(1, n_frames, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(0.2),
            nn.Linear(width, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs.transpose(1, 2)
        if sequence.shape[1] > self.position.shape[1]:
            raise ValueError(
                f"received {sequence.shape[1]} frames; maximum is "
                f"{self.position.shape[1]}"
            )
        encoded = self.encoder(
            self.projection(sequence) + self.position[:, :sequence.shape[1]]
        )
        return self.classifier(encoded.mean(dim=1)).squeeze(1)


MODEL_FACTORIES = {
    "small_cnn": SmallCNN,
    "cnn_gru": lambda: CNNRecurrent("gru"),
    "cnn_lstm": lambda: CNNRecurrent("lstm"),
    "lightweight_transformer": LightweightTransformer,
}
