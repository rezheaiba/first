"""
# @Time    : 2022/9/6 18:11
# @File    : model.py
# @Author  : rezheaiba
"""
import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),

            nn.Sigmoid()
        )

    def forward(self, inputs):
        z = self.encoder(inputs)
        outputs = self.decoder(z)

        return z, outputs