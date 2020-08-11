import torch
from torch import nn


class LocalEncoder(nn.Module):

    def __init__(self):
        super(LocalEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # 3 x 96 x 96

            nn.Conv2d(3, 16, (5, 5), stride=2, padding=2), # 48
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 32, (5, 5), stride=2, padding=2), # 24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, (5, 5), stride=2, padding=2), # 12
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
