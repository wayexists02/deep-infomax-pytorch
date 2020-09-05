import torch
from torch import nn


class GlobalEncoder(nn.Module):

    def __init__(self):
        super(GlobalEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # 64 x 12 x 12

            nn.Conv2d(64, 128, (3, 3), stride=2, padding=1), # 6
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, (3, 3), stride=2, padding=1), # 3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, (3, 3), stride=1, padding=0), # 1
        )

    def forward(self, x):
        N = x.size(0)

        x = self.encoder(x)
        # x = x.view(N, -1)
        return x
