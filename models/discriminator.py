import torch
from torch import nn


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.disc(x)
        return x
