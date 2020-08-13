import torch
from torch import nn


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        N = x.size(0)

        x = self.disc(x)
        x = x.view(N)
        return x
