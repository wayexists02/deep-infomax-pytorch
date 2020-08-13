import torch
from torch import nn
from .local_encoder import LocalEncoder
from .global_encoder import GlobalEncoder


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.local_encoder = LocalEncoder()
        self.global_encoder = GlobalEncoder()

    def forward(self, x):
        local_features = self.local_encoder(x)
        global_features = self.global_encoder(local_features)
        return local_features, global_features
