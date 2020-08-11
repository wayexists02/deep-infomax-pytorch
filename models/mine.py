import numpy as np
import torch
from torch import nn


class MINE(nn.Module):

    def __init__(self):
        super(MINE, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(64, 512),
        )

    def forward(self, local_features, global_feature):
        N = local_features.size(0)
        r = np.random.randint(0, N-1)
        random_indices = np.arange(N) + r
        random_indices[random_indices >= N] -= N

        log_bilinear_pos = self._bilinear(local_features, global_feature)
        log_bilinear_neg = self._bilinear(local_features, global_feature[random_indices])

        log_maximum = torch.where(log_bilinear_pos > log_bilinear_neg, log_bilinear_pos, log_bilinear_neg).detach()
        log_bilinear_pos -= log_maximum
        log_bilinear_neg -= log_maximum

        jsd_mine = torch.mean(-self._softplus(-log_bilinear_pos)) - torch.mean(self._softplus(log_bilinear_neg))
        return jsd_mine

    def _bilinear(self, local_features, global_feature):
        N, C, H, W = local_features.size()
        local_features = local_features.permute(0, 2, 3, 1).view(N*H*W, C)
        global_features = global_features.view(N, 1, 512)

        embedded_local_features = self.embedding(local_features)
        embedded_local_features = embedded_local_features.view(N, H*W, 512)
        log_bilinear = torch.sum(embedded_local_features*global_feature, dim=-1)

        return log_bilinear

    def _softplus(self, z):
        return torch.log(1 + torch.exp(z))
