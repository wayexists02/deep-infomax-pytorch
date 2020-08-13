import numpy as np
import torch
from torch import nn


class MINE(nn.Module):

    def __init__(self):
        super(MINE, self).__init__()
        
        self.mine_estimator = nn.Sequential(
            nn.Conv2d(576, 1, (12, 12), stride=1, padding=0), # 4
            nn.Sigmoid()
        )

    def forward(self, local_features, global_features):
        N = local_features.size(0)

        r = np.random.randint(0, N-1)
        random_indices = np.arange(N) + r
        random_indices[random_indices >= N] -= N

        t_pos = self.compute_t(local_features, global_features)
        t_neg = self.compute_t(local_features, global_features[random_indices])

        mine = torch.mean(-self._softplus(-t_pos)) - torch.mean(self._softplus(t_neg))
        return mine

    def compute_t(self, local_features, global_features):
        N, C, H, W = local_features.size()
        global_features = global_features.view(N, 512, 1, 1)
        global_features = global_features.repeat(1, 1, 12, 12)
        concated = torch.cat([local_features, global_features], dim=1)

        t = self.mine_estimator(concated)
        return t

    def _softplus(self, z):
        return torch.log(1 + torch.exp(z))
