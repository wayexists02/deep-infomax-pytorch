import numpy as np
import torch
from torch import nn


class MINE(nn.Module):

    def __init__(self):
        super(MINE, self).__init__()

        self.embedding_dim = 128
        
        self.mine_estimator = nn.Sequential(
            nn.Conv2d(576, 1, (12, 12), stride=1, padding=0),
            nn.Sigmoid()
        )

        self.local_embedder = nn.Sequential(
            nn.Conv2d(64, self.embedding_dim, (3, 3), stride=1, padding=1),
        )

        self.global_embedder = nn.Sequential(
            nn.Conv2d(512, self.embedding_dim, (1, 1), stride=1, padding=0),
        )

    def forward(self, local_features, global_features):
        N = local_features.size(0)
        
        # mutual_information = self.compute_mine(local_features, global_features)
        mutual_information = self.compute_info_nse(local_features, global_features)

        return mutual_information

    def compute_mine(self, local_features, global_features):
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

    def compute_info_nse(self, local_features, global_features):
        N, C, H, W = local_features.size()

        local_feature_embedded = self.local_embedder(local_features)
        global_feature_embedded = self.global_embedder(global_features)

        local_feature_embedded = local_feature_embedded.permute(0, 2, 3, 1).reshape(N, H*W, self.embedding_dim)
        global_feature_embedded = global_feature_embedded.view(N, -1)

        dotted = torch.einsum("nij,mj->nmi", local_feature_embedded, global_feature_embedded)
        t = torch.sigmoid(dotted)

        pos_mask = torch.eye(N).to(local_features.device).unsqueeze(-1)
        neg_mask = 1 - pos_mask

        t_pos = torch.sum(t*pos_mask, dim=0)
        t_neg = torch.log(torch.sum(torch.exp(t*neg_mask), dim=0))

        mutual_information = torch.mean(t_pos - t_neg)
        return mutual_information

    def _softplus(self, z):
        return torch.log(1 + torch.exp(z))
