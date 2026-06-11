import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(
        self, embed_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # precompute
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embs: torch.Tensor, labels: torch.Tensor):
        # normalize
        embs = F.normalize(embs, dim=1)
        weight = F.normalize(self.weight, dim=1)

        # cosine similarity
        cosine = F.linear(embs, weight)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # compute margin only for the ground-truth classes
        idx = torch.arange(embs.size(0), device=embs.device)
        target_cosine = cosine[idx, labels]
        sin_theta = torch.sqrt(1.0 - target_cosine**2)
        phi = target_cosine * self.cos_m - sin_theta * self.sin_m
        phi = torch.where(target_cosine > self.th, phi, target_cosine - self.mm)

        # update logits and compute loss
        logits = cosine.clone()
        phi = phi.to(logits.dtype)
        logits[idx, labels] = phi
        logits *= self.scale
        loss = F.cross_entropy(logits, labels)

        return loss, logits