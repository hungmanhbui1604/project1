import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    total_iters: int,
    min_lr: float = 1e-6,
) -> LambdaLR:
    """
    Cosine annealing scheduler with linear warmup.

    - Warmup: linearly ramp lr from 0 → base_lr over `warmup_iters` steps.
    - Cosine: decay from base_lr → min_lr over the remaining steps.
    """
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def _lr_lambda(step: int) -> float:
        # Use first param group's base_lr for the ratio calculation
        base_lr = base_lrs[0]
        if base_lr == 0:
            return 0.0

        if step < warmup_iters:
            # Linear warmup: 0 → 1
            return step / max(warmup_iters, 1)
        else:
            # Cosine decay: base_lr → min_lr
            progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = min_lr + (base_lr - min_lr) * cosine_decay
            return lr / base_lr

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)
