import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    total_iters: int,
    min_lr: float = 1e-6,
) -> LambdaLR:

    def get_lr_lambda(base_lr: float):
        def _lr_lambda(step: int) -> float:
            if base_lr == 0.0:
                return 0.0

            if step < warmup_iters:
                warmup_progress = (step + 1) / max(warmup_iters, 1)
                lr = min_lr + (base_lr - min_lr) * warmup_progress
                return lr / base_lr

            progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
            progress = min(progress, 1.0)

            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = min_lr + (base_lr - min_lr) * cosine_decay

            return lr / base_lr

        return _lr_lambda

    lr_lambdas = [get_lr_lambda(pg["lr"]) for pg in optimizer.param_groups]

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambdas,
    )
