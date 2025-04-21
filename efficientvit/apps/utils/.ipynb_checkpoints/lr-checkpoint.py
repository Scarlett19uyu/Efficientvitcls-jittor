import math
import jittor as jt
from typing import List, Union
from efficientvit.models.utils.list import val2list

__all__ = ["CosineLRwithWarmup", "ConstantLRwithWarmup"]

class CosineLRwithWarmup:
    def __init__(
        self,
        optimizer: jt.optim.Optimizer,
        warmup_steps: int,
        warmup_lr: float,
        decay_steps: Union[int, List[int]],
        last_epoch: int = -1,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.decay_steps = val2list(decay_steps)
        self.last_epoch = last_epoch
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr - self.warmup_lr) * (self.last_epoch + 1) / self.warmup_steps + self.warmup_lr
                for base_lr in self.base_lrs
            ]
        else:
            current_steps = self.last_epoch - self.warmup_steps
            decay_steps = [0] + self.decay_steps
            idx = len(decay_steps) - 2
            for i, decay_step in enumerate(decay_steps[:-1]):
                if decay_step <= current_steps < decay_steps[i + 1]:
                    idx = i
                    break
            current_steps -= decay_steps[idx]
            decay_step = decay_steps[idx + 1] - decay_steps[idx]
            return [0.5 * base_lr * (1 + math.cos(math.pi * current_steps / decay_step)) for base_lr in self.base_lrs]

    def step(self, epoch=None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
            
        lrs = self.get_lr()
        for i, lr in enumerate(lrs):
            if i == 0:
                self.optimizer.lr = lr
            else:
                self.optimizer.param_groups[i]['lr'] = lr

class ConstantLRwithWarmup:
    def __init__(
        self,
        optimizer: jt.optim.Optimizer,
        warmup_steps: int,
        warmup_lr: float,
        last_epoch: int = -1,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.last_epoch = last_epoch
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr - self.warmup_lr) * (self.last_epoch + 1) / self.warmup_steps + self.warmup_lr
                for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs

    def step(self, epoch=None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
            
        lrs = self.get_lr()
        for i, lr in enumerate(lrs):
            if i == 0:
                self.optimizer.lr = lr
            else:
                self.optimizer.param_groups[i]['lr'] = lr