from typing import Callable, Tuple  # Added Tuple import
import jittor as jt

__all__ = ["rand_bbox"]

def rand_bbox(
    h: int,
    w: int,
    lam: float,
    rand_func: Callable = jt.rand,
) -> Tuple[int, int, int, int]:  # Changed tuple to Tuple
    """randomly sample bbox, used in cutmix (Jittor version)"""
    cut_rat = jt.sqrt(1.0 - lam)
    cut_w = w * cut_rat
    cut_h = h * cut_rat

    # uniform sampling using Jittor
    cx = rand_func(0, w)
    cy = rand_func(0, h)

    bbx1 = int(jt.clamp(cx - cut_w / 2, min_v=0, max_v=w))
    bby1 = int(jt.clamp(cy - cut_h / 2, min_v=0, max_v=h))
    bbx2 = int(jt.clamp(cx + cut_w / 2, min_v=0, max_v=w))
    bby2 = int(jt.clamp(cy + cut_h / 2, min_v=0, max_v=h))

    return bbx1, bby1, bbx2, bby2