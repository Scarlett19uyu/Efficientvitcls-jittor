from typing import Any, Optional
import numpy as np
import jittor as jt

__all__ = [
    "jt_randint",
    "jt_random",
    "jt_shuffle",
    "jt_uniform",
    "jt_random_choices",
]

def jt_randint(low: int, high: int, generator: Optional[Any] = None) -> int:
    """uniform: [low, high)"""
    if low == high:
        return low
    else:
        assert low < high
        return int(jt.randint(low, high, shape=(1,)).item())

def jt_random(generator: Optional[Any] = None) -> float:
    """uniform distribution on the interval [0, 1)"""
    return float(jt.rand(1).item())

def jt_shuffle(src_list: list[Any], generator: Optional[Any] = None) -> list[Any]:
    """Shuffle a list using Jittor's random permutation"""
    rand_indexes = jt.randperm(len(src_list)).tolist()
    return [src_list[i] for i in rand_indexes]

def jt_uniform(low: float, high: float, generator: Optional[Any] = None) -> float:
    """uniform distribution on the interval [low, high)"""
    rand_val = jt_random(generator)
    return (high - low) * rand_val + low

def jt_random_choices(
    src_list: list[Any],
    generator: Optional[Any] = None,
    k: int = 1,
    weight_list: Optional[list[float]] = None,
) -> Any | list:
    """Random choices with optional weights"""
    if weight_list is None:
        if k == 1:
            return src_list[jt.randint(0, len(src_list), shape=(1,)).item()]
        else:
            rand_idx = jt.randint(0, len(src_list), shape=(k,))
            return [src_list[i.item()] for i in rand_idx]
    else:
        assert len(weight_list) == len(src_list)
        accumulate_weight_list = np.cumsum(weight_list)

        out_list = []
        for _ in range(k):
            val = jt_uniform(0, accumulate_weight_list[-1], generator)
            active_id = 0
            for i, weight_val in enumerate(accumulate_weight_list):
                if weight_val > val:
                    active_id = i
                    break
            out_list.append(src_list[active_id])

    return out_list[0] if k == 1 else out_list