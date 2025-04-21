import jittor as jt
from jittor import init
import collections
import os
from inspect import signature
from typing import Any, Callable, Optional, Union, Tuple
from jittor import nn

__all__ = ['is_parallel', 'get_device', 'get_same_padding', 'resize', 
           'build_kwargs_from_config', 'load_state_dict_from_file', 
           'get_submodule_weights', 'get_dtype', 'get_dtype_from_str']

def is_parallel(model: nn.Module) -> bool:
    # Jittor 的并行模型判断
    return hasattr(model, 'is_parallel') and model.is_parallel

def get_device(model: nn.Module) -> str:
    # Jittor 使用字符串表示设备
    return 'cuda' if jt.has_cuda else 'cpu'

def get_dtype(model: nn.Module) -> str:
    # 获取第一个参数的数据类型
    for param in model.parameters():
        return param.dtype
    return 'float32'

def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert ((kernel_size % 2) > 0), 'kernel size should be odd number'
        return (kernel_size // 2)

def resize(x: jt.Var, size: Optional[Any]=None, 
          scale_factor: Optional[list[float]]=None, 
          mode: str='bicubic', 
          align_corners: Optional[bool]=False) -> jt.Var:
    # Jittor 的插值实现
    if mode in {'bilinear', 'bicubic'}:
        return jt.nn.interpolate(x, size=size, scale_factor=scale_factor, 
                               mode=mode, align_corners=align_corners)
    elif mode in {'nearest', 'area'}:
        return jt.nn.interpolate(x, size=size, scale_factor=scale_factor, 
                               mode=mode)
    else:
        raise NotImplementedError(f'resize(mode={mode}) not implemented.')

def build_kwargs_from_config(config: dict, target_func: Callable) -> dict:
    # 保持不变
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs

def load_state_dict_from_file(file: str, only_state_dict=True) -> dict:
    # Jittor 的模型加载方式
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = jt.load(file)
    if only_state_dict and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    return checkpoint

def get_submodule_weights(weights: collections.OrderedDict, prefix: str):
    # 保持不变
    submodule_weights = collections.OrderedDict()
    len_prefix = len(prefix)
    for key, weight in weights.items():
        if key.startswith(prefix):
            submodule_weights[key[len_prefix:]] = weight
    return submodule_weights

def get_dtype_from_str(dtype: str) -> str:
    # Jittor 使用字符串表示数据类型
    if dtype == 'fp32':
        return 'float32'
    if dtype == 'fp16':
        return 'float16'
    if dtype == 'bf16':
        return 'bfloat16'
    raise NotImplementedError(f'dtype {dtype} is not supported')