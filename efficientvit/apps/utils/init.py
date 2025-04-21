import jittor as jt
from jittor import nn
from typing import Union, List

__all__ = ["init_modules", "zero_last_gamma"]

def init_modules(model: Union[nn.Module, List[nn.Module]], init_type: str = "trunc_normal") -> None:
    """Jittor版本的模型参数初始化
    
    Args:
        model: 要初始化的模型或模型列表
        init_type: 初始化类型，支持 "trunc_normal" 及其变体
    """
    _DEFAULT_INIT_PARAM = {"trunc_normal": 0.02}

    if isinstance(model, list):
        for sub_module in model:
            init_modules(sub_module, init_type)
    else:
        init_params = init_type.split("@")
        init_params = float(init_params[1]) if len(init_params) > 1 else None

        def init_func(param):
            if init_type.startswith("trunc_normal"):
                std = _DEFAULT_INIT_PARAM["trunc_normal"] if init_params is None else init_params
                # Jittor的截断正态分布初始化
                jt.init.trunc_normal_(param, std=std)
            else:
                raise NotImplementedError(f"Unsupported init type: {init_type}")

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                init_func(m.weight)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.Embedding):
                init_func(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.weight.fill_(1)
                m.bias.zero_()
            else:
                weight = getattr(m, "weight", None)
                bias = getattr(m, "bias", None)
                if isinstance(weight, jt.Var):
                    init_func(weight)
                if isinstance(bias, jt.Var):
                    bias.zero_()

def zero_last_gamma(model: nn.Module, init_val: float = 0) -> None:
    """Jittor版本的最后一层gamma初始化
    
    Args:
        model: 目标模型
        init_val: 初始化值 (默认为0)
    """
    import efficientvit.models.nn.ops as ops

    for m in model.modules():
        if isinstance(m, ops.ResidualBlock) and isinstance(m.shortcut, ops.IdentityLayer):
            if isinstance(m.main, (ops.DSConv, ops.MBConv, ops.FusedMBConv)):
                parent_module = m.main.point_conv
            elif isinstance(m.main, ops.ResBlock):
                parent_module = m.main.conv2
            elif isinstance(m.main, ops.ConvLayer):
                parent_module = m.main
            elif isinstance(m.main, ops.LiteMLA):
                parent_module = m.main.proj
            else:
                parent_module = None
                
            if parent_module is not None:
                norm = getattr(parent_module, "norm", None)
                if norm is not None and hasattr(norm, "weight"):
                    norm.weight.fill_(init_val)