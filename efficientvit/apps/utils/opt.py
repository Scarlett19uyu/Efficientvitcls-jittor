from typing import Any, Optional
import jittor as jt
from jittor import optim

__all__ = ["REGISTERED_OPTIMIZER_DICT", "build_optimizer"]

# 注册优化器字典 (Jittor版本)
REGISTERED_OPTIMIZER_DICT: dict[str, tuple[type, dict[str, Any]]] = {
    "sgd": (optim.SGD, {"momentum": 0.9, "nesterov": True}),
    "adam": (optim.Adam, {"betas": (0.9, 0.999), "eps": 1e-8}),
    "adamw": (optim.AdamW, {"betas": (0.9, 0.999), "eps": 1e-8}),
    "rmsprop": (optim.RMSprop, {"alpha": 0.99, "eps": 1e-8}),
}

def build_optimizer(
    net_params, 
    optimizer_name: str, 
    optimizer_params: Optional[dict], 
    init_lr: float
) -> jt.optim.Optimizer:
    """构建Jittor优化器
    
    Args:
        net_params: 网络参数 (可通过model.parameters()获取)
        optimizer_name: 优化器名称 (sgd/adam/adamw等)
        optimizer_params: 覆盖默认参数的字典
        init_lr: 初始学习率
    
    Returns:
        jt.optim.Optimizer实例
    """
    # 获取优化器类和默认参数
    optimizer_class, default_params = REGISTERED_OPTIMIZER_DICT.get(optimizer_name.lower(), (None, None))
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. "
                       f"Available options: {list(REGISTERED_OPTIMIZER_DICT.keys())}")
    
    # 合并参数 (用户参数优先)
    optimizer_params = {} if optimizer_params is None else optimizer_params
    final_params = {**default_params, **optimizer_params}
    
    # Jittor优化器不需要显式传递学习率到构造函数
    optimizer = optimizer_class(net_params, lr=init_lr, **final_params)
    
    # Jittor特有优化 (可选)
    if hasattr(optimizer, 'set_clip_grad'):
        optimizer.set_clip_grad(0.1)  # 示例: 设置梯度裁剪
    
    return optimizer