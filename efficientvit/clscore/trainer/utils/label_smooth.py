import jittor as jt
from jittor import init
from jittor import nn

__all__ = ['label_smooth']

def label_smooth(target: jt.Var, n_classes: int, smooth_factor=0.1) -> jt.Var:
    """
    完全兼容的标签平滑实现，适用于所有Jittor版本
    
    Args:
        target: 必须是1D Jittor int64张量，shape [batch_size]
        n_classes: 类别总数
        smooth_factor: 平滑系数 (0.0-1.0)
    """
    # 输入验证
    if not isinstance(target, jt.Var):
        target = jt.array(target)
    target = target.int64().reshape(-1)  # 确保是1D int64
    
    batch_size = target.shape[0]
    
    # 方案1：使用高级索引（推荐）
    soft_target = jt.zeros((batch_size, n_classes))
    rows = jt.arange(batch_size).reshape(-1, 1)
    cols = target.reshape(-1, 1)
    soft_target[rows, cols] = 1.0
    
    # 方案2：备选实现（如果方案1有问题）
    # soft_target = jt.init.eye(n_classes)[target]
    
    # 应用平滑
    soft_target = soft_target * (1 - smooth_factor) + (smooth_factor / n_classes)
    return soft_target