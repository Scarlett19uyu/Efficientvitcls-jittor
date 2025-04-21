import copy
import math
import jittor as jt
from jittor import nn
from typing import Dict, Optional

from efficientvit.models.utils import is_parallel

__all__ = ["EMA"]

class EMA:
    """Jittor优化的指数移动平均模型"""
    def __init__(self, model: nn.Module, decay: float, warmup_steps: int = 200):
        """
        参数:
            model: 要跟踪的模型
            decay: 衰减率 (0-1)
            warmup_steps: 热身步数，用于渐进式衰减
        """
        # 深拷贝模型参数并设置为评估模式
        self.shadows = copy.deepcopy(model.module if is_parallel(model) else model)
        self.shadows.eval()
        
        # 初始化参数
        self.decay = decay
        self.warmup_steps = warmup_steps
        
        # Jittor特有的参数冻结方式
        self._freeze_shadows()
        
    def _freeze_shadows(self):
        for p in self.shadows.parameters():
            p.stop_grad()  # 停止梯度计算
            p.is_attached = False  # 从计算图中分离
            p.requires_grad = False  # 兼容性设置

    def get_decay(self, global_step: int) -> float:
        """计算当前衰减率（带warmup）"""
        if self.warmup_steps <= 0:
            return self.decay
        progress = min(global_step / self.warmup_steps, 1.0)
        return self.decay * progress


    def step(self, model: nn.Module, global_step: Optional[int] = None):
        current_decay = self.get_decay(global_step or self.warmup_steps)
        
        # 使用内存高效的更新方式
        with jt.no_grad():
            model_sd = (model.module if is_parallel(model) else model).state_dict()
            for k, shadow_p in self.shadows.named_parameters():
                if k in model_sd:
                    new_val = model_sd[k].detach()
                    shadow_p.update(shadow_p * current_decay + new_val * (1 - current_decay))
                    jt.gc()  # 显式释放临时变量
                    
            # 每100步强制垃圾回收
            if global_step % 100 == 0:
                jt.gc()
                jt.sync_all()

    def state_dict(self) -> Dict:
        """返回完整状态字典"""
        return {
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
            'state_dict': self.shadows.state_dict(),
            'version': 'jittor-1.0'  # 标识Jittor专用版本
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """加载状态字典"""
        if state_dict.get('version', '') == 'jittor-1.0':
            self.decay = state_dict['decay']
            self.warmup_steps = state_dict['warmup_steps']
            self.shadows.load_state_dict(state_dict['state_dict'])
        else:
            # 兼容旧版本加载
            self.shadows.load_state_dict(state_dict)
        
        # 重新冻结参数
        self._freeze_shadows()

    def __call__(self, *args, **kwargs):
        """使EMA模型可调用"""
        return self.shadows(*args, **kwargs)

    def __repr__(self):
        return f"EMA(decay={self.decay}, warmup={self.warmup_steps}, model={self.shadows.__class__.__name__})"