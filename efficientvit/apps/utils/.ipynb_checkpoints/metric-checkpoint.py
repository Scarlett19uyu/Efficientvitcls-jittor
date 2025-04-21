import jittor as jt
from efficientvit.apps.utils.dist import sync_tensor

__all__ = ["AverageMeter"]

class AverageMeter:
    """Computes and stores the average and current value (Jittor版本)"""
    
    def __init__(self, is_distributed=True):
        self.is_distributed = is_distributed
        self.sum = 0
        self.count = 0

    def _sync(self, val: jt.Var | int | float) -> jt.Var | int | float:
        """同步分布式环境下的值"""
        if isinstance(val, (int, float)):
            val = jt.array(val)
        return sync_tensor(val, reduce="sum") if self.is_distributed else val

    def update(self, val: jt.Var | int | float, delta_n=1):
        """更新统计量
        
        Args:
            val: 当前值 (可以是标量或Jittor变量)
            delta_n: 当前batch的样本数
        """
        delta_n = jt.array(delta_n) if isinstance(delta_n, (int, float)) else delta_n
        val = jt.array(val) if isinstance(val, (int, float)) else val
        
        # 同步更新
        synced_count = self._sync(delta_n)
        synced_val = self._sync(val * delta_n)
        
        # Jittor变量需要转换为Python数值
        self.count += synced_count.item() if isinstance(synced_count, jt.Var) else synced_count
        self.sum += synced_val.item() if isinstance(synced_val, jt.Var) else synced_val

    def get_count(self) -> int | float:
        """获取总样本数"""
        return self.count

    @property
    def avg(self) -> float:
        """计算平均值"""
        if self.count == 0:
            return -1.0
        return self.sum / self.count
