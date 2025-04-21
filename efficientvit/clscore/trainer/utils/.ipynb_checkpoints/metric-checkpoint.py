import jittor as jt
from jittor import nn

__all__ = ["accuracy"]

def accuracy(output: jt.Var, target: jt.Var, topk=(1,)) -> list[jt.Var]:
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.shape[0]

    # 获取topk预测结果
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # 参数名更明确
    pred = pred.transpose(1, 0)  # 替换.t()操作
    
    # 计算正确预测数
    correct = pred == target.reshape((1, -1)).expand_as(pred)  # 修正reshape语法

    res = []
    for k in topk:
        # 计算topk准确率
        correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
        res.append(correct_k * (100.0 / batch_size))  # 替换mul_为普通乘法
        
    return res