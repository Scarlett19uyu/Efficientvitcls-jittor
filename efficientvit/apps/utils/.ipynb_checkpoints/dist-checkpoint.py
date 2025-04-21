import os
import jittor as jt
from typing import Union, List
from efficientvit.models.utils.list import list_mean, list_sum

__all__ = [
    "dist_init",
    "is_dist_initialized",
    "get_dist_rank",
    "get_dist_size",
    "is_master",
    "dist_barrier",
    "get_dist_local_rank",
    "sync_tensor",
]

def dist_init() -> None:
    """初始化Jittor分布式训练环境"""
    if is_dist_initialized():
        return
    
    # Jittor会自动从环境变量读取RANK/WORLD_SIZE等信息
    jt.distributed.init()
    
    # 如果初始化失败，设置为单机模式
    if not is_dist_initialized():
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        print("warning: dist not init, fallback to single process")

def is_dist_initialized() -> bool:
    """检查分布式环境是否已初始化"""
    return jt.in_mpi and jt.world_size > 1

def get_dist_rank() -> int:
    """获取当前进程的全局rank"""
    return int(os.getenv("RANK", "0"))

def get_dist_size() -> int:
    """获取进程总数"""
    return int(os.getenv("WORLD_SIZE", "1"))

def is_master() -> bool:
    """判断当前进程是否为主进程"""
    return get_dist_rank() == 0

def dist_barrier() -> None:
    """进程同步屏障"""
    if is_dist_initialized():
        jt.mpi.barrier()

def get_dist_local_rank() -> int:
    """获取当前节点上的本地rank"""
    return int(os.getenv("LOCAL_RANK", "0"))

def sync_tensor(tensor: Union[jt.Var, float], reduce: str = "mean") -> Union[jt.Var, List[jt.Var]]:
    """
    同步张量到所有进程 (Jittor版本)
    
    Args:
        tensor: 要同步的张量或标量值
        reduce: 同步方式 ["mean", "sum", "cat", "root", None]
    """
    if not is_dist_initialized():
        return tensor
    
    # 转换标量为Jittor张量
    if not isinstance(tensor, jt.Var):
        tensor = jt.array([float(tensor)])
    
    # 确保张量在GPU上
    if jt.has_cuda:
        tensor = tensor.cuda()
    
    # 执行同步操作
    if reduce == "mean":
        return jt.mpi.all_reduce(tensor, op="mean")
    elif reduce == "sum":
        return jt.mpi.all_reduce(tensor, op="sum")
    elif reduce == "cat":
        gathered = jt.mpi.all_gather(tensor)
        return jt.concat(gathered, dim=0)
    elif reduce == "root":
        jt.mpi.broadcast(tensor, root=0)
        return tensor
    else:
        return jt.mpi.all_gather(tensor)