from typing import Optional
import jittor as jt
from jittor import nn

__all__ = ["LayerNorm2d", "TritonRMSNorm2d", "build_norm", "reset_bn", "set_norm_eps"]

class LayerNorm2d(nn.LayerNorm):
    def execute(self, x: jt.Var) -> jt.Var:
        out = x - jt.mean(x, dim=1, keepdims=True)
        out = out / jt.sqrt(jt.sqr(out).mean(dim=1, keepdims=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)
        return out

class TritonRMSNorm2d(nn.LayerNorm):
    def execute(self, x: jt.Var) -> jt.Var:
        # Jittor暂不支持Triton，改为普通实现
        out = x / jt.sqrt(jt.mean(jt.sqr(x), dim=1, keepdims=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)
        return out

# Jittor注册的归一化层字典
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn": nn.BatchNorm2d,  # Jittor中使用"bn"而非"bn2d"
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "trms2d": TritonRMSNorm2d,
}

def build_norm(name="bn", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name in ["ln", "ln2d", "trms2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        return norm_cls(**kwargs)
    return None

def reset_bn(
    model: nn.Module,
    data_loader: list,
    sync=True,
    progress_bar=False,
) -> None:
    import copy
    from tqdm import tqdm

    bn_mean = {}
    bn_var = {}

    tmp_model = copy.deepcopy(model)
    
    # 初始化统计量收集
    for name, m in tmp_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_mean[name] = {"sum": jt.zeros(m.num_features), "count": 0}
            bn_var[name] = {"sum": jt.zeros(m.num_features), "count": 0}

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    # 计算当前batch的统计量
                    batch_mean = x.mean([0, 2, 3], keepdims=True)  # [1, C, 1, 1]
                    batch_var = (x - batch_mean).sqr().mean([0, 2, 3], keepdims=True)
                    
                    batch_mean = batch_mean.squeeze()  # [C]
                    batch_var = batch_var.squeeze()     # [C]

                    # 更新统计量（无需立即同步）
                    mean_est["sum"] += batch_mean * x.shape[0]
                    var_est["sum"] += batch_var * x.shape[0]
                    mean_est["count"] += x.shape[0]
                    var_est["count"] += x.shape[0]

                    # 执行BN前向
                    return nn.batch_norm(
                        x,
                        running_mean=batch_mean,
                        running_var=batch_var,
                        weight=bn.weight,
                        bias=bn.bias,
                        training=False,
                        momentum=0.0,
                        eps=bn.eps,
                    )
                return lambda_forward

            m.execute = new_forward(m, bn_mean[name], bn_var[name])

    if not bn_mean:
        return

    # 前向计算收集统计量
    tmp_model.eval()
    with jt.no_grad():
        for images in tqdm(data_loader, desc="reset bn", disable=not progress_bar):
            if isinstance(images, (list, tuple)):
                images = images[0]  # 处理DataLoader返回的(batch, label)情况
            images = images.cuda() if jt.has_cuda else images
            tmp_model(images)

    # 同步统计量（如果启用分布式）
    if sync and jt.in_mpi:
        for name in bn_mean:
            bn_mean[name]["sum"] = jt.mpi.all_reduce(bn_mean[name]["sum"], "sum")
            bn_var[name]["sum"] = jt.mpi.all_reduce(bn_var[name]["sum"], "sum")
            bn_mean[name]["count"] = jt.mpi.all_reduce(bn_mean[name]["count"], "sum")
            bn_var[name]["count"] = jt.mpi.all_reduce(bn_var[name]["count"], "sum")

    # 更新模型BN参数
    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name]["count"] > 0:
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = bn_mean[name]["sum"] / bn_mean[name]["count"]
                m.running_var = bn_var[name]["sum"] / bn_var[name]["count"]

def set_norm_eps(model: nn.Module, eps: Optional[float] = None) -> None:
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
            if eps is not None:
                m.eps = eps