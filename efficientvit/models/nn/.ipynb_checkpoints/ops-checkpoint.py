from typing import Any, Optional, Tuple, Dict, List,Union
import jittor as jt
from jittor import nn
from jittor import Module
from jittor import Function

__all__ = [
    "ConvLayer",
    "InterpolateConvUpSampleLayer",
    "UpSampleLayer",
    "ConvPixelUnshuffleDownSampleLayer",
    "PixelUnshuffleChannelAveragingDownSampleLayer",
    "ConvPixelShuffleUpSampleLayer",
    "ChannelDuplicatingPixelUnshuffleUpSampleLayer",
    "LinearLayer",
    "IdentityLayer",
    "DSConv",
    "MBConv",
    "FusedMBConv",
    "ResBlock",
    "LiteMLA",
    "EfficientViTBlock",
    "ResidualBlock",
    "DAGBlock",
    "OpSequential",
]

# Helper functions (需要实现这些辅助函数)
def get_same_padding(kernel_size):
    return kernel_size // 2

def val2list(x, repeat_time=1):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x] * repeat_time

def val2tuple(x, repeat_time=1):
    if isinstance(x, tuple):
        return x
    return (x,) * repeat_time

def list_sum(x):
    return sum(x) if x else 0

def build_act(act_func):
    if act_func == "relu":
        return nn.ReLU()
    elif act_func == "relu6":
        return nn.ReLU6()
    elif act_func == "hswish":
        class Hardswish(nn.Module):
            def execute(self, x):
                return x * nn.relu6(x + 3) / 6
        return Hardswish()
    elif act_func == "silu":
        return nn.SiLU()
    return None

def build_norm(norm, num_features):
    if norm == "bn2d":
        return nn.BatchNorm2d(num_features)
    elif norm == "ln2d":
        return nn.LayerNorm(num_features)
    return None

def resize(x, size=None, scale_factor=None, mode="bicubic", align_corners=False):
    if size is not None:
        return nn.interpolate(x, size=size, mode=mode, align_corners=align_corners)
    else:
        return nn.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

#################################################################################
#                             Basic Layers                                      #
#################################################################################

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn",
        act_func="relu",
    ):
        super().__init__()
        
        padding = get_same_padding(kernel_size) * dilation
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias
        )
        self.norm = build_norm(norm, out_channels)
        self.act = build_act(act_func)

    def execute(self, x: jt.Var) -> jt.Var:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: Optional[int | tuple[int, int] | list[int]] = None,
        factor=2,
        align_corners=False,
    ):
        super().__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def execute(self, x: jt.Var) -> jt.Var:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return nn.resize(
            x,
            size=self.size,
            scale_factor=self.factor,
            mode=self.mode,
            align_corners=self.align_corners
        )

class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.conv(x)
        return nn.pixel_unshuffle(x, self.factor)

class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def execute(self, x: jt.Var) -> jt.Var:
        x = nn.pixel_unshuffle(x, self.factor)
        B, C, H, W = x.shape
        x = x.reshape(B, self.out_channels, self.group_size, H, W)
        return x.mean(dim=2)

class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.conv(x)
        return nn.pixel_shuffle(x, self.factor)

class InterpolateConvUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = nn.interpolate(x, scale_factor=self.factor, mode=self.mode)
        return self.conv(x)

class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def execute(self, x: jt.Var) -> jt.Var:
        x = x.repeat(1, self.repeats, 1, 1)
        return nn.pixel_shuffle(x, self.factor)

class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.norm = build_norm(norm, out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: jt.Var) -> jt.Var:
        if x.ndim > 2:
            x = x.flatten(1)
        return x

    def execute(self, x: jt.Var) -> jt.Var:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class IdentityLayer(nn.Module):
    def execute(self, x: jt.Var) -> jt.Var:
        return x

#################################################################################
#                             Basic Blocks                                      #
#################################################################################
class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn", "bn"),
        act_func=("relu6", None),
    ):
        super().__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn", "bn", "bn"),
        act_func=("relu6", "relu6", None),
    ):
        super().__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn", "bn"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x

class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=(None, None, "ln"),
        act_func=("silu", "silu", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.glu_act = build_act(act_func[1])
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size,
            stride=stride,
            groups=mid_channels * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=None,
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act_func=act_func[2],
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        
        x, gate = jt.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate
        
        x = self.point_conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn", "bn"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class LiteMLA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn"),
        act_func=(None, None),
        kernel_func="relu",
        scales: Tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        
        # Jittor的ModuleList需要显式注册
        self.aggreg = nn.ModuleList()
        for scale in scales:
            seq = nn.Sequential(
                nn.Conv2d(
                    3 * total_dim,
                    3 * total_dim,
                    scale,
                    padding=get_same_padding(scale),
                    groups=3 * total_dim,
                    bias=use_bias[0],
                ),
                nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
            )
            self.aggreg.append(seq)

        self.kernel_func = build_act(kernel_func)
        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def relu_linear_att(self, qkv: jt.Var) -> jt.Var:
        B, _, H, W = qkv.shape
        # 1. 更高效的重塑和切片操作
        qkv = qkv.reshape((B, -1, 3 * self.dim, H * W))
        q, k, v = (
            qkv[:, :, :self.dim],
            qkv[:, :, self.dim:2*self.dim],
            qkv[:, :, 2*self.dim:],
        )
        # 2. 使用原地操作减少内存分配
        with jt.no_grad():
            q = self.kernel_func(q)
            k = self.kernel_func(k)
        # 3. 优化矩阵乘法顺序
        trans_k = k.transpose(-1, -2)
        v_pad = jt.nn.pad(v, [0, 0, 0, 1], mode="constant", value=1)
        # 4. 使用融合操作
        vk = jt.bmm(v_pad, trans_k)
        out = jt.bmm(vk, q)
        # 5. 安全除法
        denominator = out[:, :, -1:] + self.eps
        out = out[:, :, :-1].divide(denominator)
        # 6. 最终形状调整
        return out.reshape([B, -1, H, W])

    def relu_quadratic_att(self, qkv: jt.Var) -> jt.Var:
        B, _, H, W = qkv.shape

        qkv = qkv.reshape((B, -1, 3 * self.dim, H * W))
        q, k, v = (
            qkv[:, :, :self.dim],
            qkv[:, :, self.dim:2*self.dim],
            qkv[:, :, 2*self.dim:],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = jt.matmul(k.transpose(-1, -2), q)
        att_map = att_map / (jt.sum(att_map, dim=2, keepdims=True) + self.eps)
        out = jt.matmul(v, att_map)
        
        return out.reshape((B, -1, H, W))

    def execute(self, x: jt.Var) -> jt.Var:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
            
        qkv = jt.concat(multi_scale_qkv, dim=1)
        H, W = qkv.shape[-2:]
        
        if H * W > self.dim:
            out = self.relu_linear_att(qkv)
        else:
            out = self.relu_quadratic_att(qkv)
            
        return self.proj(out)

class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales: Tuple[int, ...] = (5,),
        norm: str = "bn",
        act_func: str = "hswish",
        context_module: str = "LiteMLA",
        local_module: str = "MBConv",
    ):
        super().__init__()
        
        if context_module == "LiteMLA":
            self.context_module = ResidualBlock(
                LiteMLA(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    heads_ratio=heads_ratio,
                    dim=dim,
                    norm=(None, norm),
                    scales=scales,
                ),
                IdentityLayer(),
            )
        else:
            raise ValueError(f"Unsupported context_module: {context_module}")
            
        if local_module == "MBConv":
            self.local_module = ResidualBlock(
                MBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                IdentityLayer(),
            )
        elif local_module == "GLUMBConv":
            self.local_module = ResidualBlock(
                GLUMBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                IdentityLayer(),
            )
        else:
            raise NotImplementedError(f"Unsupported local_module: {local_module}")

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.context_module(x)
        x = self.local_module(x)
        return x

#################################################################################
#                             Functional Blocks                                 #
#################################################################################   
class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def execute_main(self, x: jt.Var) -> jt.Var:
        if self.pre_norm is None:
            return self.main(x) if self.main else x
        else:
            return self.main(self.pre_norm(x))

    def execute(self, x: jt.Var) -> jt.Var:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.execute_main(x)
        else:
            res = self.execute_main(x) + (self.shortcut(x) if self.shortcut else 0)
            if self.post_act:
                res = self.post_act(res)
        return res

class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: Dict[str, nn.Module],
        merge: str,
        post_input: Optional[nn.Module],
        middle: nn.Module,
        outputs: Dict[str, nn.Module],
    ):
        super().__init__()
        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input
        self.middle = middle
        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def execute(self, feature_dict: Dict[str, jt.Var]) -> Dict[str, jt.Var]:
        # 处理输入分支
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        
        # 合并策略
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = jt.concat(feat, dim=1)
        else:
            raise NotImplementedError(f"Merge method {self.merge} not supported")
        
        # 后处理
        if self.post_input:
            feat = self.post_input(feat)
        
        # 中间处理
        feat = self.middle(feat)
        
        # 输出分支
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
            
        return feature_dict

class OpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super().__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def execute(self, x: jt.Var) -> jt.Var:
        for op in self.op_list:
            x = op(x)
        return x