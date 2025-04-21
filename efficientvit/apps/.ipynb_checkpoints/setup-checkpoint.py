import os
import time
from copy import deepcopy
from typing import Optional
import jittor as jt
from jittor import nn

from efficientvit.apps.data_provider import DataProvider
from efficientvit.apps.trainer.run_config import RunConfig
from efficientvit.apps.utils import (
    dump_config,
    init_modules,
    load_config,
    partial_update_config,
    zero_last_gamma,
)

__all__ = [
    "save_exp_config",
    "setup_dist_env",
    "setup_seed",
    "setup_exp_config",
    "setup_data_provider",
    "setup_run_config",
    "init_model",
]

def save_exp_config(exp_config: dict, path: str, name="config.yaml") -> None:
    dump_config(exp_config, os.path.join(path, name))

def setup_dist_env(gpu: Optional[str] = None) -> None:
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # Jittor自动初始化CUDA，无需手动设置
    jt.flags.use_cuda = 1  # 启用CUDA加速

def setup_seed(manual_seed: int, resume: bool) -> None:
    if resume:
        manual_seed = int(time.time())
    # Jittor设置随机种子
    jt.set_global_seed(manual_seed)

def setup_exp_config(config_path: str, recursive=True, opt_args: Optional[dict] = None) -> dict:
    if not os.path.isfile(config_path):
        raise ValueError(config_path)

    fpaths = [config_path]
    if recursive:
        extension = os.path.splitext(config_path)[1]
        while os.path.dirname(config_path) != config_path:
            config_path = os.path.dirname(config_path)
            fpath = os.path.join(config_path, "default" + extension)
            if os.path.isfile(fpath):
                fpaths.append(fpath)
        fpaths = fpaths[::-1]

    default_config = load_config(fpaths[0])
    exp_config = deepcopy(default_config)
    for fpath in fpaths[1:]:
        partial_update_config(exp_config, load_config(fpath))
    if opt_args is not None:
        partial_update_config(exp_config, opt_args)

    return exp_config

def build_kwargs_from_config(config, target_class):
    """自动从配置中提取目标类需要的参数"""
    import inspect
    valid_params = inspect.signature(target_class.__init__).parameters
    return {k: v for k, v in config.items() if k in valid_params}

def setup_data_provider(
    exp_config: dict, 
    data_provider_classes: list[type[DataProvider]], 
    is_distributed: bool = True
) -> DataProvider:
    dp_config = exp_config["data_provider"]
    # Jittor分布式需要特殊处理，这里暂时简化
    dp_config["num_replicas"] = 1  # 替换get_dist_size()
    dp_config["rank"] = 0         # 替换get_dist_rank()
    dp_config["test_batch_size"] = dp_config.get("test_batch_size", None)
    dp_config["test_batch_size"] = (
        dp_config["base_batch_size"] * 2 if dp_config["test_batch_size"] is None else dp_config["test_batch_size"]
    )
    dp_config["batch_size"] = dp_config["train_batch_size"] = dp_config["base_batch_size"]

    data_provider_lookup = {provider.name: provider for provider in data_provider_classes}
    data_provider_class = data_provider_lookup[dp_config["dataset"]]

    data_provider_kwargs = build_kwargs_from_config(dp_config, data_provider_class)
    data_provider = data_provider_class(**data_provider_kwargs)
    return data_provider

def setup_run_config(exp_config: dict, run_config_cls: type[RunConfig]) -> RunConfig:
    # Jittor分布式需要特殊处理，这里简化
    exp_config["run_config"]["init_lr"] = exp_config["run_config"]["base_lr"] * 1  # 替换get_dist_size()

    run_config = run_config_cls(**exp_config["run_config"])
    return run_config

def init_model(
    network: nn.Module,
    init_from: Optional[str] = None,
    backbone_init_from: Optional[str] = None,
    rand_init="trunc_normal",
    last_gamma=None,
) -> None:
    # initialization
    init_modules(network, init_type=rand_init)
    # zero gamma of last bn in each block
    if last_gamma is not None:
        zero_last_gamma(network, last_gamma)

    # load weight
    if init_from is not None and os.path.isfile(init_from):
        network.load_state_dict(jt.load(init_from))  # Jittor的加载方式
        print(f"Loaded init from {init_from}")
    elif backbone_init_from is not None and os.path.isfile(backbone_init_from):
        network.backbone.load_state_dict(jt.load(backbone_init_from))
        print(f"Loaded backbone init from {backbone_init_from}")
    else:
        print(f"Random init ({rand_init}) with last gamma {last_gamma}")