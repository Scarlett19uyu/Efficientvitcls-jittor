import jittor as jt
from jittor import init
from jittor import nn
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.apps import setup
from efficientvit.apps.utils import dump_config, parse_unknown_args
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from efficientvit.clscore.data_provider import ImageNetDataProvider
from efficientvit.clscore.trainer import ClsRunConfig, ClsTrainer
from efficientvit.models.nn.drop import apply_drop_func

# 初始化Jittor环境
jt.flags.use_cuda = 1  # 启用CUDA


parser = argparse.ArgumentParser()
parser.add_argument("config", metavar="FILE", help="config file")
parser.add_argument("--path", type=str, metavar="DIR", help="run directory")
parser.add_argument("--gpu", type=str, default=None)
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--amp", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")

# 初始化参数
parser.add_argument("--rand_init", type=str, default="trunc_normal@0.02")
parser.add_argument("--last_gamma", type=float, default=0)

parser.add_argument("--auto_restart_thresh", type=float, default=1.0)
parser.add_argument("--save_freq", type=int, default=1)

def main():
    # 解析参数
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    # 设置随机种子
    if args.manual_seed is not None:
        jt.set_global_seed(args.manual_seed)
    
    # 设置路径并保存参数
    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))

    # 设置实验配置
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)
    setup.save_exp_config(config, args.path)

    # 设置数据提供器
    data_provider = setup.setup_data_provider(config, [ImageNetDataProvider], is_distributed=False)  # Jittor分布式需要特殊处理
    
    # 设置运行配置
    run_config = setup.setup_run_config(config, ClsRunConfig)

    # 创建模型
    model = create_efficientvit_cls_model(
        config["net_config"]["name"], 
        pretrained=False, 
        dropout=config["net_config"]["dropout"]
    )
    apply_drop_func(model.backbone.stages, config["backbone_drop"])

    # 设置训练器
    trainer = ClsTrainer(
        path=args.path,
        model=model,
        data_provider=data_provider,
        auto_restart_thresh=args.auto_restart_thresh,
    )
    
    # 模型初始化
    setup.init_model(
        trainer.network,
        rand_init=args.rand_init,
        last_gamma=args.last_gamma,
    )

    # 准备训练
    trainer.prep_for_training(run_config, config["ema_decay"], args.amp)

    # 恢复训练
    if args.resume:
        trainer.load_model()
        trainer.data_provider = setup.setup_data_provider(config, [ImageNetDataProvider], is_distributed=False)
    else:
        trainer.sync_model()

    # 开始训练
    trainer.train(save_freq=args.save_freq)

if __name__ == "__main__":
    main()