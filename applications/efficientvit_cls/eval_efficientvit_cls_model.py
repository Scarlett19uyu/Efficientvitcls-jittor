import argparse
import os
import sys
import jittor as jt
from jittor import nn
from jittor.dataset import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)


from efficientvit_jittor.apps import setup_jt as setup
from efficientvit_jittor.apps.utils import dump_config, parse_unknown_args
from efficientvit_jittor.cls_model_zoo import create_efficientvit_cls_model
from efficientvit_jittor.clscore.data_provider import ImageNetDataProvider
from efficientvit_jittor.clscore.trainer import ClsRunConfig, ClsTrainer
from efficientvit_jittor.models.nn.drop import apply_drop_func

def main():
    # 参数解析保持不变
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--path", type=str, metavar="DIR", help="run directory")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", type=str, choices=["fp32", "fp16"], default="fp32")  # Jittor目前主要支持fp16
    
    # 初始化参数
    parser.add_argument("--rand_init", type=str, default="trunc_normal@0.02")
    parser.add_argument("--last_gamma", type=float, default=0)
    parser.add_argument("--auto_restart_thresh", type=float, default=1.0)
    parser.add_argument("--save_freq", type=int, default=1)
    
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    # Jittor初始化
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    if args.gpu:
        jt.flags.gpu_id = args.gpu
    jt.set_global_seed(args.manual_seed)

    # 创建输出目录
    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))

    # 加载配置
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)
    setup.save_exp_config(config, args.path)

    # 数据加载器 (需要适配Jittor的Dataset)
    data_provider = ImageNetDataProvider(config, batch_size=config["data_provider"]["batch_size"])
    train_loader = data_provider.get_train_loader()
    val_loader = data_provider.get_val_loader()

    # 模型创建
    model = create_efficientvit_cls_model(
        config["net_config"]["name"], 
        pretrained=False, 
        dropout=config["net_config"]["dropout"]
    )
    apply_drop_func(model.backbone.stages, config["backbone_drop"])

    # 优化器设置
    optimizer = nn.AdamW(
        model.parameters(), 
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"]
    )
    
    # 学习率调度器
    lr_scheduler = nn.CosineAnnealingLR(
        optimizer, 
        T_max=config["run_config"]["n_epochs"],
        eta_min=config["optimizer"]["min_lr"]
    )

    # 损失函数
    if config.get("bce", False):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_val = 0.0
    for epoch in range(config["run_config"]["n_epochs"]):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 前向传播
            with jt.flag_scope(amp_level=args.amp):  # Jittor的混合精度
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.step(loss)
            
            # 学习率更新
            lr_scheduler.step()
            
            # 打印训练信息
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # 验证
        model.eval()
        total_correct = 0
        total_samples = 0
        with jt.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.shape[0]
        
        val_acc = total_correct / total_samples
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        # 保存模型 (不实现自动回滚)
        if val_acc > best_val:
            best_val = val_acc
            model.save(os.path.join(args.path, f"model_best.pkl"))
        
        # 定期保存
        if epoch % args.save_freq == 0:
            model.save(os.path.join(args.path, f"checkpoint_{epoch}.pkl"))

if __name__ == "__main__":
    main()