import os
from typing import Any, Optional
import jittor as jt
from jittor import nn, Module
from efficientvit.apps.data_provider import DataProvider, parse_image_size
from efficientvit.apps.trainer.run_config import RunConfig
from efficientvit.apps.utils import EMA, dist_barrier, get_dist_local_rank, is_master
from efficientvit.models.nn.norm import reset_bn
from efficientvit.models.utils import is_parallel, load_state_dict_from_file

__all__ = ["Trainer"]

class Trainer:
    def __init__(self, path: str, model: Module, data_provider: DataProvider):
        self.path = os.path.realpath(os.path.expanduser(path))
        self.model = model
        self.data_provider = data_provider

        self.ema = None
        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        self.logs_path = os.path.join(self.path, "logs")
        for path in [self.path, self.checkpoint_path, self.logs_path]:
            os.makedirs(path, exist_ok=True)

        self.best_val = 0.0
        self.start_epoch = 0

    @property
    def network(self) -> Module:
        return self.model.module if is_parallel(self.model) else self.model

    @property
    def eval_network(self) -> Module:
        if self.ema is None:
            model = self.model
        else:
            model = self.ema.shadows
        model = model.module if is_parallel(model) else model
        return model

    def write_log(self, log_str, prefix="valid", print_log=True, mode="a") -> None:
        if is_master():
            with open(os.path.join(self.logs_path, f"{prefix}.log"), mode) as fout:
                fout.write(log_str + "\n")
            if print_log:
                print(log_str)

    def save_model(
        self,
        checkpoint=None,
        only_state_dict=True,
        epoch=0,
        model_name=None,
    ) -> None:
        if is_master():
            if checkpoint is None:
                if only_state_dict:
                    checkpoint = {"state_dict": self.network.state_dict()}
                else:
                    checkpoint = {
                        "state_dict": self.network.state_dict(),
                        "epoch": epoch,
                        "best_val": self.best_val,
                        "optimizer": self.optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                        "ema": self.ema.state_dict() if self.ema is not None else None,
                    }

            model_name = "checkpoint.pkl" if model_name is None else model_name
            latest_fname = os.path.join(self.checkpoint_path, "latest.txt")
            model_path = os.path.join(self.checkpoint_path, model_name)
            
            with open(latest_fname, "w") as fout:
                fout.write(model_path + "\n")
            jt.save(checkpoint, model_path)

    def load_model(self, model_fname=None) -> None:
        latest_fname = os.path.join(self.checkpoint_path, "latest.txt")
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname = fin.readline().strip()
                
        try:
            if model_fname is None:
                model_fname = f"{self.checkpoint_path}/checkpoint.pkl"
            elif not os.path.exists(model_fname):
                model_fname = os.path.join(self.checkpoint_path, os.path.basename(model_fname))
                if not os.path.exists(model_fname):
                    model_fname = f"{self.checkpoint_path}/checkpoint.pkl"
                    
            print(f"=> loading checkpoint {model_fname}")
            checkpoint = load_state_dict_from_file(model_fname, False)
        except Exception as e:
            self.write_log(f"fail to load checkpoint from {self.checkpoint_path}: {str(e)}")
            return

        # load checkpoint
        self.network.load_state_dict(checkpoint["state_dict"], strict=False)
        log = []
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
            self.run_config.update_global_step(self.start_epoch)
            log.append(f"epoch={self.start_epoch - 1}")
        if "best_val" in checkpoint:
            self.best_val = checkpoint["best_val"]
            log.append(f"best_val={self.best_val:.2f}")
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            log.append("optimizer")
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            log.append("lr_scheduler")
        if "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
            log.append("ema")
            
        self.write_log("Loaded: " + ", ".join(log))

    """ validate """
    def reset_bn(
        self,
        network: Optional[Module] = None,
        subset_size: int = 16000,
        subset_batch_size: int = 100,
        data_loader=None,
        progress_bar=False,
    ) -> None:
        network = self.network if network is None else network
        if data_loader is None:
            data_loader = []
            for data in self.data_provider.build_sub_train_loader(subset_size, subset_batch_size):
                if isinstance(data, (list, tuple)):
                    data_loader.append(data[0])
                elif isinstance(data, dict):
                    data_loader.append(data["data"])
                elif isinstance(data, jt.Var):
                    data_loader.append(data)
                else:
                    raise NotImplementedError

        network.eval()
        reset_bn(
            network,
            data_loader,
            sync=True,
            progress_bar=progress_bar,
        )

    def _validate(self, model: Module, data_loader, epoch: int) -> dict[str, Any]:
        raise NotImplementedError

    def validate(self, model=None, data_loader=None, is_test=True, epoch=0) -> dict[str, Any]:
        model = self.eval_network if model is None else model
        if data_loader is None:
            data_loader = self.data_provider.test if is_test else self.data_provider.valid

        model.eval()
        return self._validate(model, data_loader, epoch)

    def multires_validate(
        self,
        model=None,
        data_loader=None,
        is_test=True,
        epoch=0,
        eval_image_size=None,
    ) -> dict[str, dict[str, Any]]:
        eval_image_size = self.run_config.eval_image_size if eval_image_size is None else eval_image_size
        eval_image_size = self.data_provider.image_size if eval_image_size is None else eval_image_size
        model = self.eval_network if model is None else model

        if not isinstance(eval_image_size, list):
            eval_image_size = [eval_image_size]

        output_dict = {}
        for r in eval_image_size:
            self.data_provider.assign_active_image_size(parse_image_size(r))
            if self.run_config.reset_bn:
                self.reset_bn(
                    network=model,
                    subset_size=self.run_config.reset_bn_size,
                    subset_batch_size=self.run_config.reset_bn_batch_size,
                    progress_bar=True,
                )
            output_dict[f"r{r}"] = self.validate(model, data_loader, is_test, epoch)
        return output_dict

    """ training """
    def prep_for_training(self, run_config: RunConfig, ema_decay: Optional[float] = None, amp: str = "fp32") -> None:
        self.run_config = run_config
        
        # Jittor分布式训练设置
        if jt.in_mpi:
            self.model = nn.DataParallel(self.model)
        
        self.run_config.global_step = 0
        self.run_config.batch_per_epoch = len(self.data_provider.train)
        assert self.run_config.batch_per_epoch > 0, "Training set is empty"

        # 构建优化器
        self.optimizer, self.lr_scheduler = self.run_config.build_optimizer(self.model)

        if ema_decay is not None:
            self.ema = EMA(self.network, ema_decay)

        # AMP设置
        self.amp = amp
        if amp != "fp32":
            jt.flags.amp_level = 3  # O3级别自动混合精度

    @property
    def enable_amp(self) -> bool:
        return self.amp != "fp32"

    def sync_model(self):
        print("Sync model")
        self.save_model(model_name="sync.pkl")
        dist_barrier()
        checkpoint = jt.load(os.path.join(self.checkpoint_path, "sync.pkl"))
        dist_barrier()
        if is_master():
            os.remove(os.path.join(self.checkpoint_path, "sync.pkl"))
        dist_barrier()

        # 加载checkpoint
        state_dict = checkpoint["state_dict"]
        if hasattr(self.network, 'load_state_dict'):
            # Jittor 版本的加载方式
            self.network.load_state_dict(state_dict)
        else:
            # 兼容性处理
            for k, v in state_dict.items():
                if k in self.network.state_dict():
                    self.network.state_dict()[k].assign(v)


        
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])

    def before_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        # Jittor自动处理设备转移，无需手动操作
        return feed_dict

    def run_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def after_step(self) -> None:
    # Jittor不需要显式unscale，混合精度由Jittor自动处理
    
    # 梯度裁剪 (Jittor实现)
        if getattr(self.run_config, 'grad_clip', None) is not None:
            for p in self.model.parameters():
                try:
                # Jittor使用opt_grad获取梯度，需要传入optimizer
                    grad = p.opt_grad(self.optimizer)
                    if grad is not None:
                    # 实现梯度值裁剪
                        clip_value = self.run_config.grad_clip
                        clipped_grad = jt.clamp(grad, -clip_value, clip_value)
                        p.opt_grad(self.optimizer).assign(clipped_grad)
                except RuntimeError as e:
                    if "not managed by this optimizer" in str(e):
                        continue
                    raise
    
    # 参数更新 (Jittor的step不需要scaler包装)
        self.optimizer.step()
    
    # Jittor的混合精度自动更新，不需要scaler.update()
    
    # 学习率调度
        self.lr_scheduler.step()
        self.run_config.step()
    
    # 更新EMA模型
        if self.ema is not None:
            self.ema.step(self.network, self.run_config.global_step)

    def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
        raise NotImplementedError

    def train_one_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        self.data_provider.set_epoch(epoch)
        return self._train_one_epoch(epoch)

    def train(self) -> None:
        raise NotImplementedError