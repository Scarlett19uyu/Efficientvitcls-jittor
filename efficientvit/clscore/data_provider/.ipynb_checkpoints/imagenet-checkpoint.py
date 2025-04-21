import copy
import math
import os
from typing import Any, Optional

import jittor as jt
import jittor.transform as transforms
from jittor.dataset import ImageFolder
from PIL import Image

from efficientvit.apps.data_provider import DataProvider
from efficientvit.apps.data_provider.augment import RandAug
from efficientvit.apps.data_provider.random_resolution import MyRandomResizedCrop, get_interpolate
from efficientvit.apps.utils import partial_update_config
from efficientvit.models.utils import val2list

__all__ = ["ImageNetDataProvider"]


class FixedImageFolder(jt.dataset.ImageFolder):
    def __init__(self, root, transform=None, size=224):
        super().__init__(root, transform=transform)
        self.size = size if isinstance(size, int) else size[0]
        self.default_class = 0
        
        # 确保兼容性：同时设置 samples 和 imgs 属性（Jittor/PyTorch 双兼容）
        self.samples = self.imgs  # 添加 samples 属性（与 PyTorch 对齐）
        self.img_paths = [item[0] for item in self.imgs]  # 可选：方便路径访问

    def __getitem__(self, idx):
        try:
            path, target = self.imgs[idx]
            
            # 自动修复-checkpoint后缀问题
            if "-checkpoint.JPEG" in path:
                corrected_path = path.replace("-checkpoint.JPEG", ".JPEG")
                if os.path.exists(corrected_path):
                    path = corrected_path
                    self.imgs[idx] = (corrected_path, target)  # 更新原始数据
                    self.samples[idx] = (corrected_path, target)  # 同步更新 samples
            
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                
        except Exception as e:
            print(f"Warning: Failed to load {path}, using random image. Error: {str(e)}")
            dummy_img = jt.randn(3, self.size, self.size)
            return dummy_img, self.default_class
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class ImageNetDataProvider(DataProvider):
    name = "imagenet"
    data_dir = "autodl-tmp/imagenet"
    n_classes = 100
    _DEFAULT_RRC_CONFIG = {
        "train_interpolate": "bilinear",  # !
        "test_interpolate": "bicubic",
        "test_crop_ratio": 1.0,
    }

    def __init__(
        self,
        data_dir: Optional[str] = None,
        rrc_config: Optional[dict] = None,
        data_aug: Optional[dict | list[dict]] = None,
        train_batch_size=128,
        test_batch_size=128,
        valid_size: Optional[int | float] = None,
        n_worker=8,
        image_size: int | list[int] = 224,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        train_ratio: Optional[float] = None,
        drop_last: bool = False,
    ):
        self.data_dir = self.data_dir if data_dir is None else data_dir
        self.rrc_config = partial_update_config(
            copy.deepcopy(self._DEFAULT_RRC_CONFIG),
            {} if rrc_config is None else rrc_config,
        )
        self.data_aug = data_aug

        super().__init__(
            train_batch_size,
            test_batch_size,
            valid_size,
            n_worker,
            image_size,
            num_replicas,
            rank,
            train_ratio,
            drop_last,
        )

    def build_valid_transform(self, image_size: Optional[tuple[int, int]] = None) -> Any:
        image_size = (self.active_image_size if image_size is None else image_size)[0]
        crop_size = int(math.ceil(image_size / self.rrc_config["test_crop_ratio"]))
        
        return transforms.Compose([
            transforms.Resize(
                size=crop_size,
                mode=get_interpolate(self.rrc_config["test_interpolate"])
            ),
            transforms.CenterCrop(size=image_size),
            transforms.ImageNormalize(mean=self.mean_std["mean"], std=self.mean_std["std"])
        ])

    def build_train_transform(self, image_size: Optional[tuple[int, int]] = None) -> Any:
        image_size = self.image_size if image_size is None else image_size

        train_transforms = [
            MyRandomResizedCrop(interpolation=self.rrc_config["train_interpolate"]),
            transforms.RandomHorizontalFlip(),
        ]

        post_aug = []
        if self.data_aug is not None:
            for aug_op in val2list(self.data_aug):
                if aug_op["name"] == "randaug":
                    data_aug = RandAug(aug_op, mean=self.mean_std["mean"])
                elif aug_op["name"] == "erase":
                    from efficientvit.apps.data_provider.augment import RandomErasing
                    random_erase = RandomErasing(aug_op["p"])
                    post_aug.append(random_erase)
                    data_aug = None
                else:
                    raise NotImplementedError
                if data_aug is not None:
                    train_transforms.append(data_aug)
        
        train_transforms.extend([
            transforms.ImageNormalize(mean=self.mean_std["mean"], std=self.mean_std["std"]),
            *post_aug
        ])
        
        return transforms.Compose(train_transforms)

    def build_datasets(self) -> tuple[Any, Any, Any]:
        train_transform = self.build_train_transform()
        valid_transform = self.build_valid_transform()
        image_size = self.image_size if isinstance(self.image_size, int) else self.image_size[0]

        # 创建数据集实例
        train_dataset = FixedImageFolder(
            os.path.join(self.data_dir, "train"), 
            transform=train_transform,
            size=image_size
        )
        test_dataset = FixedImageFolder(
            os.path.join(self.data_dir, "val"),
            transform=valid_transform,
            size=image_size
        )

        train_dataset, val_dataset = self.sample_val_dataset(train_dataset, valid_transform)
        return train_dataset, val_dataset, test_dataset