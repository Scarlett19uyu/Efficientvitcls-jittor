import copy
import warnings
import math
import os
from typing import Any, Optional, Tuple, List, Union

import jittor as jt
from jittor.dataset import Dataset, DataLoader
import numpy as np

from efficientvit.apps.data_provider.random_resolution import RRSController
from efficientvit.models.utils import val2tuple

__all__ = ["parse_image_size", "random_drop_data", "DataProvider", "DistributedSampler"]

class DistributedSampler:
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            num_replicas = jt.world_size
        if rank is None:
            rank = jt.rank
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            jt.set_global_seed(self.epoch)
            indices = jt.randperm(len(self.dataset)).numpy().tolist()
        else:
            indices = list(range(len(self.dataset)))

        # 填充使长度一致
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # 获取本rank对应的数据
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def parse_image_size(size: Union[int, str]) -> Tuple[int, int]:
    if isinstance(size, str):
        size = [int(val) for val in size.split("-")]
        return size[0], size[1]
    else:
        return val2tuple(size, 2)

def random_drop_data(dataset, drop_size: int, seed: int, keys=("samples",)):
    jt.set_global_seed(seed)
    rand_indexes = jt.randperm(len(dataset)).numpy().tolist()

    dropped_indexes = rand_indexes[:drop_size]
    remaining_indexes = rand_indexes[drop_size:]

    dropped_dataset = copy.deepcopy(dataset)
    for key in keys:
        setattr(dropped_dataset, key, [getattr(dropped_dataset, key)[idx] for idx in dropped_indexes])
        setattr(dataset, key, [getattr(dataset, key)[idx] for idx in remaining_indexes])
    return dataset, dropped_dataset

class DataProvider:
    data_keys = ("samples",)
    mean_std = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    SUB_SEED = 937162211
    VALID_SEED = 2147483647

    name: str

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: Optional[int],
        valid_size: Optional[Union[int, float]],
        n_worker: int,
        image_size: Union[int, List[int], str, List[str]],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        train_ratio: Optional[float] = None,
        drop_last: bool = False,
    ):
        warnings.filterwarnings("ignore")

        # 初始化基本参数
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size if test_batch_size is not None else train_batch_size
        self.valid_size = valid_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.train_ratio = 0.1

        # 处理图像尺寸
        if isinstance(image_size, list):
            self.image_size = [parse_image_size(size) for size in image_size]
            self.image_size.sort()
            RRSController.IMAGE_SIZE_LIST = copy.deepcopy(self.image_size)
            self.active_image_size = RRSController.ACTIVE_SIZE = self.image_size[-1]
        else:
            self.image_size = parse_image_size(image_size)
            RRSController.IMAGE_SIZE_LIST = [self.image_size]
            self.active_image_size = RRSController.ACTIVE_SIZE = self.image_size

        # 构建数据集
        train_dataset, val_dataset, test_dataset = self.build_datasets()

        # 处理训练集子集
        if train_ratio is not None and train_ratio < 1.0:
            assert 0 < train_ratio < 1
            _, train_dataset = random_drop_data(
                train_dataset,
                int((1 - train_ratio) * len(train_dataset)),
                self.SUB_SEED,
                self.data_keys,
            )

        # 构建数据加载器
        self.train = self.build_dataloader(
            train_dataset, self.train_batch_size, n_worker, drop_last=drop_last, train=True
        )
        self.valid = self.build_dataloader(
            val_dataset, self.test_batch_size, n_worker, drop_last=False, train=False
        )
        self.test = self.build_dataloader(
            test_dataset, self.test_batch_size, n_worker, drop_last=False, train=False
        )
        
        if self.valid is None:
            self.valid = self.test
            
        self.sub_train = None

    @property
    def data_shape(self) -> Tuple[int, ...]:
        return 3, self.active_image_size[0], self.active_image_size[1]

    def build_valid_transform(self, image_size: Optional[Tuple[int, int]] = None) -> Any:
        raise NotImplementedError

    def build_train_transform(self, image_size: Optional[Tuple[int, int]] = None) -> Any:
        raise NotImplementedError

    def build_datasets(self) -> Tuple[Any, Any, Any]:
        raise NotImplementedError

    def build_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        n_worker: int,
        drop_last: bool,
        train: bool
    ) -> DataLoader:
        if dataset is None:
            return None
            
        # 随机分辨率数据加载器
        if isinstance(self.image_size, list) and train:
            from efficientvit.apps.data_provider.random_resolution._data_loader import RRSDataLoader
            dataloader = RRSDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=train and (self.num_replicas is None or self.num_replicas <= 1),
                num_workers=n_worker,
                drop_last=drop_last
            )
        else:
            # 标准Jittor数据加载器
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=train and (self.num_replicas is None or self.num_replicas <= 1),
                num_workers=n_worker,
                drop_last=drop_last
            )
        
        # 分布式处理
        if self.num_replicas is not None and self.num_replicas > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=train
            )
            dataloader.sampler = sampler
            
        return dataloader

    def set_epoch(self, epoch: int) -> None:
        RRSController.set_epoch(epoch, len(self.train))
        if hasattr(self.train, 'sampler') and isinstance(self.train.sampler, DistributedSampler):
            self.train.sampler.set_epoch(epoch)

    def assign_active_image_size(self, new_size: Union[int, Tuple[int, int]]) -> None:
        self.active_image_size = val2tuple(new_size, 2)
        new_transform = self.build_valid_transform(self.active_image_size)
        
        if hasattr(self.valid, 'dataset'):
            self.valid.dataset.transform = new_transform
        if hasattr(self.test, 'dataset'):
            self.test.dataset.transform = new_transform

    def sample_val_dataset(self, train_dataset, valid_transform) -> Tuple[Any, Any]:
        if self.valid_size is not None:
            if 0 < self.valid_size < 1:
                valid_size = int(self.valid_size * len(train_dataset))
            else:
                valid_size = int(self.valid_size)
                
            train_dataset, val_dataset = random_drop_data(
                train_dataset,
                valid_size,
                self.VALID_SEED,
                self.data_keys,
            )
            val_dataset.transform = valid_transform
        else:
            val_dataset = None
            
        return train_dataset, val_dataset

    def build_sub_train_loader(self, n_samples: int, batch_size: int) -> Any:
        if self.sub_train is None:
            self.sub_train = {}

        if self.active_image_size in self.sub_train:
            return self.sub_train[self.active_image_size]

        # 构建子训练集
        train_dataset = copy.deepcopy(self.train.dataset)
        if n_samples < len(train_dataset):
            _, train_dataset = random_drop_data(
                train_dataset,
                len(train_dataset) - n_samples,
                self.SUB_SEED,
                self.data_keys,
            )

        RRSController.ACTIVE_SIZE = self.active_image_size
        train_dataset.transform = self.build_train_transform(image_size=self.active_image_size)
        
        data_loader = self.build_dataloader(
            train_dataset,
            batch_size,
            getattr(self.train, 'num_workers', 4),
            drop_last=True,
            train=False
        )

        # 预取数据
        self.sub_train[self.active_image_size] = [
            data for data in data_loader 
            for _ in range(max(1, n_samples // len(train_dataset)))
        ]

        return self.sub_train[self.active_image_size]