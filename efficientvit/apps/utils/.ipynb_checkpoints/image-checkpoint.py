import os
import pathlib
from typing import Any, Callable, Optional, Union, List, Tuple
import numpy as np
from PIL import Image
import jittor as jt
from jittor.dataset import Dataset

__all__ = ["load_image", "load_image_from_dir", "DMCrop", "CustomImageFolder", "ImageDataset"]

def load_image(data_path: str, mode: str = "rgb") -> Image.Image:
    """加载图像文件 (Jittor版本)"""
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return img

def load_image_from_dir(
    dir_path: str,
    suffix: Union[str, Tuple[str, ...], List[str]] = (".jpg", ".JPEG", ".png"),
    return_mode: str = "path",
    k: Optional[int] = None,
    shuffle_func: Optional[Callable] = None,
) -> Union[List, Tuple[List, List]]:
    """从目录加载图像文件 (Jittor版本)"""
    suffix = [suffix] if isinstance(suffix, str) else suffix

    file_list = []
    for dirpath, _, fnames in os.walk(dir_path):
        for fname in fnames:
            if pathlib.Path(fname).suffix not in suffix:
                continue
            image_path = os.path.join(dirpath, fname)
            file_list.append(image_path)

    if shuffle_func is not None and k is not None:
        shuffle_file_list = shuffle_func(file_list)
        file_list = shuffle_file_list or file_list
        file_list = file_list[:k]

    file_list = sorted(file_list)

    if return_mode == "path":
        return file_list
    else:
        files = []
        path_list = []
        for file_path in file_list:
            try:
                files.append(load_image(file_path))
                path_list.append(file_path)
            except Exception:
                print(f"Fail to load {file_path}")
        if return_mode == "image":
            return files
        else:
            return path_list, files

class DMCrop:
    """扩散模型使用的中心/随机裁剪 (Jittor版本)"""
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, pil_image: Image.Image) -> Image.Image:
        """执行裁剪操作"""
        image_size = self.size
        if pil_image.size == (image_size, image_size):
            return pil_image

        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])

class CustomImageFolder(Dataset):
    """自定义图像文件夹数据集 (Jittor版本)"""
    def __init__(self, root: str, transform: Optional[Callable] = None, return_dict: bool = False):
        super().__init__()
        root = os.path.expanduser(root)
        self.return_dict = return_dict
        self.transform = transform
        
        # 加载所有图像路径和标签
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        
        # 构建类别索引
        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        # 收集样本
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)
            for root_dir, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root_dir, fname)
                    self.samples.append(path)
                    self.targets.append(class_index)
        
        self.total_len = len(self.samples)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, index: int) -> Union[dict[str, Any], Tuple[Any, Any]]:
        path = self.samples[index]
        target = self.targets[index]
        
        image = load_image(path)
        if self.transform is not None:
            image = self.transform(image)
            
        if self.return_dict:
            return {
                "index": index,
                "image_path": path,
                "image": image,
                "label": target,
            }
        else:
            return image, target

class ImageDataset(Dataset):
    """通用图像数据集 (Jittor版本)"""
    def __init__(
        self,
        data_dirs: Union[str, List[str]],
        splits: Optional[Union[str, List[Optional[str]]]] = None,
        transform: Optional[Callable] = None,
        suffix: Tuple[str, ...] = (".jpg", ".JPEG", ".png"),
        pil: bool = True,
        return_dict: bool = True,
    ) -> None:
        super().__init__()
        self.data_dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs
        
        if isinstance(splits, list):
            assert len(splits) == len(self.data_dirs)
            self.splits = splits
        elif isinstance(splits, str):
            assert len(self.data_dirs) == 1
            self.splits = [splits]
        else:
            self.splits = [None for _ in range(len(self.data_dirs))]

        self.transform = transform
        self.pil = pil
        self.return_dict = return_dict

        # 加载所有图像路径
        self.samples = []
        for data_dir, split in zip(self.data_dirs, self.splits):
            if split is None:
                samples = load_image_from_dir(data_dir, suffix, return_mode="path")
            else:
                samples = []
                with open(split, "r") as fin:
                    for line in fin.readlines():
                        relative_path = line.strip()
                        full_path = os.path.join(data_dir, relative_path)
                        samples.append(full_path)
            self.samples += samples
        
        self.total_len = len(self.samples)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, index: int, skip_image: bool = False) -> dict[str, Any]:
        image_path = self.samples[index]

        if skip_image:
            image = None
        else:
            try:
                image = load_image(image_path)
                if self.transform is not None:
                    image = self.transform(image)
            except Exception:
                print(f"Fail to load {image_path}")
                raise OSError
                
        if self.return_dict:
            return {
                "index": index,
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "data": image,
            }
        else:
            return image