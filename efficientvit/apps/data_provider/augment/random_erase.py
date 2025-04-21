import jittor as jt
import math
import numpy as np
import random
from PIL import Image
__all__ = ['RandomErasing']
class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        """
        实现可调用接口
        """
        if random.random() > self.p:
            return img
            
        if isinstance(img, jt.Var):
            img = img.numpy()
            
        # 获取图像尺寸
        img_h, img_w = img.shape[-2], img.shape[-1]
        
        # 计算擦除区域
        area = img_h * img_w
        for _ in range(10):  # 最多尝试10次
            erase_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            h = int(round((erase_area * aspect_ratio) ** 0.5))
            w = int(round((erase_area / aspect_ratio) ** 0.5))
            
            if h < img_h and w < img_w:
                # 随机选择位置
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                
                # 执行擦除
                if isinstance(img, np.ndarray):
                    if len(img.shape) == 3:  # CHW格式
                        img[:, i:i+h, j:j+w] = self.value
                    else:  # HWC格式
                        img[i:i+h, j:j+w, :] = self.value
                return img
                
        return img  # 如果10次尝试都失败，返回原图

    def __repr__(self):
        return f"RandomErasing(p={self.p}, scale={self.scale}, ratio={self.ratio}, value={self.value})"