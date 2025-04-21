import jittor as jt
from jittor import init
from jittor import nn
from efficientvit.apps.data_provider.augment import rand_bbox
from efficientvit.models.utils.random import jt_randint, jt_shuffle

__all__ = ['apply_mixup', 'mixup', 'cutmix']

def apply_mixup(images: jt.Var, labels: jt.Var, lam: float, mix_type='mixup') -> tuple:
    """
    Apply mixup or cutmix augmentation
    
    Args:
        images: input images tensor [B,C,H,W]
        labels: target labels tensor [B,...]
        lam: mixing ratio
        mix_type: 'mixup' or 'cutmix'
    
    Returns:
        mixed images and labels
    """
    if mix_type == 'mixup':
        return mixup(images, labels, lam)
    elif mix_type == 'cutmix':
        return cutmix(images, labels, lam)
    else:
        raise NotImplementedError(f"Mix type {mix_type} not implemented")

def mixup(images: jt.Var, target: jt.Var, lam: float) -> tuple:
    """
    Apply mixup augmentation
    
    Args:
        images: input images [B,C,H,W]
        target: target labels [B,...]
        lam: mixing ratio
    
    Returns:
        mixed images and labels
    """
    batch_size = images.shape[0]
    rand_index = jt_shuffle(list(range(batch_size)))
    flipped_images = images[rand_index]
    flipped_target = target[rand_index]
    
    mixed_images = lam * images + (1 - lam) * flipped_images
    mixed_target = lam * target + (1 - lam) * flipped_target
    
    return mixed_images, mixed_target

def cutmix(images: jt.Var, target: jt.Var, lam: float) -> tuple:
    """
    Apply cutmix augmentation
    
    Args:
        images: input images [B,C,H,W]
        target: target labels [B,...]
        lam: mixing ratio
    
    Returns:
        mixed images and labels
    """
    batch_size, _, h, w = images.shape
    rand_index = jt_shuffle(list(range(batch_size)))
    flipped_images = images[rand_index]
    flipped_target = target[rand_index]
    
    lam_list = []
    for i in range(batch_size):
        bbx1, bby1, bbx2, bby2 = rand_bbox(h=h, w=w, lam=lam, rand_func=jt_randint)
        images[i, :, bby1:bby2, bbx1:bbx2] = flipped_images[i, :, bby1:bby2, bbx1:bbx2]
        lam_list.append(1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (h * w))
    
    lam_tensor = jt.array(lam_list).float32().view((batch_size, 1))
    mixed_target = lam_tensor * target + (1 - lam_tensor) * flipped_target
    
    return images, mixed_target