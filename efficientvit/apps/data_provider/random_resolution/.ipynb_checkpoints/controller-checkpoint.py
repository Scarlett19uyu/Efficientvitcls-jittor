import copy
import jittor as jt
from jittor import transform
import jittor.transform as transforms
from efficientvit.models.utils import jt_random_choices
from PIL import Image  # 添加PIL导入

__all__ = [
    "RRSController",
    "get_interpolate",
    "MyRandomResizedCrop",
]

class RRSController:
    ACTIVE_SIZE = (224, 224)
    IMAGE_SIZE_LIST = [(224, 224)]
    CHOICE_LIST = None

    @staticmethod
    def get_candidates() -> list[tuple[int, int]]:
        return copy.deepcopy(RRSController.IMAGE_SIZE_LIST)

    @staticmethod
    def sample_resolution(batch_id: int) -> None:
        RRSController.ACTIVE_SIZE = RRSController.CHOICE_LIST[batch_id]

    @staticmethod
    def set_epoch(epoch: int, batch_per_epoch: int) -> None:
        jt.set_global_seed(epoch)
        RRSController.CHOICE_LIST = jt_random_choices(
            RRSController.get_candidates(),
            k=batch_per_epoch,
        )

def get_interpolate(name: str) -> int:
    """
    修改为返回PIL的插值枚举值
    """
    mapping = {
        "nearest": Image.NEAREST,    # 0
        "bilinear": Image.BILINEAR,  # 2
        "bicubic": Image.BICUBIC,    # 3
        "box": Image.BOX,            # 4
        "hamming": Image.HAMMING,    # 5
        "lanczos": Image.LANCZOS,    # 1
        "random": Image.BILINEAR     # 将random映射为bilinear
    }
    if name.lower() in mapping:
        return mapping[name.lower()]
    else:
        raise NotImplementedError(f"Interpolation {name} not implemented")

class MyRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        size=224,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = "bilinear",  # 默认值改为bilinear
    ):
        super().__init__(
            size=size,
            scale=scale,
            ratio=ratio,
            interpolation=get_interpolate(interpolation)  # 转换为合法插值
        )
        self._interpolation_name = interpolation  # 保留原始名称用于显示

    def execute(self, img: jt.Var) -> jt.Var:
        """
        使用父类的execute方法，确保正确处理插值
        """
        return super().execute(img)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        format_string += f"(\n\tsize={RRSController.get_candidates()},\n"
        format_string += f"\tscale={tuple(round(s, 4) for s in self.scale)},\n"
        format_string += f"\tratio={tuple(round(r, 4) for r in self.ratio)},\n"
        format_string += f"\tinterpolation={self._interpolation_name})"  # 使用原始名称
        return format_string