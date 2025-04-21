import jittor as jt
from jittor import init
from jittor import nn
from efficientvit.apps.trainer.run_config import RunConfig
__all__ = ['ClsRunConfig']

class ClsRunConfig(RunConfig):
    label_smooth: float
    mixup_config: dict
    bce: bool
    mesa: dict

    @property
    def none_allowed(self):
        return (['mixup_config', 'mesa'] + super().none_allowed)