"""Random resolution data loader compatible with multi-processing and distributed training.

Replace Pytorch's DataLoader with RRSDataLoader to support random resolution
at the training time, resolution sampling is controlled by RRSController
"""


import jittor as jt
from jittor import init
from jittor import nn
from .controller import *
