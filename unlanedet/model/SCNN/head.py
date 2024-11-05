import torch
from torch import nn
import torch.nn.functional as F
from ..module.core.lane import Lane
import cv2
import numpy as np

from ..module.head import LaneSeg

class SCNNHead(LaneSeg):
    def __init__(self, decoder, exist=None, thr=0.6, sample_y=None, cfg=None):
        super().__init__(decoder, exist, thr, sample_y, cfg)
