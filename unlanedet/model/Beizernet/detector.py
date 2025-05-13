import torch
import torch.nn as nn

from ..module import Detector
from ...layers import DilatedBlocks

class BezierLaneNet(Detector):
    def __init__(self, 
                 backbone=None, 
                 aggregator=None, 
                 neck=None, 
                 head=None,
                 dilated_blocks=None,):
        super().__init__(backbone, aggregator, neck, head)
        
        if dilated_blocks is not None:
            self.dilated_blocks = DilatedBlocks(**dilated_blocks)
        else:
            self.dilated_blocks = None
            
    def forward(self, batch):
        output = {}
        batch = self.to_cuda(batch)
        fea = self.backbone(batch['img'])
        
        f = fea[2]

        if self.dilated_blocks is not None:
            x = self.dilated_blocks(f)

        if self.neck:
            x = self.neck(x)

        if self.aggregator:
            x = self.aggregator(x)

        if self.training:
            out = self.head(x, f)
            output.update(self.head.loss(out,batch))
        else:
            output = self.head(x,f)

        return output