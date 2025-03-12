import torch
from ..module import Detector

class ADNet(Detector):
    def __init__(self, backbone=None, aggregator=None, neck=None, head=None):
        super().__init__(backbone, aggregator, neck, head)
    
    def forward(self, batch):
        output = {}
        batch = self.to_cuda(batch)
        fea = self.backbone(batch['img'])

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            out = self.head(fea, batch=batch)
            output.update(self.head.loss(out, batch))
        else:
            output = self.head(fea, batch=batch)

        return output
