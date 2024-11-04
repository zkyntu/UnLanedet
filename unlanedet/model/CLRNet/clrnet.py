from ..module import Detector

class CLRNet(Detector):
    def __init__(self, backbone=None, aggregator=None, neck=None, head=None):
        super().__init__(backbone, aggregator, neck, head)
    
    def forward(self, batch):
        return super().forward(batch)
    