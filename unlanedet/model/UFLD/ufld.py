from ..module.architecture import Detector

class UFLD(Detector):
    def __init__(self, backbone=None, aggregator=None, neck=None, head=None):
        super().__init__(backbone, aggregator, neck, head)
    
    def get_lanes(self, output):
        return super().get_lanes(output)

    def forward(self, batch):
        return super().forward(batch)