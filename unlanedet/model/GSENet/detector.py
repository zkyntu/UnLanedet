import torch
from ..module import Detector

class GSENet(Detector):
    def forward(self, batch):
        batch = self.to_cuda(batch)
        output = {}
        fea = self.backbone(batch['img'])
        fea2 = torch.clone(fea[-1])
        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            output = self.head(fea,fea2, batch=batch)
        else:
            output = self.head(fea,fea2)

        return output