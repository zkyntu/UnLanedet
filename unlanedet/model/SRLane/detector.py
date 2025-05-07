from typing import List

import torch
import torch.nn as nn
from ..module.architecture import Detector

class SRLane(Detector):
    def __init__(self,
                 backbone=None,
                 aggregator=None,
                 neck=None,
                 head=None,
                 rpn_head=None):
        super().__init__(backbone,aggregator,neck,head)
        self.rpn_head = rpn_head

    def move_list_data_to_cuda(self,data):
        for i in range(len(data)):
            # import pdb;pdb.set_trace()
            if not isinstance(data[i], torch.Tensor):
                if isinstance(data[i],List):
                    data[i] = self.move_list_data_to_cuda(data[i])
                else:
                    continue
            else:
                data[i] = self._move_to_current_device(data[i])
        return data

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                if isinstance(batch[k],List):
                    batch[k] = self.move_list_data_to_cuda(batch[k])
                else:
                    continue
            else:
                batch[k] = self._move_to_current_device(batch[k])
        return batch

    def forward(self, batch):
        output = {}
        # import pdb;pdb.set_trace()
        batch = self.to_cuda(batch)
        fea = self.backbone(batch['img'])

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            rpn_result_dict = self.rpn_head(fea)
            rpn_loss = self.rpn_head.loss(**rpn_result_dict, **batch)
            output.update(rpn_loss)
            roi_result_dict = self.head(fea, **rpn_result_dict)
            roi_loss = self.head.loss(roi_result_dict, batch=batch)
            output.update(roi_loss)
        else:
            rpn_result_dict = self.rpn_head(fea)
            output = self.head(fea, **rpn_result_dict)

        return output
