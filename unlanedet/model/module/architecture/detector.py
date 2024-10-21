import torch
import torch.nn as nn

@torch.jit.script_if_tracing
def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Tracing friendly way to cast tensor to another tensor's device. Device will be treated
    as constant during tracing, scripting the casting process as whole can workaround this issue.
    """
    return src.to(dst.device)

class Detector(nn.Module):
    def __init__(self,backbone=None,aggregator=None,neck=None,head=None):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.neck = neck
        self.head = head

        self.register_buffer("device_tensor", torch.tensor([1,1,1]).view(-1, 1, 1), False)

    def _move_to_current_device(self, x):
        return move_device_like(x, self.device_tensor)

    def to_cuda(self, batch):
#        import pdb;pdb.set_trace()
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = self._move_to_current_device(batch[k])
        return batch

    def get_lanes(self, output):
        return self.head.get_lanes(output)

    def forward(self, batch):
        output = {}
        batch = self.to_cuda(batch)
#        import pdb;pdb.set_trace()
        fea = self.backbone(batch['img'])

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            out = self.head(fea, batch=batch)
            output.update(self.head.loss(out, batch))
        else:
            output = self.head(fea)

        return output