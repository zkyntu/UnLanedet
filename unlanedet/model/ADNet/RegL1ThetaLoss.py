import torch.nn as nn
import torch.nn.functional as F
class RegL1ThetaLoss(nn.Module):

    def __init__(self):
        super(RegL1ThetaLoss, self).__init__()

    def forward(self, output, target, mask):
        loss = F.l1_loss(output * mask, target * mask, size_average=False)
        mask = mask.bool().float()
        loss = loss / (mask.sum() + 1e-4)
        return loss