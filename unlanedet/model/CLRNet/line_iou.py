import torch

def line_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def liou_loss(pred, target, img_w, length=15):
    return (1 - line_iou(pred, target, img_w, length)).mean()

## nn.module implementation
class CLRNetIoULoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, lane_width=15 / 800):
        """
        LineIoU loss employed in CLRNet.
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/losses/lineiou_loss.py
        Args:
            weight (float): loss weight.
            lane_width (float): virtual lane half-width.
        """
        super(CLRNetIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.lane_width = lane_width

    def calc_iou(self, pred, target, pred_width, target_width):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nl, Nr), relative coordinate
            target: ground truth, shape: (Nl, Nr), relative coordinate
            pred_width (torch.Tensor): virtual lane half-widths for prediction at pre-defined rows, shape (Nl, Nr).
            target_width (torch.Tensor): virtual lane half-widths for GT at pre-defined rows, shape (Nl, Nr).
        Returns:
            torch.Tensor: calculated IoU, shape (N).
        Nl: number of lanes, Nr: number of rows.
        """
        px1 = pred - pred_width
        px2 = pred + pred_width
        tx1 = target - target_width
        tx2 = target + target_width

        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)

        invalid_masks = (invalid_mask < 0) | (invalid_mask >= 1.0)
        ovr[invalid_masks] = 0.0
        union[invalid_masks] = 0.0
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou

    def forward(self, pred, target):
        assert (
            pred.shape == target.shape
        ), "prediction and target must have the same shape!"
        width = torch.ones_like(target) * self.lane_width
        iou = self.calc_iou(pred, target, width, width)
        return (1 - iou).mean() * self.loss_weight