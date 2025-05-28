import pdb

import torch


def gline_iou(pred, target, img_w, img_h,length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    pred_y = torch.zeros_like(pred).cuda()
    pred_y_ = torch.linspace(0, img_h, 72).cuda()
    pred_y[:] = pred_y_.cuda()
    cat = torch.zeros(pred.shape[0],1).cuda()
    target_y = torch.zeros_like(target).cuda()
    target_y_ = torch.linspace(0, img_h, 72).cuda()
    target_y[:] = target_y_.cuda()

    pred_grad = ((pred[:, 1:] - pred[:, :-1]).cuda() / ((pred_y[:, 1:] - pred_y[:, :-1]) + 1e-9).cuda())

    target_grad = ((target[:, 1:] - target[:, :-1]).cuda() / ((target_y[:, 1:] - target_y[:, :-1]) + 1e-9).cuda())

    sinp = torch.abs(torch.sin(torch.atan(pred_grad))).cuda()
    cosp = torch.abs(torch.cos(torch.atan(pred_grad))).cuda()

    sing = torch.abs(torch.sin(torch.atan(target_grad))).cuda()
    cosg = torch.abs(torch.cos(torch.atan(target_grad))).cuda()

    S_P = 2 * length * torch.sqrt((pred[:, 1:] - pred[:, :-1]).cuda() ** 2 + (pred_y[:, 1:] - pred_y[:, :-1]).cuda() ** 2)
    S_G = 2 * length * torch.sqrt((target[:, 1:] - target[:, :-1]).cuda() ** 2 + (target_y[:, 1:] - target_y[:, :-1]).cuda() ** 2)

    y_top = target_y[:,1:] + length * torch.max(sinp,sing)
    y_bottom = target_y[:,:-1] - length * torch.max(sinp,sing)

    xl1 = pred[:,1:].cuda() - length * cosp
    xl2 = pred[:,:-1].cuda() - length * cosp
    xl3 = target[:,1:].cuda() - length * cosg
    xl4 = target[:,:-1].cuda() - length * cosg
    x_left = torch.minimum(torch.minimum(torch.minimum(xl1, xl2), xl3), xl4).cuda()

    xr1 = pred[:, 1:].cuda() + length * cosp
    xr2 = pred[:, :-1].cuda() + length * cosp
    xr3 = target[:, 1:].cuda() + length * cosg
    xr4 = target[:, :-1].cuda() + length * cosg
    x_right = torch.maximum(torch.maximum(torch.maximum(xr1,xr2),xr3),xr4).cuda()

    S_bo = ((y_top - y_bottom) * (x_right - x_left)).cuda()

    G = ((S_bo - torch.minimum(S_bo,S_G + S_P)) / S_bo).cuda()
    G = torch.cat((cat,G),dim = 1)

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
    iou = (ovr.sum(dim=-1).cuda() / (union.sum(dim=-1).cuda() + 1e-9))
    G = G.mean(dim = -1)
    giou = iou-G
    return giou


def gliou_loss(pred, target, img_w,img_h, length=15):
    return (1 - gline_iou(pred, target, img_w,img_h,length)).mean()