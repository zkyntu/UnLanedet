from ..CLRNet.dynamic_assign import focal_cost,dynamic_k_assign,distance_cost
from .lane_iou import LaneIoUCost
import torch

# For CUlane. Other datasets change the img_w and img_h
lane_iou_dynamic = LaneIoUCost(use_pred_start_end=False,use_giou=True)

lane_iou_cost = LaneIoUCost(lane_width=30 / 800,use_pred_start_end=True,use_giou=True)

def clrernet_cost(predictions,targets,pred_xs, target_xs, reg_weight = 3):
    start = end = None
    length = predictions[:,5].detach().clone()
    y0 = predictions[:,2].detach().clone()
    start = y0.clamp(min=0, max=1)
    end = (start + length).clamp(min=0, max=1)    
    iou_cost = lane_iou_cost(
        pred_xs,
        target_xs,
        start,
        end,
    )
    iou_score = 1 - (1 - iou_cost) / torch.max(1 - iou_cost) + 1e-2
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())

    cost = -iou_score * reg_weight + cls_score
    return cost

def assign(    
        predictions,
        targets,
        img_w,
        img_h,):
    '''
    computes dynamicly matching based on the cost, including cls cost and lane similarity cost
    Args:
        predictions (Tensor): predictions predicted by each stage, shape: (num_priors, 78)
        targets (Tensor): lane targets, shape: (num_targets, 78)
    return:
        matched_row_inds (Tensor): matched predictions, shape: (num_targets)
        matched_col_inds (Tensor): matched targets, shape: (num_targets)
    '''
    pred_xs = predictions[:,6:]
    target_xs = targets[:, 6:] / (img_w - 1)  # abs -> relative

    iou_dynamick = lane_iou_dynamic(pred_xs, target_xs)

    cost = clrernet_cost(predictions,targets,pred_xs,target_xs)

    matched_row_inds, matched_col_inds = dynamic_k_assign(cost, iou_dynamick)

    return matched_row_inds, matched_col_inds
