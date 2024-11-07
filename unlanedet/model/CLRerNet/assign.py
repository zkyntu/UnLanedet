from ..CLRNet.dynamic_assign import focal_cost,dynamic_k_assign,distance_cost
from .lane_iou import LaneIoUCost
import torch

lane_iou_cost = LaneIoUCost(use_pred_start_end=False,use_giou=True)

def assign(
    predictions,
    targets,
    img_w,
    img_h,
    distance_cost_weight=3.,
    cls_cost_weight=1.,
):
    '''
    computes dynamicly matching based on the cost, including cls cost and lane similarity cost
    Args:
        predictions (Tensor): predictions predicted by each stage, shape: (num_priors, 78)
        targets (Tensor): lane targets, shape: (num_targets, 78)
    return:
        matched_row_inds (Tensor): matched predictions, shape: (num_targets)
        matched_col_inds (Tensor): matched targets, shape: (num_targets)
    '''
    predictions = predictions.detach().clone()
    predictions[:, 3] *= (img_w - 1)
    predictions[:, 6:] *= (img_w - 1)
    targets = targets.detach().clone()

    # distances cost
    distances_score = distance_cost(predictions, targets, img_w)
    distances_score = 1 - (distances_score / torch.max(distances_score)
                           ) + 1e-2  # normalize the distance

    # classification cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    target_start_xys = targets[:, 2:4]  # num_targets, 2
    target_start_xys[..., 0] *= (img_h - 1)
    prediction_start_xys = predictions[:, 2:4]
    prediction_start_xys[..., 0] *= (img_h - 1)

    start_xys_score = torch.cdist(prediction_start_xys, target_start_xys,
                                  p=2).reshape(num_priors, num_targets)
    start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

    target_thetas = targets[:, 4].unsqueeze(-1)
    theta_score = torch.cdist(predictions[:, 4].unsqueeze(-1),
                              target_thetas,
                              p=1).reshape(num_priors, num_targets) * 180
    theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

    cost = -(distances_score * start_xys_score * theta_score
             )**2 * distance_cost_weight + cls_score * cls_cost_weight

    iou = lane_iou_cost(predictions[..., 6:], targets[..., 6:])
    matched_row_inds, matched_col_inds = dynamic_k_assign(cost, iou)

    return matched_row_inds, matched_col_inds