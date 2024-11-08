import torch
import torch.nn.functional as F
from ..module import FocalLoss
from .lane_iou import LaneIoULoss
from .assign import assign
from ..CLRNet import CLRHead

class CLRerHead(CLRHead):
    def __init__(self, 
                 num_points=72, 
                 prior_feat_channels=64, 
                 fc_hidden_dim=64, 
                 num_priors=192, 
                 num_fc=2, 
                 refine_layers=3, 
                 sample_points=36, 
                 cfg=None):
        super().__init__(num_points, 
                         prior_feat_channels, 
                         fc_hidden_dim, 
                         num_priors, 
                         num_fc, 
                         refine_layers, 
                         sample_points, 
                         cfg)
        self.iou_loss = LaneIoULoss(loss_weight=self.cfg.iou_loss_weight,)
    
    def loss(self, 
             output, 
             batch, 
             cls_loss_weight=2, 
             xyt_loss_weight=0.5,  
             seg_loss_weight=1):
        if 'cls_loss_weight' in self.cfg:
            cls_loss_weight = self.cfg.cls_loss_weight
        if 'xyt_loss_weight' in self.cfg:
            xyt_loss_weight = self.cfg.xyt_loss_weight
        if 'seg_loss_weight' in self.cfg:
            seg_loss_weight = self.cfg.seg_loss_weight

        predictions_lists = output['predictions_lists']
        targets = batch['lane_line'].clone()
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
        cls_loss = torch.tensor(0.).to(self.priors.device)
        reg_xytl_loss = torch.tensor(0.).to(self.priors.device)
        iou_loss = torch.tensor(0.).to(self.priors.device)

        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h)

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]

                # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= (self.img_w - 1)
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()

                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()

                with torch.no_grad():
                    predictions_starts = torch.clamp(
                        (predictions[matched_row_inds, 2] *
                         self.n_strips).round().long(), 0,
                        self.n_strips)  # ensure the predictions starts is valid
                    target_starts = (target[matched_col_inds, 2] *
                                     self.n_strips).round().long()
                    target_yxtl[:, -1] -= (predictions_starts - target_starts
                                           )  # reg length

                # Loss calculation
                cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum(
                ) / target.shape[0]

                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180
                reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                    reg_yxtl, target_yxtl,
                    reduction='none').mean()

                iou_loss = iou_loss + self.iou_loss(
                    reg_pred / self.img_w, reg_targets / self.img_w)

        # extra segmentation loss
        seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1),batch['seg'].long())

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)

        # loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
        #     + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight

        return_value = {
            'cls_loss': cls_loss * cls_loss_weight,
            'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
            'seg_loss': seg_loss * seg_loss_weight,
            'iou_loss': iou_loss

        }

        return return_value
