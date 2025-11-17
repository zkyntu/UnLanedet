import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers import Conv2d,get_norm
from ..module.head import PlainDecoder
from ..CLRNet.roi_gather import LinearModule,ROIGather
from ..CLRNet.dynamic_assign import assign
from ..CLRNet.line_iou import liou_loss
from ..module.core.lane import Lane
from ..module.losses import FocalLoss
from .angle_loss import a_loss
from .dyconv import DynamicConv,AddMerge

from ...utils.events import get_event_storage

from ...layers.ops import nms

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def LinearNormModule(hidden_dim):
    return nn.ModuleList(
        [nn.Linear(hidden_dim, hidden_dim),
         nn.LayerNorm(hidden_dim),
         nn.ReLU(inplace=True)])

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)
    

class DiffusionLaneHead(nn.Module):
    def __init__(self, 
                 num_points=72, 
                 prior_feat_channels=64, 
                 fc_hidden_dim=64, 
                 time_dim=64,
                 num_priors=192, 
                 num_fc=2, 
                 refine_layers=3, 
                 sample_points=36, 
                 dropout=0.1,
                 activation="relu",
                 dim_feedforward=192,
                 cfg=None):
        super().__init__()

        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim

        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) *
                                self.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1,
                                       0,
                                       steps=self.n_offsets,
                                       dtype=torch.float32))

        self.prior_feat_channels = prior_feat_channels

        # generate xys for feature map
        self.seg_decoder = PlainDecoder(cfg)

        self.inst_interact = DynamicConv(cfg)
        self.self_attn = nn.MultiheadAttention(self.prior_feat_channels, 8, dropout=dropout)
        self.norm1 = nn.LayerNorm(prior_feat_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(prior_feat_channels)
        self.norm3 = nn.LayerNorm(prior_feat_channels)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(prior_feat_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, prior_feat_channels)
        self.activation = nn.ReLU()
        self.fcs = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)
        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors,
                                    self.sample_points, self.fc_hidden_dim,
                                    self.refine_layers)

        init_weight = torch.tensor([1,1],dtype=torch.float32)
        # self.merge_sa = AddMerge(True,self.prior_feat_channels,False,init_weight)
        self.merage_gather = AddMerge(True,self.prior_feat_channels,False,init_weight)

        # self.roi_gather2 = ROIGather(self.prior_feat_channels, self.num_priors,
        #                             self.sample_points, self.fc_hidden_dim,
        #                             self.refine_layers -3 if self.refine_layers > 3 else self.refine_layers)

        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearNormModule(self.fc_hidden_dim)]
            cls_modules += [*LinearNormModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        # self.dyconv = nn.Sequential(*[DyConv(self.fc_hidden_dim,self.fc_hidden_dim) for i in range(3)])

        self.reg_layers = nn.Linear(
            self.fc_hidden_dim, self.n_offsets + 1 + 2 +
            1)  # n offsets + 1 length + start_x + start_y + theta
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)
        # self.cls_emb_bias = nn.Embedding(self.num_priors,2)

        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        self.criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label,
                                     weight=weights)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(fc_hidden_dim),
            nn.Linear(fc_hidden_dim, time_dim*4),
            nn.GELU(),
            nn.Linear(time_dim*4, time_dim*4),
        )

        self.block_time_mlp = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(time_dim*4, time_dim * 2)) 
                                             for i in range(self.refine_layers)])

        # init the weights here
        self.init_weights()

    # function to init layer weights
    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        # for m in self.cls_emb_bias.parameters():
        #     nn.init.zeros_(m)
        

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        '''
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W) 
        '''

        batch_size = batch_features.shape[0]

        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1)

        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
#        import pdb;pdb.set_trace()
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
        feature = F.grid_sample(batch_features, grid,
                                align_corners=True).permute(0, 2, 1, 3)

        feature = feature.reshape(batch_size * num_priors,
                                  self.prior_feat_channels, self.sample_points,
                                  1)
        return feature


    def forward(self,x,t,priors,**kwargs):
        batch_features = list(x[len(x) - self.refine_layers:])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]
        

        priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs]
    

        predictions_lists = []

        # iterative refine
        prior_features_stages = []

        time_emb = self.time_mlp(t)

        # batch_features = self.dyconv(batch_features)

        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])

            # dynamic conv
            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs)    
            
            # import pdb;pdb.set_trace()
            pro_features = F.relu(self.fc_norm(self.fcs(batch_prior_features.view(batch_size*self.num_priors,-1))))
            pro_features = pro_features.view(batch_size,num_priors,self.prior_feat_channels)
            # pro_features  = batch_prior_features.view(batch_size,num_priors,self.prior_feat_channels,-1).mean(-1,keepdim=False)
            pro_features = pro_features.permute(1, 0, 2)
            pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
            pro_features = pro_features + self.dropout1(pro_features2)
            pro_features = self.norm1(pro_features)

            pro_obj_features = pro_features.clone().permute(1, 0 ,2)

            # prior_fea_self_stage = pro_features.view(batch_size * self.num_priors,self.prior_feat_channels,self.sample_points,1)
            prior_features_stages.append(batch_prior_features)

            # import pdb;pdb.set_trace()
            batch_prior_features_ = batch_prior_features.view(batch_size*num_priors,self.prior_feat_channels,-1).permute(2, 0, 1)

            pro_features = pro_features.view(num_priors, batch_size, self.prior_feat_channels).permute(1, 0, 2).reshape(1, batch_size * num_priors, self.prior_feat_channels)
            pro_features2 = self.inst_interact(pro_features, batch_prior_features_)
            pro_features = pro_features + self.dropout2(pro_features2)
            obj_features = self.norm2(pro_features)

            obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
            obj_features = obj_features + self.dropout3(obj_features2)
            obj_features_ = self.norm3(obj_features)  
            obj_features_ = obj_features_.view(num_priors, batch_size,-1).permute(1, 0 ,2)

            if stage > len(batch_features) -1 :
                stage_ = stage - len(batch_features)
                batch_prior_features = self.pool_prior_features(
                    batch_features[stage_], num_priors, prior_xs)
                prior_features_stages.append(batch_prior_features)
                obj_features = self.roi_gather(prior_features_stages,
                                              batch_features[stage_], stage)
            else:
                # batch_prior_features = self.pool_prior_features(
                #     batch_features[stage], num_priors, prior_xs)

                # prior_features_stages.append(batch_prior_features)

                obj_features = self.roi_gather(prior_features_stages,
                                              batch_features[stage], stage, pro_obj_features)
                
            obj_features = self.merage_gather([obj_features_,obj_features])
            fc_features = obj_features.view(num_priors, batch_size,
                                           -1).reshape(batch_size * num_priors,
                                                       self.fc_hidden_dim)
            
            scale_shift = self.block_time_mlp[stage](time_emb)
            scale_shift = torch.repeat_interleave(scale_shift, num_priors, dim=0)
            scale, shift = scale_shift.chunk(2, dim=1)
            fc_features = fc_features * (scale + 1) + shift

            fc_features = fc_features.view(num_priors, batch_size,
                                           -1).reshape(batch_size * num_priors,
                                                       self.fc_hidden_dim)

            cls_features = fc_features.clone()
            reg_features = fc_features.clone()

            # apply classification and location

            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            cls_logits = self.cls_layers(cls_features)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1])  # (B, num_priors, 2)
            # cls_bias = self.cls_emb_bias.weight.unsqueeze(0).repeat(batch_size,1,1)
            # cls_logits = cls_logits + cls_bias
            reg = reg.reshape(batch_size, -1, reg.shape[1])

            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits

            predictions[:, :,
                        2:5] += reg[:, :, :3]  # also reg theta angle here
            predictions[:, :, 5] = reg[:, :, 3]  # length

            def tran_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)

            predictions[..., 6:] = (
                tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
                ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
                  tran_tensor(predictions[..., 2])) * self.img_h /
                 torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]

            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        if self.training:
            seg_features = torch.cat([
                F.interpolate(feature,
                              size=[
                                  batch_features[-1].shape[2],
                                  batch_features[-1].shape[3]
                              ],
                              mode='bilinear',
                              align_corners=False)
                for feature in batch_features
            ],
                                     dim=1)
#            import pdb;pdb.set_trace()
            seg = self.seg_decoder(seg_features)
            output = {'predictions_lists': predictions_lists}
            output.update(**seg)
            return output

        return predictions_lists[-1]


    def predictions_to_pred(self, predictions):
        '''
        Convert predictions to internal Lane structure for evaluation.
        '''
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) +
                       self.cfg.cut_height) / self.cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes
    
    def loss_aux(self,output,batch):

        if 'cls_loss_weight' in self.cfg:
            cls_loss_weight = self.cfg.cls_loss_weight
        if 'xyt_loss_weight' in self.cfg:
            xyt_loss_weight = self.cfg.xyt_loss_weight
        if 'iou_loss_weight' in self.cfg:
            iou_loss_weight = self.cfg.iou_loss_weight
        if 'seg_loss_weight' in self.cfg:
            seg_loss_weight = self.cfg.seg_loss_weight
        if 'angle_loss_weight' in self.cfg:
            angle_loss_weight = self.cfg.angle_loss_weight


        predictions_lists = output['predictions_lists']
        targets = batch['lane_line'].clone()
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
        cls_loss = torch.tensor(0.).to(self.prior_ys.device)
        reg_xytl_loss = torch.tensor(0.).to(self.prior_ys.device)
        iou_loss = torch.tensor(0.).to(self.prior_ys.device)
        angle_loss = torch.tensor(0.).to(self.prior_ys.device)        

        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]
                target_num = target.shape[0]

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    if stage >= 3:
                    # if stage == self.refine_layers-1:
                        use_dynamic = True
                    else:
                        use_dynamic = False
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h,use_vfl_iter=self.cfg.use_vfl_iter,dynamic_k=4,use_dynamic=use_dynamic)

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
#                import pdb;pdb.set_trace()
#                cls_target[:target_num] = 1 # force positive assignment
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

                line_iou_loss, ious = liou_loss(reg_pred, reg_targets,self.img_w, length=15)
                iou_loss = iou_loss + line_iou_loss

                angle_loss  += a_loss(reg_pred, reg_targets, self.img_h)

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)
        angle_loss /= (len(targets) * self.refine_layers)

        # loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
        #     + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight

        return_value = {
            'aux_cls_loss': cls_loss * cls_loss_weight,
            'aux_reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
            'aux_iou_loss': iou_loss * iou_loss_weight,
            'aux_angle_loss': angle_loss * angle_loss_weight
        }

        return return_value

    def loss(self,
             output,
             batch,
             cls_loss_weight=2.,
             xyt_loss_weight=0.5,
             iou_loss_weight=2.,
             seg_loss_weight=1.):
        if 'cls_loss_weight' in self.cfg:
            cls_loss_weight = self.cfg.cls_loss_weight
        if 'xyt_loss_weight' in self.cfg:
            xyt_loss_weight = self.cfg.xyt_loss_weight
        if 'iou_loss_weight' in self.cfg:
            iou_loss_weight = self.cfg.iou_loss_weight
        if 'seg_loss_weight' in self.cfg:
            seg_loss_weight = self.cfg.seg_loss_weight
        if 'angle_loss_weight' in self.cfg:
            angle_loss_weight = self.cfg.angle_loss_weight

        predictions_lists = output['predictions_lists']
        targets = batch['lane_line'].clone()
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
        cls_loss = torch.tensor(0.).to(self.prior_ys.device)
        reg_xytl_loss = torch.tensor(0.).to(self.prior_ys.device)
        iou_loss = torch.tensor(0.).to(self.prior_ys.device)
        angle_loss = torch.tensor(0.).to(self.prior_ys.device)

        storge = get_event_storage()

        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]
                target_num = target.shape[0]

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    if stage >= 3:
                    # if stage == self.refine_layers-1:
                        use_dynamic = True
                    else:
                        use_dynamic = False
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h,use_vfl_iter=self.cfg.use_vfl_iter,dynamic_k=4,use_dynamic=use_dynamic)

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
#                import pdb;pdb.set_trace()
#                cls_target[:target_num] = 1 # force positive assignment
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

                line_iou_loss, ious = liou_loss(reg_pred, reg_targets,self.img_w, length=15)
                iou_loss = iou_loss + line_iou_loss

                angle_loss  += a_loss(reg_pred, reg_targets, self.img_h)

                # if storge.iter >= self.cfg.use_vfl_iter and stage == len(self.refine_layers) - 1:
                # if stage == self.refine_layers- 1:
                #     cls_loss = cls_loss + cls_vfl_criterion(cls_pred, cls_target,ious).sum(
                #     ) / target.shape[0]
                # else:
                #     cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum(
                #     ) / target.shape[0]

        # extra segmentation loss
        seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1),batch['seg'].long())

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)
        angle_loss /= (len(targets) * self.refine_layers)

        # loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
        #     + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight

        return_value = {
            'cls_loss': cls_loss * cls_loss_weight,
            'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
            'seg_loss': seg_loss * seg_loss_weight,
            'iou_loss': iou_loss * iou_loss_weight,
            'angle_loss': angle_loss * angle_loss_weight
        }

        return return_value


    def get_lanes(self, output, as_lanes=True):
        '''
        Convert model output to lanes.
        '''
        softmax = nn.Softmax(dim=1)

        decoded = []
        for predictions in output:
            # filter out the conf lower than conf threshold
            threshold = self.cfg.test_parameters.conf_threshold
            # threshold = 0.05
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            nms_predictions = predictions.detach().clone()
            nms_predictions = torch.cat(
                [nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[...,
                            5:] = nms_predictions[..., 5:] * (self.img_w - 1)

            keep, num_to_keep, _ = nms(
                nms_predictions,
                scores,
                overlap=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes)
            keep = keep[:num_to_keep]
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)

        return decoded
