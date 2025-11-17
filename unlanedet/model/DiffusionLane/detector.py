import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Detector

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffLane(Detector):
    def __init__(self, 
                 backbone=None, 
                 aggregator=None, 
                 neck=None, 
                 head=None,
                 cfg=None):
        super().__init__(backbone, aggregator, neck, head)

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        self.num_proposals = self.head.num_priors
        self.num_points = self.head.n_offsets
        self.scale = cfg.SNR_SCALE
        self.n_offsets = self.head.n_offsets
        self.box_renewal = True
        self.use_ensemble = True
        self.softmax = nn.Softmax(dim=-1)
        self.cfg = cfg
        self.positive_num = cfg.positive_num

        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.head.sample_points, dtype=torch.float32) *
                                self.head.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.head.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1,
                                       0,
                                       steps=self.n_offsets,
                                       dtype=torch.float32))

    def forward(self, batch):
        output = {}
        batch = self.to_cuda(batch)
        fea = self.backbone(batch['img'])

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            x_boxes, noises, t = self.prepare_targets(batch['lane_line'])
            t = t.squeeze(-1)
            out = self.head(fea,t,x_boxes, batch=batch)
            output.update(self.head.loss(out, batch))
        else:
            output = self.ddim_sample(fea,batchs=batch)

        return output

    def _init_prior_embeddings(self,num_priors):
        # [start_y, start_x, theta] -> all normalize
        box_placeholder = torch.randn(num_priors, 3,
                                          device=self.device_tensor.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6

        return box_placeholder

    def generate_priors_from_embeddings(self,num_priors,batch=None):

        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob
        if self.training:
            box_placeholder = self._init_prior_embeddings(num_priors)
            priors = box_placeholder.new_zeros(
                (num_priors, 2 + 2 + 2 + self.n_offsets), device=box_placeholder.device)
            priors[:,2:5] = box_placeholder

            priors[:, 6:] = (
                priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) *
                (self.head.img_w - 1) +
                ((1 - self.prior_ys.repeat(num_priors, 1) -
                priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
                self.head.img_h / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(
                    1, self.n_offsets) * math.pi + 1e-5))) / (self.head.img_w - 1)
        else:
            shape = (batch, num_priors, 3)
            box_placeholder = torch.randn(shape,device=self.device_tensor.device)

            priors = box_placeholder.new_zeros(
                (batch,num_priors, 2 + 2 + 2 + self.n_offsets), device=box_placeholder.device)
            priors[...,2:5] = box_placeholder
            # print(priors.shape)
            # import pdb;pdb.set_trace()

            priors[..., 6:] = (
                priors[..., 3].unsqueeze(2).clone().repeat(1, 1, self.n_offsets) *
                (self.head.img_w - 1) +
                ((1 - self.prior_ys.repeat(num_priors*batch, 1).view(batch,num_priors,-1) -
                priors[..., 2].unsqueeze(2).clone().repeat(1, 1, self.n_offsets)) *
                self.head.img_h / torch.tan(priors[..., 4].unsqueeze(2).clone().repeat(1, 1, self.n_offsets) * math.pi + 1e-5))) / (self.head.img_w - 1)

        return priors

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, x, t, x_self_cond=None, clip_x_start=False,batch=None):
        x_boxes_denoise = torch.clamp(x[...,2:5],min=-1 * self.scale, max=self.scale)
        x_boxes_denoise = ((x_boxes_denoise / self.scale) + 1) / 2
        x_boxes = self.reflash_batch_lane(x_boxes_denoise,x_boxes_denoise)
        prediction_list = self.head(backbone_feats,t,x_boxes,batch=batch)
#        prediction_list = self.head(backbone_feats, x_boxes, t, None)

        outputs_class = prediction_list[:,:,:2]
        outputs_coord = prediction_list[:,:,2:]
        # x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = (outputs_coord[:,:,:3] * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        x_ori_coord = x[...,2:5].clone()
        x_ori_coord = (x_ori_coord*2 - 1.) * self.scale
#        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
#        import pdb;pdb.set_trace()
        pred_noise = self.predict_noise_from_start(x_ori_coord, t, x_start)
#        outputs_coord[...,:3] = pred_noise

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, backbone_feats,clip_denoised=True,batchs=None):
#        import pdb;pdb.set_trace()
        batch = backbone_feats[0].shape[0]
        shape = (batch, self.num_proposals, 3)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

#        img = torch.randn(shape, device=self.device_tensor.device)
#        img[:,:2] = 0
        # img = torch.randn(shape,device=self.device_tensor.device)
        # import pdb;pdb.set_trace()
        img = self.generate_priors_from_embeddings(self.num_proposals,batch=batch)
#        img = img.repeat(batch,1,1)
        ensemble_label, ensemble_coord = [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device_tensor.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised,batch=batchs)
            
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            ensemble_coord.append(outputs_coord)
            ensemble_label.append(outputs_class)
            
            # only 1 iteration
            if time_next < 0:
                img = x_start
                continue

            img = self.renewal_batch_lane(outputs_class,
                                          pred_noise,
                                          x_start,
                                          img,
                                          time,
                                          time_next,
                                          eta)

#            import pdb;pdb.set_trace()
#            ensemble_coord.append(outputs_coord)
#            ensemble_label.append(outputs_class)

        if self.sampling_timesteps > 1:
#            import pdb;pdb.set_trace()
            box_pred_per_image = torch.cat(ensemble_coord, dim=1)
            labels_per_image = torch.cat(ensemble_label, dim=1)
            box_pred_per_image,labels_per_image = self.process_multi_sample_step(box_pred_per_image,labels_per_image)
        else:
#            import pdb;pdb.set_trace()
            box_pred_per_image = outputs_coord
            labels_per_image = outputs_class

        prediction_list = box_pred_per_image.new_zeros(batch,box_pred_per_image.shape[1],self.num_points + 6)

        prediction_list[:,:,:2] = labels_per_image
        prediction_list[:,:,2:] = box_pred_per_image

        return prediction_list

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    
    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device_tensor.device).long()
        noise = torch.randn(self.num_proposals, 3, device=self.device_tensor.device)
#        noise = torch.randn(self.num_proposals, self.num_points + 4, device=self.device_tensor.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = self.generate_priors_from_embeddings(self.num_proposals)
#            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device_tensor.device)
            num_gt = self.num_proposals

        if num_gt < self.num_proposals:
#            import pdb;pdb.set_trace()
            # gt_boxes = torch.repeat_interleave(gt_boxes,self.positive_num,dim=0)
            # num_gt *= self.positive_num
            box_placeholder = self.generate_priors_from_embeddings(self.num_proposals- num_gt)
            box_placeholder = torch.clip(box_placeholder, min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_coord = x_start[:,2:5].clone()
        x_coord = (x_coord * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_coord, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        x_start = self.reflash_lane(x,gt_boxes)

        return x_start, noise, t

    def prepare_targets(self, targets):
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in targets:
            targets_per_image = targets_per_image.clone()
            targets_per_image = targets_per_image[targets_per_image[:, 1] == 1]
            # normalize the target
#            import pdb;pdb.set_trace()
            targets_per_image[:,3] = targets_per_image[:,3] / (self.head.img_w - 1)
            targets_per_image[:,5] = targets_per_image[:,5] / self.head.n_strips
            targets_per_image[:,6:] = targets_per_image[:,6:] / (self.head.img_w - 1)
            targets_per_image[:,6:][targets_per_image[:,6:] < 0] = 0
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(targets_per_image)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)

        return torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    def renewal_batch_lane(self,
                           scores,
                           pred_noise,
                           x_start,
                           img,
                           time,
                           time_next,
                           eta):
        threshold = self.cfg.test_parameters.conf_threshold

        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        batch_img = []
        for scores_per_img, pred_noise_per_img, x_start_per_img, per_img in zip(scores,pred_noise,x_start,img):
#            import pdb;pdb.set_trace()
            scores_per_img_ = self.softmax(scores_per_img)[:,1]
            keep_inds = scores_per_img_ >= threshold

            num_remain = torch.sum(keep_inds)
            pred_noise_per_img = pred_noise_per_img[keep_inds, :]
            x_start_per_img = x_start_per_img[keep_inds, :]
            per_img = per_img[keep_inds, :]

            noise = torch.randn_like(pred_noise_per_img)
#            import pdb;pdb.set_trace()

            per_img[:,2:5] = x_start_per_img * alpha_next.sqrt() + \
                  c * pred_noise_per_img + \
                  sigma * noise

            if self.num_proposals - num_remain > 0:
                per_img = torch.cat((per_img, self.generate_priors_from_embeddings(self.num_proposals - num_remain,batch=1)[0]), dim=0)
            batch_img.append(per_img)
        
        return torch.stack(batch_img)

    def reflash_lane(self,noise_lane,gts):
        priors = gts.new_zeros(
            (noise_lane.shape[0], 2 + 2 + 2 + self.n_offsets), device=noise_lane.device)        

        priors[:,2:5] = noise_lane

        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) *
            (self.head.img_w - 1) +
            ((1 - self.prior_ys.repeat(noise_lane.shape[0], 1) -
            priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
            self.head.img_h / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(
                1, self.n_offsets) * math.pi + 1e-5))) / (self.head.img_w - 1)

        return priors
    
    def reflash_batch_lane(self,noise_lane,gts):
        batch = noise_lane.shape[0]
        num_priors = noise_lane.shape[1]
        priors = gts.new_zeros(
            (batch, num_priors, 2 + 2 + 2 + self.n_offsets), device=noise_lane.device)        

        priors[...,2:5] = noise_lane

        priors[..., 6:] = (
            priors[..., 3].unsqueeze(2).clone().repeat(1, 1, self.n_offsets) *
            (self.head.img_w - 1) +
            ((1 - self.prior_ys.repeat(num_priors*batch, 1).view(batch,num_priors,-1) -
            priors[..., 2].unsqueeze(2).clone().repeat(1, 1, self.n_offsets)) *
            self.head.img_h / torch.tan(priors[..., 4].unsqueeze(2).clone().repeat(
                1, 1, self.n_offsets) * math.pi + 1e-5))) / (self.head.img_w - 1)

        return priors

    def process_multi_sample_step(self,box_pred_per_image,labels_per_image):
        batch_pred_bbox = []
        batch_pred_label = []
        for pred_box_per_image,pred_labels_per_image in zip(box_pred_per_image,labels_per_image):
            pred_box_per_image = pred_box_per_image.detach().clone()
            pred_labels_per_imag_ = pred_labels_per_image.detach().clone()
#            import pdb;pdb.set_trace()

            pred_labels_per_imag_ = self.softmax(pred_labels_per_imag_)[:,1]
#            import pdb;pdb.set_trace()
            _,select_topk = torch.topk(pred_labels_per_imag_,k=self.num_proposals,sorted=False)

            pred_bbox = pred_box_per_image[select_topk]
            pred_label = pred_labels_per_image[select_topk]

            batch_pred_bbox.append(pred_bbox)
            batch_pred_label.append(pred_label)
        
        return torch.stack(batch_pred_bbox),torch.stack(batch_pred_label)
