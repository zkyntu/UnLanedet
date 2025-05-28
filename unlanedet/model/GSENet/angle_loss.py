import torch
import torch.nn.functional as F

def compute_angel(pred,target,img_h):
    #pred -> [num_pred,72]
    #target -> [num_target,72]
    pred_y = torch.zeros_like(pred).cuda()
    pred_y_ = torch.linspace(0,img_h,72).cuda()
    pred_y[:] = pred_y_.cuda()

    target_y = torch.zeros_like(target).cuda()
    target_y_ = torch.linspace(0, img_h, 72).cuda()
    target_y[:] = target_y_.cuda()

    pred_grad = (pred_y[:,1:] - pred_y[:,:-1]) / ((pred[:, 1:] - pred[:, :-1]) + 2e-9).cuda()
    #pred_grad = torch.cat((torch.zeros(pred.shape[0],1).cuda(),pred_grad),dim = -1).cuda()

    target_grad = (target_y[:, 1:] - target_y[:, :-1]) / ((target[:, 1:] - target[:, :-1]) + 2e-9).cuda()
    #target_grad = torch.cat((torch.zeros(target.shape[0], 1).cuda(), target_grad), dim=-1).cuda()

    line_angel = torch.abs(pred_grad - target_grad) / (1 + pred_grad * target_grad).cuda()
    line_angel = 1 - (torch.cos(torch.atan(line_angel))).cuda()

    angel_loss = line_angel.mean(dim = -1).cuda()
    return angel_loss

def a_loss(pred,target,img_h):
    return compute_angel(pred,target,img_h).mean().cuda()