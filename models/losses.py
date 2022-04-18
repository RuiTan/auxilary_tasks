import torch
import torch.nn.functional as F
import numpy as np
from libKMCUDA import kmeans_cuda
from torch.nn import MSELoss
import wandb

def cross_entropy(input, target, device=None, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)
    
def cross_entropy_with_auxilary(input, target, device=None, weight=None, lam=5, reduction='mean', ignore_index=255):
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
    ce = F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)
    wandb.log({"loss_ce": ce})
    # print('辅助Loss计算开始---')
    aux = auxilary_loss(input, target, device)
    wandb.log({"loss_aux": aux})
    wandb.log({"lam": lam})
    # print('---辅助Loss计算结束')
    return ce + lam*aux


def auxilary_loss(input, target, device):
    input2 = input.clone()
    input2 = input2[:,0,:,:]
    input_ = input2.reshape(-1, 1)
    # 获取聚类结果
    _, estimator = kmeans_cuda(input_.cpu().detach().numpy(), 2, verbosity=0, seed=3)
    # reshape，构建tensor
    estimator = estimator.reshape(input2.shape).astype(np.uint8)
    km_label = torch.tensor(estimator).to(device)
    # 将聚类到0（表示真实不变区域，且预测概率较低）的像素点设为0
    input2[km_label == 0] = 0
    shape = input2.shape
    label = torch.tensor(np.zeros(shape)).to(device).float()
    mseloss = torch.nn.MSELoss(reduction='mean')
    # print(input2.type(), label.type())
    return mseloss(input2, label)


def focal_loss(input, target, device, weight=None):
    alpha = 0.9
    gamma = 2
    input = input[:,1,:,:]
    target = target.squeeze()
    input = input.flatten()
    target = target.flatten()
    ones = torch.ones_like(input)
    alpha_t = target*alpha + (ones-target)*(1-alpha)
    p_t = input*target + (ones-input)*(ones-target) + 1e-10
    fl = - alpha_t * torch.pow((ones - p_t), gamma) * torch.log(p_t)
    result = torch.mean(fl)
    return result

