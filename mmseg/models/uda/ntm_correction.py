'''
Author: qibin.he qibin.he@outlook.com
Date: 2023-09-24 19:51:00
LastEditors: qibin.he qibin.he@nio.com
LastEditTime: 2023-10-29 18:49:16
'''

import random

import torch
from torch import nn

from mmseg.models.uda.teacher_module import EMATeacher
from mmseg.models.utils.dacs_transforms import get_mean_std, strong_transform
from mmseg.models.utils.masking_transforms import build_mask_generator

from mmseg.ops import resize
import numpy as np


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)


class weighting_matrix(nn.Module):
    def __init__(self, num_classes):
        super(weighting_matrix, self).__init__()
        self.num_classes = num_classes
        self.weight = nn.parameter.Parameter(torch.ones(
            num_classes, num_classes) / num_classes)
        self.identity = nn.parameter.Parameter(torch.zeros(
            num_classes, num_classes) - torch.eye(num_classes))
    
    def forward(self):
        ind = np.diag_indices(self.num_classes)
        with torch.no_grad():
            self.weight[ind[0], ind[1]] = -10000. * torch.ones(self.num_classes).cuda()

        w = torch.softmax(self.weight, dim = 1) + self.identity.detach()        
        return w


class NTMCorrection(nn.Module):

    def __init__(self, num_classes, all_point_cnt, vol_weight=0, con_weight=0):
        super(NTMCorrection, self).__init__()
        self.num_classes = num_classes
        # self.ntm = nn.parameter.Parameter(num_classes * torch.eye(num_classes) - torch.ones(num_classes))
        self.ntm = nn.parameter.Parameter((num_classes-1) * torch.eye(num_classes) + torch.ones(num_classes))
        self.posterior = nn.parameter.Parameter(torch.ones((num_classes, 1)) / \
                                      num_classes + torch.rand((num_classes,1))*0.1) 
        self.all_point_cnt = all_point_cnt
        self.vol_weight = vol_weight
        self.con_weight = con_weight
        if self.con_weight > 0:
            self.convex_matrix = weighting_matrix(num_classes)
            self.loss_mse = torch.nn.MSELoss(reduction='sum').cuda()
            
