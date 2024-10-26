# PyTorch Implementation of SeReNet
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physics-driven self-supervised learning for fast high-resolution robust 3D reconstruction of light-field microscopy
#       In submission, 2024
# Contact: ZHI LU (luzhi@tsinghua.edu.cn)
# Date: 7/7/2024
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import GELU, ReLU, Sigmoid
from torch.nn.modules.conv import Conv2d, Conv3d
from torch.nn.modules.normalization import GroupNorm
import numpy as np


from models import register
import utils

class upsample3dhw(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale=scale

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=(1,self.scale,self.scale),align_corners=False,mode='trilinear')
        return x



@register('serenet')
class SERENET(nn.Module):
    def __init__(self, inChannels, outChannels=101, negative_slope=0.1, reset_param=False, usingbias=True):
        super().__init__()

        self.fusion =  nn.Sequential( # 8conv
            nn.Conv3d(inChannels, 64, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(64, 32, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(32, 16, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            upsample3dhw(2),
            nn.Conv3d(16,16,kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(16,8,kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            upsample3dhw(2),
            nn.Conv3d(8,8,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(8,4,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(4,1,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
        )

        if reset_param: self.reset_params()
        
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self, inp, scale):

        buffer = self.fusion(inp).squeeze(1)

        if len(buffer.shape) == 4:
            ret = F.interpolate(buffer, size = (round((inp.shape[-2])*scale), round((inp.shape[-1])*scale)),mode='bilinear',align_corners=False).unsqueeze(1)
        else:
            ret = F.interpolate(buffer, size=(inp.shape[-3],round((inp.shape[-2])*scale.item()), round((inp.shape[-1])*scale.item())) ,mode='trilinear',align_corners=False)

        
        if self.training:
            return ret, buffer
        else:
            return ret
