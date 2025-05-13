"""
# File       : TV.py.py
# Time       ：2023/2/10 上午9:02
# Author     ：chenlu
# version    ：python 3.9
# Description：
"""


import torch
import torch.nn as nn
from torch.autograd import Variable


class TVLoss2D(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss2D,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

def main1():
    x = Variable(torch.FloatTensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]).
                 view(1, 2, 3, 3),requires_grad=True)
    print(x.size)
    addition = TVLoss2D()
    z = addition(x)
    print(z)


class TVLoss3D(nn.Module):
    def __init__(self, w=None):
        super(TVLoss3D,self).__init__()
        if w is None:
            w = [3, 1, 1]
        self.w = w

    def forward(self, x):
        b, c, d, h, w = x.size()
        count_d = h * w * (d - 1) * c * b
        count_h =  (h - 1) * w * d * c * b
        count_w = h * (w - 1) * d * c * b

        d_tv = torch.pow((x[:, :, 1:, :, :] - x[:, :, :d - 1, :, :]), 2).sum()
        h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w-1]), 2).sum()

        loss =  self.w[0]*d_tv/count_d + self.w[1]*h_tv/count_h + self.w[2]*w_tv/count_w

        return loss

def main():
    x = Variable(torch.FloatTensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]).
                 view(1, 1, 2, 3, 3),requires_grad=True)
    print(x.size())
    addition = TVLoss3D()
    z = addition(x)
    print(z)

if __name__ == '__main__':
    main()
