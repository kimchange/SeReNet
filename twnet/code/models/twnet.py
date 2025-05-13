# PyTorch Implementation of TWNet
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physics-driven self-supervised learning for fast high-resolution robust 3D reconstruction of light-field microscopy
#       Nature, Methods 2025
# Contact: ZHI LU (luzhi@tsinghua.edu.cn)
# Date: 7/7/2024
from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

from models import register


import torch
import torch.nn.functional as F

def pad_to(x, stride):
    # https://www.coder.work/article/7536422
    # just match size to 2^n
    if len(x.shape) == 4:
        d = 0
        h, w = x.shape[-2:]
    elif len(x.shape) == 5:
        d, h, w = x.shape[-3:]
    
    if d % stride > 0:
        new_d = d + stride - d % stride
    else:
        new_d = d

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w

    ld, ud = int((new_d-d) / 2), int(new_d-d) - int((new_d-d) / 2)
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh, ld, ud)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    assert (len(pad) + 1) % 2, 'pad length should be an even number'
    x = x[:,...,pad[4]:x.shape[-3]-pad[5], pad[2]:x.shape[-2]-pad[3], pad[0]:x.shape[-1]-pad[1]  ]
    # if pad[2]+pad[3] > 0:
    #     x = x[:,:,pad[2]:-pad[3],:]
    # if pad[0]+pad[1] > 0:
    #     x = x[:,:,:,pad[0]:-pad[1]]
    return x


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, ndims = 2, kSize=3):
        super().__init__()
        convblock = getattr(nn, 'Conv%dd' %ndims)
        self.conv = nn.Sequential(*[
            convblock(inChannels, outChannels, kSize, padding=(kSize-1)//2, stride=1),
            # nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        ])
    def forward(self,x):
        y = self.conv(x)
        return y


@register('twnet')
class TWNet(nn.Module):
    def __init__(self, ndims=2, kSize=3, channels=[18, 18, 36, 36, 72]):
        super().__init__()

        self.ndims =ndims

        pyramid_channels = channels

        convblock = getattr(nn, 'Conv%dd' %ndims)

        maxpoolblock = getattr(nn, 'MaxPool%dd' %ndims)
        interpolationMode = 'bilinear' if ndims == 2 else 'trilinear'

        self.conv_init = convblock(9, pyramid_channels[0], kSize, padding=(kSize-1)//2, stride=1)

        self.down_layers = nn.Sequential(*[
            ConvBlock(pyramid_channels[ii], pyramid_channels[ii+1], ndims = ndims, kSize=kSize)
            for ii in range(0, len(pyramid_channels)-1, 1)
        ])

        # self.poolings = [maxpoolblock(2) for ii in range(0, len(pyramid_channels)-1, 1)]
        self.pooling = maxpoolblock(2)
        self.interpolation = lambda x:F.interpolate(x, scale_factor=2, mode=interpolationMode, align_corners=False)

        self.up_layers = [ConvBlock(pyramid_channels[-1], pyramid_channels[-1], ndims = ndims, kSize=kSize)]
        for ii in range(len(pyramid_channels)-1, 0, -1):
            self.up_layers.append(ConvBlock(pyramid_channels[ii]*2, pyramid_channels[ii-1], ndims = ndims, kSize=kSize))

        self.up_layers = nn.Sequential(*self.up_layers)

        self.conv_final = nn.Sequential(*[
            convblock(pyramid_channels[0], pyramid_channels[0]*9, 1, padding=0, stride=1),
            nn.PixelShuffle(3),
            nn.ReLU(),
            convblock(pyramid_channels[0], 1, kSize, padding=(kSize-1)//2, stride=1),
            nn.Sigmoid(),
        ])

        


    def forward(self, x):
        # x = torch.cat((x0, x1), dim=1)
        # x = x1
        # x = F.pixel_unshuffle(x,3)

        if self.ndims !=len(x.shape[2:]):
            if self.ndims == 2:
                x = x.squeeze(1)
            else:
                x = x.unsqueeze(1)

        x, pads = pad_to(x, 2**len(self.down_layers))
        
        x = self.conv_init(x)

        x_history = []
        for level, down_layer in enumerate( self.down_layers):
            x = down_layer(x)
            x_history.append(x)
            x = self.pooling(x)

        for level, up_layer in enumerate( self.up_layers):
            x = up_layer(x)
            if len(x_history):
                x = torch.cat( (self.interpolation(x), x_history.pop()),dim=1 )


        x = unpad(x,pads)

        x = self.conv_final(x)


        return x

