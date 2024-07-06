# PyTorch Implementation of FSeReNet
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physical model-driven self-supervised learning permits fast, high-resolution, robust, broadly-generalized 3D reconstruction for scanning light-field microscopy
#       In submission, 2024
# Contact: ZHI LU (luz18@mails.tsinghua.edu.cn)
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
import models

class upsample3dhw(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale=scale

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=(1,self.scale,self.scale),align_corners=False,mode='trilinear')
        return x


def pad_to3d(x, stride=[1,1,1]):
    # https://www.coder.work/article/7536422
    # just match size to 2^n
    assert len(x.shape)==5, 'input shoud be 5d'
    d, h, w = x.shape[-3:]
    
    if d % stride[-3] > 0:
        new_d = d + stride[-3] - d % stride[-3]
    else:
        new_d = d

    if h % stride[-2] > 0:
        new_h = h + stride[-2] - h % stride[-2]
    else:
        new_h = h
    if w % stride[-1] > 0:
        new_w = w + stride[-1] - w % stride[-1]
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

def unpad3d(x, pad):
    assert (len(pad) + 1) % 2, 'pad length should be an even number'
    x = x[:,...,pad[4]:x.shape[-3]-pad[5], pad[2]:x.shape[-2]-pad[3], pad[0]:x.shape[-1]-pad[1]  ]
    # if pad[2]+pad[3] > 0:
    #     x = x[:,:,pad[2]:-pad[3],:]
    # if pad[0]+pad[1] > 0:
    #     x = x[:,:,:,pad[0]:-pad[1]]
    return x





class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, ndims = 3, kSize=3, negative_slope=0.1):
        super().__init__()
        convblock = getattr(nn, 'Conv%dd' %ndims)
        self.conv = nn.Sequential(*[
            convblock(inChannels, outChannels, kSize, padding=(kSize-1)//2, stride=1),
            # nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(negative_slope, inplace=True)
        ])
    def forward(self,x):
        y = self.conv(x)
        return y

class UNET(nn.Module):
    def __init__(self, channels, kSize=3,ndims=3, negative_slope=0.1):
        super().__init__()
        maxpoolblock = getattr(nn, 'MaxPool%dd' %ndims)
        interpolationMode = 'bilinear'  if ndims == 2 else 'trilinear'
        self.pooling = maxpoolblock(2)
        self.interpolation = lambda x:F.interpolate(x, scale_factor=2, mode=interpolationMode, align_corners=False)

        conv = getattr(nn, 'Conv%dd' %ndims)

        self.down_layers = nn.Sequential(*[
            ConvBlock(channels[ii], channels[ii+1], ndims = ndims, kSize=kSize, negative_slope=negative_slope)
            for ii in range(0, len(channels)-1, 1)
        ])
        
        self.up_layers = [ConvBlock(channels[-1], channels[-1], ndims = ndims, kSize=kSize, negative_slope=negative_slope)]
        for ii in range(len(channels)-1, 0, -1):
            self.up_layers.append(ConvBlock(channels[ii]*2, channels[ii-1], ndims = ndims, kSize=kSize, negative_slope=negative_slope))

        self.up_layers = nn.Sequential(*self.up_layers)


    def forward(self,x):
        x, pads = pad_to3d(x, [2**len(self.down_layers), 2**len(self.down_layers), 2**len(self.down_layers)] )
        x_history = []
        for level, down_layer in enumerate( self.down_layers):
            x = down_layer(x)
            x_history.append(x)
            x = self.pooling(x)

        for level, up_layer in enumerate( self.up_layers):
            x = up_layer(x)
            if len(x_history):
                x = torch.cat( (self.interpolation(x), x_history.pop()),dim=1 )

        x = unpad3d(x,pads)
        return x

@register('fserenet')
class FSERENET(nn.Module):
    def __init__(self, inChannels, outChannels=101, negative_slope=0.1):
        super().__init__()

        self.fusion =  nn.Sequential( # 8conv
            nn.Conv3d(inChannels, 64, kernel_size=(3,3,3),stride=1,padding=(1,1,1) ),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(64, 32, kernel_size=(3,3,3),stride=1,padding=(1,1,1) ),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3,3,3),stride=1,padding=(1,1,1) ),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(32, 16, kernel_size=(3,3,3),stride=1,padding=(1,1,1) ),nn.LeakyReLU(negative_slope, inplace=True),
            upsample3dhw(2),
            nn.Conv3d(16,16,kernel_size=(3,3,3),stride=1,padding=(1,1,1)),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(16,8,kernel_size=(3,3,3),stride=1,padding=(1,1,1)),nn.LeakyReLU(negative_slope, inplace=True),
            upsample3dhw(2),
        )

        self.decoder = UNET(channels=[8, 8, 16], kSize=3, ndims=3, negative_slope=negative_slope)# models.make({'name': 'unet', 'args':{'ndims':3, 'channels':[4, 8, 16],'kSize':3}})

        self.final_conv = nn.Sequential( 
            nn.Conv3d(8,1,kernel_size=(3,3,3),stride=1, padding=(1,1,1)),nn.LeakyReLU(negative_slope, inplace=True),
        )
            



    def forward(self, inp, scale):

        volume = self.fusion(inp)
        volume = self.decoder(volume)
        ret = self.final_conv(volume).squeeze(1)

        if len(ret.shape) == 4:

            ret = F.interpolate(ret, size = (round((inp.shape[-2])*scale), round((inp.shape[-1])*scale)),mode='bilinear',align_corners=False).unsqueeze(1)
        else:
            ret = F.interpolate(ret, size=(inp.shape[-3],round((inp.shape[-2])*scale.item()), round((inp.shape[-1])*scale.item())) ,mode='trilinear',align_corners=False)

        return ret

