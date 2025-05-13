# Implementation of HYLFM-Net (https://github.com/kreshuklab/hylfm-net)
# we modify the first layer input channel number to 169, that is the number of spatial-angular views of our setup 
# hylfmnet originally, designed for 19*19=361 views, output depth is 61
# and originally hylfmnet has 361, 488, 488, 244, 244, 427, ---> 7*(61*h*w, 3d), 7,7,7,1
# 488 is 61*8, 244 is 61*4, 427 is 61*7, they are all multiples of output depth
# we have 13*13=169 spatial-angular views, output depth is 101
# so our modified hylfmnet has 169,808, 808, 404, 404, 707, ---> 7*(101*h*w, 3d),7,7,7,1
# # Line 92-100, 116 modified

from argparse import Namespace
import torch
import torch.nn as nn

from models import register

class Crop(nn.Module):
    def __init__(self, *slices: slice):
        super().__init__()
        self.slices = slices

    def extra_repr(self):
        return str(self.slices)

    def forward(self, input):
        return input[self.slices]

class RES2BLOCK(nn.Module):
    def __init__(self, inChannels, outChannels, keepoutsize=True, kSize=3):
        super().__init__()
        self.act = nn.ReLU()
        PADDING = (kSize-1)//2 if keepoutsize else 0
        if inChannels != outChannels:
            self.projection_layer = nn.Conv2d(inChannels, outChannels, kernel_size=1)
        else:
            self.projection_layer = None

        if keepoutsize:
            self.crop = None
        else:
            crop_each_side = [2 * (ks // 2) for ks in (kSize,)*2]
            self.crop =  Crop(..., *[slice(c, -c) for c in crop_each_side])
        self.conv = nn.Sequential(*[
            nn.Conv2d(inChannels, outChannels, kSize, padding=PADDING, stride=1),
            nn.ReLU(),
            nn.Conv2d(outChannels, outChannels, kSize, padding=PADDING, stride=1),
        ])
    def forward(self, x):
        y = self.conv(x)
        if self.crop is not None:
            x = self.crop(x)
        if self.projection_layer is None:
            y = y + x
        else:
            y = y + self.projection_layer(x)
        return self.act(y)

class RES3DBLOCK(nn.Module):
    def __init__(self, inChannels, outChannels, keepoutsize=False, kSize=3):
        super().__init__()
        self.act = nn.ReLU()
        PADDING = (kSize-1)//2 if keepoutsize else 0
        if inChannels != outChannels:
            self.projection_layer = nn.Conv2d(inChannels, outChannels, kernel_size=1)
        else:
            self.projection_layer = None

        if keepoutsize:
            self.crop = None
        else:
            crop_each_side = [2 * (ks // 2) for ks in (kSize,)*3]
            self.crop =  Crop(..., *[slice(c, -c) for c in crop_each_side])
        self.conv = nn.Sequential(*[
            nn.Conv3d(inChannels, outChannels, kSize, padding=PADDING, stride=1),
            nn.ReLU(),
            nn.Conv3d(outChannels, outChannels, kSize, padding=PADDING, stride=1),
        ])
    def forward(self, x):
        y = self.conv(x)
        if self.crop is not None:
            x = self.crop(x)
        if self.projection_layer is None:
            y = y + x
        else:
            y = y + self.projection_layer(x)
        return self.act(y)
@register('hylfmnet')
class HYLFMNET(nn.Module):
    def __init__(self, inChannels=169, outChannels=101, kSize=3):
        super().__init__()

        self.res2d = nn.Sequential(*[ # originally, inChannels are 361, outChannels are 61, now inChannels are 169, outChannels are 101
            RES2BLOCK(inChannels, 8*(outChannels), keepoutsize = True, kSize=kSize),  # 169->808
            RES2BLOCK(8*(outChannels), 8*(outChannels), keepoutsize = True, kSize=kSize),  # 808->808
            nn.ConvTranspose2d(8*(outChannels), 4*(outChannels),  kernel_size=2, stride=2, padding=0, output_padding=0), # 808->404
            RES2BLOCK(4*(outChannels), 4*(outChannels), keepoutsize = True, kSize=kSize), # 404->404
            nn.Conv2d(4*(outChannels), 7*(outChannels), kernel_size=1),# 404->707
            nn.ReLU(),
        ])
        self.c2z = lambda ipt, c_in_3d=7: ipt.view(ipt.shape[0], c_in_3d, (outChannels), *ipt.shape[2:]) # 707->7, 101
        self.res3d = nn.Sequential(*[
            RES3DBLOCK(7,7, keepoutsize = True, kSize=kSize),
            nn.ConvTranspose3d(7,7, kernel_size=(3, 2, 2),stride=(1, 2, 2),padding=(1, 0, 0),output_padding=0),
            RES3DBLOCK(7,7, keepoutsize = True, kSize=kSize),
            RES3DBLOCK(7,7, keepoutsize = True, kSize=kSize),
            nn.Conv3d(7, 1, kernel_size=1),
            nn.Sigmoid(),
        ])
            

    def forward(self,x,scale):
        output_h, output_w = round((x.shape[-2])*scale), round((x.shape[-1])*scale)
        x = self.res2d(x)
        x = self.c2z(x)
        x = self.res3d(x).squeeze(1)
        ret = torch.nn.functional.interpolate(x, size = (output_h, output_w),mode='bilinear',align_corners=False).unsqueeze(1)# to match GT volume sampling rate

        return ret

