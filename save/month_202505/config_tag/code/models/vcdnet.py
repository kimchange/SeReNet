# PyTorch Implementation of VCDNet
# modified from https://github.com/xinDW/VCD-Net
# vcdnet have 2 main parts, upsampling(interpolation) network using subpixel conv and a Unet
# In PyTorch, subpixel conv is the combination of nn.Conv and nn.PixelShuffle() 
# we modify the first layer input channel number to 169, that is the number of spatial-angular views of our setup 
# Line 60,114 modified

from argparse import Namespace
import torch
import torch.nn as nn

from models import register


class pixelshuffle3dhw(nn.Module):
    def __init__(self, r):
        super(pixelshuffle3dhw, self).__init__()
        self.r = r
    def forward(self,x):
        ret = x.view([x.shape[0], x.shape[1] // self.r**2, self.r, self.r] +[int(x.shape[i]) for i in range(2,len(x.shape[2:])+2 )])
        ret = ret.permute(0,1,4,5,2,6,3)
        return ret.reshape(ret.shape[0],ret.shape[1], ret.shape[2], self.r*x.shape[3], self.r*x.shape[4])



class down(nn.Module):
    def __init__(self, inChannels, outChannels, kSize=3):
        super().__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(inChannels, outChannels, kSize, padding=(kSize-1)//2, stride=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU()
        ])
        self.maxpool = nn.MaxPool2d(kernel_size=(3,3), padding=(3-1)//2, stride=(2,2))
    def forward(self,x):
        y = self.conv(x)
        y = torch.cat((x, torch.zeros(x.shape[0], y.shape[1]-x.shape[1], x.shape[2], x.shape[3], device=x.device) ), 1) + y
        y = self.maxpool(y)
        return y

class up(nn.Module):
    def __init__(self, inChannels, outChannels, scale_factor=2, kSize=3):
        super().__init__()
        # self.upsample = nn.Upsample(scale_factor = scale_factor, mode='bilinear',align_corners=False)
        self.conv = nn.Sequential(*[
            nn.Conv2d(inChannels, outChannels, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(outChannels),
        ])
    def forward(self, x0, x1):
        y = torch.cat( (nn.functional.interpolate(x0, size = (x1.shape[2],x1.shape[3]) , mode='bilinear',align_corners=False), x1), 1) 
        y = self.conv(y)
        return y

@register('vcdnet')
class VCDNET(nn.Module):
    def __init__(self, inChannels=169, outChannels=101, channels_interp = 256, kSize=3):
        super().__init__()

        self.f1 = nn.Conv2d(inChannels, channels_interp, 7, padding=(7-1)//2, stride=1)
        # originally, vcdnet have 11*11=121 views, in our setup, we have 13*13 = 169 views, so inchannels are modified from 121 to 169

        self.interp = nn.Sequential(*[
            nn.PixelShuffle(2),
            nn.Conv2d(channels_interp // 4, channels_interp//2, kSize, padding=(kSize-1)//2, stride=1), 
            nn.PixelShuffle(2),
            nn.Conv2d(channels_interp // 2 // 4, channels_interp//4, kSize, padding=(kSize-1)//2, stride=1), 
            nn.PixelShuffle(2),
            nn.Conv2d(channels_interp // 4 // 4, channels_interp//8, kSize, padding=(kSize-1)//2, stride=1), 
            nn.PixelShuffle(2),
            nn.Conv2d(channels_interp // 8 // 4, channels_interp//16, kSize, padding=(kSize-1)//2, stride=1),
            nn.Conv2d(channels_interp//16, channels_interp//16, kSize, padding=(kSize-1)//2, stride=1), 
            nn.BatchNorm2d(channels_interp//16),
            nn.ReLU()
        ])

        pyramid_channels = [128, 256, 512, 512, 512]
        
        self.f2 = nn.Sequential(*[
            nn.Conv2d(channels_interp//16, 64, kSize, padding=(kSize-1)//2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])

        self.down0 = down(64, pyramid_channels[0])
        self.down1 = down(pyramid_channels[0], pyramid_channels[1])
        self.down2 = down(pyramid_channels[1], pyramid_channels[2])
        self.down3 = down(pyramid_channels[2], pyramid_channels[3])
        self.down4 = down(pyramid_channels[3], pyramid_channels[4])
        
        self.up4 = up(pyramid_channels[4] + pyramid_channels[3], pyramid_channels[3], scale_factor=1)
        self.up3 = up(pyramid_channels[3] + pyramid_channels[2], pyramid_channels[2])
        self.up2 = up(pyramid_channels[2] + pyramid_channels[1], pyramid_channels[1])
        self.up1 = up(pyramid_channels[1] + pyramid_channels[0], pyramid_channels[0])
        self.up0 = up(pyramid_channels[0] + 64, outChannels)

    def forward(self,x, scale):
        output_h, output_w = round((x.shape[-2])*scale), round((x.shape[-1])*scale)
        x = self.f1(x)
        x = self.interp(x)
        l0 = self.f2(x)
        l1 = self.down0(l0)
        l2 = self.down1(l1)
        l3 = self.down2(l2)
        l4 = self.down3(l3)
        l5 = self.down4(l4)
        l5 = nn.functional.interpolate(l5, size = (l4.shape[2],l4.shape[3]) , mode='bilinear',align_corners=False)

        l6 = self.up4(l5, l4)
        l7 = self.up3(l6, l3)
        l8 = self.up2(l7, l2)
        l9 = self.up1(l8, l1)
        l10 = torch.tanh(self.up0(l9, l0))
        l10 = torch.nn.functional.interpolate(l10, size = (output_h, output_w),mode='bilinear',align_corners=False).unsqueeze(1)# to match GT volume sampling rate


        return l10

