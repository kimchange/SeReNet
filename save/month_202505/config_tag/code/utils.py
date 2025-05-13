# (SeReNet) toolbox 
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physics-driven self-supervised learning for fast high-resolution robust 3D reconstruction of light-field microscopy
#       Nature, Methods 2025
# Contact: ZHI LU (luzhi@tsinghua.edu.cn)
# Date: 7/7/2024
import os
import time
import shutil
import math

import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
# from scripts.imresizend import *
from PIL import Image
import torch.nn.functional as F
from torch.fft import *
# import taichi as ti
from typing import List, Optional, Tuple, Union
import torchvision
from collections import namedtuple

Tensor = torch.tensor

cmap = [[0 ,0 ,0],
 [0,0,255],
 [255,0,255],
 [255,0,0],
 [255,255,0],
 [255,255,255]]
cmap = torch.tensor(cmap,dtype=torch.float32).unsqueeze(0).unsqueeze(1)
cmap = torch.nn.functional.interpolate(cmap, size=[256,3],mode='bilinear',align_corners=True)
cmap = cmap.squeeze()
cmap = cmap.type(torch.uint8)

def normalize(im):  
    assert im.dtype in [np.uint8, np.uint16]
    
    x = im.astype(np.float32)
    max_ = 255. if im.dtype == np.uint8 else 65536.
    # x = x / (max_ / 2.) - 1.
    x = x / (max_)
    return x

def normalize_percentile(im, low=0.2, high=99.8):
    p_low  = np.percentile(im, low)
    p_high = np.percentile(im, high)

    eps = 1e-3
    x = (im - p_low) / (p_high - p_low + eps)
    # print('%.2f-%.2f' %  (np.min(x), np.max(x)))
    return x

def tv_3d_loss(volume):
    gv = list(torch.gradient(volume.squeeze(), dim=[-3,-2,-1]) )
    loss = sum([( gv[ii].abs()  ).mean() for ii in range(len(gv)) ])
    return loss 

def tv_2d_loss(volume):
    gv = list(torch.gradient(volume.squeeze(), dim=[-2,-1]) )
    loss = sum([( gv[ii].abs()  ).mean() for ii in range(len(gv)) ])
    return loss 

class NLLMPGLoss(torch.nn.Module):
    def __init__(self, miu=120,var=256):
        super(NLLMPGLoss, self).__init__()
        self.miu = miu
        self.var = var
        self.sigma = int(var ** 0.5)
        
        # Numerical approximation of gaussian distribution
        if self.sigma < 1:
            self.x_range = torch.tensor([0.], dtype=torch.float32)
            self.weight = torch.tensor([1.], dtype=torch.float32)
        else:
            gaussian_sample_steps = min(self.sigma*6+1, 16*6+1)
            self.x_range = torch.linspace(-3*self.sigma, 3*self.sigma, steps=gaussian_sample_steps)
            # print(self.x_range)
            pdf = torch.exp(-0.5 * (self.x_range / self.sigma).pow(2))
            self.weight = pdf / pdf.sum()

    def forward(self, input, target):

        wdf = (input.unsqueeze(-1) + self.x_range.to(input.device) * 0).clamp(min=0)

        # print(f'wdf.shape is {wdf.shape}')
        x = (target.unsqueeze(-1) - self.miu + self.x_range.to(input.device)).clamp(min=0)

        neg_poissonnlllossvec = -0.5*torch.log(2*torch.pi*x + 1e-8) + x*torch.log(wdf+1e-8) - x*torch.log(x+1e-8) -wdf+x
        loss = - torch.log( (torch.exp(neg_poissonnlllossvec - neg_poissonnlllossvec.max(dim=-1,keepdim=True).values) * self.weight.to(input.device)).sum(-1) +1e-8) - neg_poissonnlllossvec.max(dim=-1,keepdim=False).values
        loss = loss.mean()

        return loss

def imagingLFM(volume, psf):
    return blurringLFM(volume, psf).sum(-3)

def imagingLFMFFT3(volume, psf):
    # volume b, 1, d, H, W
    # psf b, a, d, h, w
    # ret b,a,h,w
    shapevolume = volume.shape
    shapepsf = psf.shape
    volume = F.pad(volume, [(i//2*2+1-i)*j for i in shapevolume[-1:-3:-1] for j in [0,1]], "constant", 0)#np.percentile(volume.detach().cpu(),5))
    volume = F.pad(volume, tuple([i//2 for i in shapepsf[-1:-3:-1] for j in range(2)]),"constant", 0)#np.percentile(volume.detach().cpu(),5))
    psf = F.pad(psf, tuple([i//2 for i in shapevolume[-1:-3:-1] for j in range(2)]),"constant", 0)
    sumupXG1 = fftn(ifftshift(volume, dim=(-3,-2,-1) ) , dim=(-3,-2,-1)) * fftn(ifftshift(psf.flip(-3), dim=(-3,-2,-1) ) , dim=(-3,-2,-1)) 
    sumupXG1 = sumupXG1.sum(-3, keepdim=False)  / shapepsf[-3]
    ret = fftshift( ifftn( sumupXG1, dim=(-2,-1)) , dim=(-2,-1)).real
    return ret[:,...,shapepsf[-2]//2:shapepsf[-2]//2+shapevolume[-2], shapepsf[-1]//2:shapepsf[-1]//2+shapevolume[-1]]

def blurringLFM(volume, psf):
    shapevolume = volume.shape
    shapepsf = psf.shape
    volume = F.pad(volume, [(i//2*2+1-i)*j for i in shapevolume[-1:-3:-1] for j in [0,1]], "constant", 0)#np.percentile(volume.detach().cpu(),5))
    volume = F.pad(volume, tuple([i//2 for i in shapepsf[-1:-3:-1] for j in range(2)]),"constant", 0)#np.percentile(volume.detach().cpu(),5))
    psf = F.pad(psf, tuple([i//2 for i in shapevolume[-1:-3:-1] for j in range(2)]),"constant", 0)
    ret = fftshift(ifft2(fft2(ifftshift(volume)) * fft2(ifftshift(psf)))).real
    return ret[:,...,shapepsf[-2]//2:shapepsf[-2]//2+shapevolume[-2], shapepsf[-1]//2:shapepsf[-1]//2+shapevolume[-1]]

def get_shift(zspacing = 0.2*1e-6, M = 72.5, NA = 1.4, MLPitch = 100*1e-6, depth=101, Nnum = 13):
    print('geometrical optics line simplification is used')
    selfshift = torch.zeros(Nnum,Nnum,depth,2)
    shiftgrid = torch.stack(torch.meshgrid(torch.arange(Nnum)-Nnum//2,torch.arange(Nnum)-Nnum//2,torch.arange(depth)-depth//2,indexing='ij'),dim=-1)
    pixel_size = MLPitch/Nnum
    xxxx = 1 #1.25
    selfshift[:,:,:,0] = shiftgrid[:,:,:,0]*shiftgrid[:,:,:,2]*zspacing*M**2 *2*NA/np.sqrt(M**2-NA**2)/Nnum/pixel_size / xxxx 
    selfshift[:,:,:,1] = shiftgrid[:,:,:,1]*shiftgrid[:,:,:,2]*zspacing*M**2 *2*NA/np.sqrt(M**2-NA**2)/Nnum/pixel_size / xxxx 
    selfshift = selfshift.reshape([-1,selfshift.size(2),selfshift.size(3)])
    return selfshift

def get_centerofmass(psf):
    """ calculate psf center of mass.(psf>0 == True)
    input: [C,D,H,W]
    return:[C,D,2]
    """
    C, D, H, W = psf.shape# [-2], psf.shape[-1]
    # psf = psf.view(-1, H, W).unsqueeze(-1)
    psf = psf.reshape(C, D, H, W, 1)
    hwgrid = torch.stack(torch.meshgrid(torch.arange(H) - (H-1)/2,torch.arange(W) - (W-1)/2,indexing='ij'), dim=-1).unsqueeze(0) # 1, H, W, 2
    psf_centerofmass = torch.zeros(C, D, 2)
    for c in range(C):
        psf_centerofmass[c, :,:] = (  psf[c, :,:,:,:] * hwgrid / (psf[c, :,:,:,:].sum((-3,-2),keepdim =True) + 1e-9)  ).sum((-3,-2))

    # return (psf*hwgrid/(psf.sum((-3,-2),keepdim =True) + 1e-9)).sum((-3,-2)).view(C, D, 2)
    return psf_centerofmass



# todo
def rolldim_shiftgridsample(input, shift):
    """ shift correspondingly and concat
    input: [C,H,W]
    shift: [C,D,2]
    return:[C,D,H,W]
    """
    grid = make_coord([i for i in [input.shape[-2], input.shape[-1]]], flatten=False).flip(-1).unsqueeze(0).to(input.device) # 1,H,W,2
    new_grid = grid + shift.view(-1, 2).flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([input.shape[-1], input.shape[-2]], device=input.device) # C*D, H,W,2
    input = input.unsqueeze(1).repeat(1, shift.shape[1], 1, 1).view(-1, 1, input.shape[-2], input.shape[-1]) # C*D,1,H,W
    inp_all = F.grid_sample(input, new_grid, mode='bilinear', align_corners=False).view(shift.shape[0], shift.shape[1], input.shape[-2], input.shape[-1])

    # del grid, new_grid, input, shift
    # torch.cuda.empty_cache()

    return inp_all

def rolldim_gridsample(input, new_grid):
    # grid = make_coord([i for i in [input.shape[-2], input.shape[-1]]], flatten=False).flip(-1).unsqueeze(0).to(input.device)
    # new_grid = grid + shift.view(-1, 2).flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([input.shape[-1], input.shape[-2]], device=input.device)
    C = input.shape[0]
    D = new_grid.shape[0] // C

    input = input.unsqueeze(1).repeat(1, D, 1, 1).view(-1, 1, input.shape[-2], input.shape[-1]) # C*D,1,H,W
    inp_all = F.grid_sample(input, new_grid, mode='bilinear', align_corners=False).view(C, D, input.shape[-2], input.shape[-1])

    # del new_grid, input
    # torch.cuda.empty_cache()

    return inp_all



def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def ravel_multi_index(multi_index, shape):
    out = 0
    multi_index = list(multi_index)
    for dim in shape[1:]:
        out += multi_index[0]
        out *= dim
        multi_index.pop(0)
    out += multi_index[0]
    return out



# this is a perceptual loss
class PerceptualLoss_rgb(torch.nn.Module):
    def __init__(self, model_name, model_path, reduction='mean'):
        super(PerceptualLoss_rgb, self).__init__()

        vgg16_first1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        vgg16_first2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        vgg16_first3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        vgg19_first12 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        if model_name == 'vgg16_1':
            self.model = vgg16_first1
        if model_name == 'vgg16_2':
            self.model = vgg16_first2
        if model_name == 'vgg16_3':
            self.model = vgg16_first3
        if model_name == 'vgg19_12':
            self.model = vgg19_first12

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.reduction = reduction

    def forward(self, input, target):
        """
        input : pred_lf
        target : ground truth
        """
        input = torch.permute(input, (1, 0, 2, 3))
        input = input.repeat((1, 3, 1, 1))
        target = torch.permute(target, (1, 0, 2, 3))
        target = target.repeat((1, 3, 1, 1))

        self.model.to(input.device)
        input_pc = self.model(input)
        target_pc = self.model(target)

        l2 = torch.square(input_pc - target_pc)
        if self.reduction == 'sum':
            return torch.sum(l2)
        if self.reduction == 'mean':
            return torch.mean(l2)



# Learned perceptual metric
class LPIPS_vgg16_single_channel(nn.Module):
    def __init__(self, pretrained=True, net='vgg', version='0.1', lpips=True, spatial=False, 
        pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        """ Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(LPIPS_vgg16_single_channel, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'%
                ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if(self.pnet_type in ['vgg','vgg16']):
            # net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        else:
            raise NotImplementedError
        # elif(self.pnet_type=='alex'):
        #     net_type = pn.alexnet
        #     self.chns = [64,192,384,256,256]
        # elif(self.pnet_type=='squeeze'):
        #     net_type = pn.squeezenet
        #     self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns)

        self.net = vgg16_singlechannel(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if(pretrained):
                if(model_path is None):
                    import inspect
                    import os
                    model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'pretrained_weights/lpips_weights/v%s/%s.pth'%(version,net)))

                if(verbose):
                    print('Loading model from: %s'%model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)          

        if(eval_mode):
            self.eval()

    def forward(self, in0, in1, retPerLayer=False):
        # input shound have value range [0,1]


        # if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
        #     in0 = 2 * in0  - 1
        #     in1 = 2 * in1  - 1

        in0_input = in0.reshape(-1,1,in0.shape[-2], in0.shape[-1])
        in1_input = in1.reshape(-1,1,in1.shape[-2], in1.shape[-1])

        # max_intensity_value = np.quantile(in1_input.cpu(), 0.999)
        # in0_input = in0_input / (abs(max_intensity_value) + 1e-6)
        # in1_input = in1_input / (abs(max_intensity_value) + 1e-6)

        
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0_input), self.scaling_layer(in1_input)) if self.version=='0.1' else (in0_input, in1_input)



        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            # feats0[kk], feats1[kk] = normalize_double_tensor(outs0[kk], outs1[kk])
            diff = ( (feats0[kk]-feats1[kk])**2 )
            diffs[kk] = diff # .mean(dim=1, keepdim=True) #.reshape( (in0.shape[0], in0.shape[1]) + diff.shape[-2:] )

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True).mean(dim=0,keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True).mean(dim=0,keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val

def normalize_double_tensor(in_feat, ref_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(ref_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps), ref_feat/(norm_factor+eps)

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()

        # value range [-1,1]
        # self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        # self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

        # value range [0, 1]
        # self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None].mean(dim=1,keepdim=True))
        # self.register_buffer('scale', torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None].mean(dim=1,keepdim=True))

        # self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None].mean(dim=1,keepdim=True))
        # self.register_buffer('scale', torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None].mean(dim=1,keepdim=True))
        self.shift =  torch.Tensor([0.485, 0.456, 0.406]).mean(dim=0).item()
        self.scale =  torch.Tensor([0.229, 0.224, 0.225]).mean(dim=0).item()
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class vgg16_singlechannel(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, lpips=True, use_dropout=False):
        super(vgg16_singlechannel, self).__init__()
        # vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        vgg_pretrained_features = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        # vgg_pretrained_features = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        self.slice1.add_module(str(0), torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1) ))

        self.slice1[0].weight = nn.Parameter( vgg_pretrained_features[0].weight.sum(dim=1,keepdim=True) )
        self.slice1[0].bias = nn.Parameter( vgg_pretrained_features[0].bias )


        for x in range(1, 4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out






class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        if not np.isnan(v):
            self.v = (self.v * self.n + v * n) / (self.n + n)
            self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and ( input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            print('removed !')
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer

def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True,device='cpu'):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device, dtype=torch.float32)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range

    valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
