# (SeReNet) toolbox 
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physical model-driven self-supervised learning permits fast, high-resolution, robust, broadly-generalized 3D reconstruction for scanning light-field microscopy
#       In submission, 2024
# Contact: ZHI LU (luz18@mails.tsinghua.edu.cn)
# Date: 7/7/2024
import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD, Adam

# from scripts.imresizend import *
from PIL import Image
import torch.nn.functional as F
from torch.fft import *
# import taichi as ti
from typing import List, Optional, Tuple, Union

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
        if remove and (basename.startswith('_') and input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            print('removed !')
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    return log

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


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
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
