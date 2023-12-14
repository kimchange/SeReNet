# (TWNet) toolbox 
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physical model-driven self-supervised learning permits fast, high-resolution, robust, broadly-generalized 3D reconstruction for scanning light-field microscopy
#       In submission, 2023
# Contact: ZHI LU (luz18@mails.tsinghua.edu.cn)
# Date: 11/11/2023
import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW

import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

Tensor = torch.tensor

def substitute_outlier(img, substitute, kernel_size=3, threshold=0.2):
    '''
    img: B, C, H, W
    C == 1
    '''
    # 只有差异大于某个阈值，才做替换
    # 滤波的目的是去噪，基于噪声点与周围的点有明显差异

    mask_ind = list(range(kernel_size ** 2))
    deleted_ind = list(range(kernel_size ** 2))
    del deleted_ind[ len(deleted_ind)//2 ]

    B, C, H, W = img.shape
    img2 = F.pad(img, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), mode='reflect') # 在边缘区镜面
    img2 = F.unfold(img2 , kernel_size = kernel_size, padding=0)
    img2 = img2[:,mask_ind,:]
    deleted_neiborhood= img2[:,deleted_ind,:]
    threshold_img = deleted_neiborhood.mean(dim=1).reshape(B,C,H,W)


    mask = (img-threshold_img).abs()> (threshold * threshold_img)

    img[mask] = substitute[mask]
    # img[mask] = img2[mask]
    return img


def rand_shift(inp, shift = 3):
    B,C,H,W = inp.shape
    grid = make_coord([i for i in [H,W]], flatten=False).flip(-1).unsqueeze(0).to(inp.device).repeat(B,1,1,1) + ( (torch.rand(B,1,1,2)-0.5) * 2 * shift / torch.tensor([W,H]) ).to(inp.device)  # B,H,W,2 
    ret = F.grid_sample(inp, grid, mode='nearest', align_corners=False, padding_mode="border")
    return ret


class valueAverager():
    def __init__(self):
        self.n = 0.0
        self.min = 0.0
        self.max = 1000.0
        self.weight = torch.ones(int(self.max)) / self.max
        

    def add(self, x, n=1.0):
        if not torch.isnan(x).any():
            weight,_ = torch.histogram(x.view(-1).clamp(self.min,self.max).cpu(), bins=int(self.max), range=(self.min,self.max),density=True,)
            self.weight = (self.weight * self.n + weight * n) / (self.n + n)
            self.n += n

    def item(self):
        return self.weight

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
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
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
        'adam': Adam,
        'adamw': AdamW
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



def calc_psnr(sr, hr, rgb_range=1):
    diff = (sr - hr) / rgb_range

    valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
