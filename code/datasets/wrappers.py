import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.functional import align_tensors
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
# from utils import to_pixel_samples
from utils import *
import utils


@register('sere-wrapper')
class SERE(object):
    def __init__(self, dataset, randomSeed=None, inp_size = None, volume_depth = None, augment=False,  \
        sample_views = 13, sample_centerview = None, roi = None, \
            M = 63, Nnum = 13, rand_factor = None, RPN_noise = None, RGN_noise = None, zspacing = None, scanning = None,psfshift = None,config=None):
        self.dataset = dataset
        self.randomSeed = randomSeed
        self.sample_views = sample_views

        self.inp_size = inp_size
        self.roi = roi
        self.augment = augment
        self.volume_depth = volume_depth
        self.Nnum = Nnum
        self.scanning = scanning
        self.rand_factor = rand_factor
        self.RPN_noise = RPN_noise
        self.RGN_noise = RGN_noise
        self.shift = psfshift if config.get('shiftmode') == 'psfcenterofmass' else utils.get_shift(zspacing = eval(zspacing), M = M, NA = 1.4, MLPitch = 100*1e-6,depth = volume_depth, Nnum=Nnum)

        self.input_views = torch.tensor(config.get('input_views'))#b[weight>=1]
        self.config = config
        

        if randomSeed is not None:
            torch.manual_seed(randomSeed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(randomSeed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            random.seed(randomSeed)
            np.random.seed(randomSeed)
            

    def __getitem__(self, idx):

        #lfstack, volume = self.dataset[idx]
        lfstack = self.dataset[idx]
        lfstack[lfstack.isnan()] = 0
        lfstack = lfstack[self.input_views,:,:]

        if lfstack.shape[-1] > self.Nnum * 51:
            h00 = random.randint(0, lfstack.shape[1] - self.Nnum * 51)
            w00 = random.randint(0, lfstack.shape[2] - self.Nnum * 51)
            lfstack = lfstack[:,h00:h00+self.Nnum * 51,w00:w00+self.Nnum * 51]
        # scale = random.uniform(1, self.scale_max)
        # scale = 13 / 3
        scale = self.Nnum / self.scanning

        if self.roi is not None:
            lfstack = lfstack[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]


        if self.rand_factor is not None:
            rand_factor = torch.rand(1).item()*0.9 + 0.1
            lfstack = lfstack * rand_factor
        if self.RPN_noise is not None: # add noise to achieve noise robustness
            poisson_lambda = self.RPN_noise[torch.multinomial(torch.ones(len(self.RPN_noise)),1)]
            lfstack_max = lfstack.max()
            lfstack = lfstack / lfstack_max * poisson_lambda
            lfstack = torch.poisson(lfstack)
            lfstack = lfstack * lfstack_max / poisson_lambda
        if self.RGN_noise is not None: # add noise to achieve noise robustness
            gaussian_miu = self.RGN_noise[0]
            gaussian_sigma = self.RGN_noise[1] ** 0.5
            lfstack = lfstack + gaussian_miu + torch.randn(lfstack.shape)*gaussian_sigma
            lfstack[lfstack<0] = 0

        lfinp = lfstack

        shift = (self.shift / scale)# .round().int()

        if self.inp_size is not None:
            h,w = self.inp_size, self.inp_size
            h0 = random.randint(0, lfstack.shape[1] - h)
            w0 = random.randint(0, lfstack.shape[2] - w)

            lfstack = lfstack[:,h0:h0+h,w0:w0+w]


        inp = utils.rolldim_shiftgridsample(lfinp[:,h0:h0+h,w0:w0+w], shift) # [C,D,H,W]


        return {
            'inp': inp,
            'lf': lfstack,
        }


    def __len__(self):
        return len(self.dataset)

@register('fsere-wrapper')
class SERESF(object):
    def __init__(self, dataset, randomSeed=None, inp_size = None, volume_depth = None, augment=False,  \
        sample_views = 13, sample_centerview = None, roi = None, \
            M = 63, Nnum = 13, rand_factor = None, RPN_noise = None,RGN_noise = None, zspacing = None, scanning = None,psfshift = None,config=None):

        self.dataset = dataset
        self.randomSeed = randomSeed
        self.sample_views = sample_views

        self.inp_size = inp_size
        self.roi = roi
        self.augment = augment

        self.volume_depth = volume_depth

        self.Nnum = Nnum
        self.scanning = scanning
        self.rand_factor = rand_factor
        self.RPN_noise = RPN_noise
        self.RGN_noise = RGN_noise
        self.shift = psfshift if config.get('shiftmode') == 'psfcenterofmass' else utils.get_shift(zspacing = eval(zspacing), M = M, NA = 1.4, MLPitch = 100*1e-6,depth = volume_depth, Nnum=Nnum)
        
        self.input_views = torch.tensor(config.get('input_views'))#b[weight>=1]
        self.config = config


        

        if randomSeed is not None:
            torch.manual_seed(randomSeed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(randomSeed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            random.seed(randomSeed)
            np.random.seed(randomSeed)
            



    def __getitem__(self, idx):

        lfstack, volume = self.dataset[idx]
        # lfstack = self.dataset[idx]
        lfstack[lfstack.isnan()] = 0
        lfstack[lfstack<0] = 0

        scale = self.Nnum / self.scanning

        if self.roi is not None:
            lfstack = lfstack[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            volume = volume[:, self.roi[0]*scale:self.roi[1]*scale, self.roi[2]*scale:self.roi[3]*scale]


        if self.inp_size is not None:
            h,w = self.inp_size, self.inp_size
            H,W = round(h*scale), round(w*scale) 
            h0 = random.randint(0, lfstack.shape[1] - h) // self.scanning * self.scanning
            w0 = random.randint(0, lfstack.shape[2] - w) // self.scanning * self.scanning

            H0,W0 = round(h0*scale), round(w0*scale)
            # grid = grid[:,h0:h0+h,w0:w0+w, :]
            lfstack = lfstack[:,h0:h0+h,w0:w0+w]
            volume = volume[:, H0:H0+H, W0:W0+W]

        if self.rand_factor is not None:
            rand_factor = torch.rand(1).item()*0.9 + 0.1
            lfstack = lfstack * rand_factor
            volume = volume * rand_factor

        if self.RPN_noise is not None: # add noise to achieve noise robustness
            poisson_lambda = self.RPN_noise[torch.multinomial(torch.ones(len(self.RPN_noise)),1)]
            lfstack_max = lfstack.max()
            lfstack = lfstack / lfstack_max * poisson_lambda
            lfstack = torch.poisson(lfstack)
            lfstack = lfstack * lfstack_max / poisson_lambda
        if self.RGN_noise is not None: # add noise to achieve noise robustness
            gaussian_miu = self.RGN_noise[0]
            gaussian_sigma = self.RGN_noise[1] ** 0.5
            lfstack = lfstack + gaussian_miu + torch.randn(lfstack.shape)*gaussian_sigma
            lfstack[lfstack<0] = 0
            volume = volume + gaussian_miu / (self.Nnum*self.Nnum)

        lfinp = lfstack[self.input_views, :, :]

        shift = (self.shift / scale)# .round().int()

        # # coarse version but save memory
        # inp = rolldim(lfinp, -shift.round().int()) # [C,D,H,W]

        inp = utils.rolldim_shiftgridsample(lfinp, shift) # [C,D,H,W]


        lfstack = lfstack[self.input_views,:,:]
        # index_bp = torch.arange(len(self.input_views))

        return {
            'inp': inp,
            'scale': torch.tensor([scale],dtype=torch.float32),
            'lf': lfstack,
            # 'index': index_bp,
            'volume': volume,
        }


    def __len__(self):
        return len(self.dataset)