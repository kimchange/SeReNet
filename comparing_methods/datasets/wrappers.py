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

from utils import *
import utils



@register('sere-wrapper')
class SERE(object):
    def __init__(self, dataset, randomSeed=None, inp_size = None, rand_size = None, volume_depth = None, augment=False,  \
        sample_views = 13, sample_centerview = None, roi = None, M = 63, Nnum = 13, \
            rand_factor = None, rand_interpolate = None, rand_galvo_voltage = None, RPN_noise = None, RGN_noise = None, zspacing = None, scanning = None,psfshift = None,config=None):
        self.dataset = dataset
        self.randomSeed = randomSeed
        self.sample_views = sample_views

        self.inp_size = inp_size
        self.roi = roi
        self.augment = augment
        self.volume_depth = volume_depth
        self.Nnum = Nnum
        self.scanning = scanning
        self.rand_size = rand_size
        self.rand_galvo_voltage = rand_galvo_voltage
        self.rand_factor = rand_factor
        self.rand_interpolate = rand_interpolate
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

        lfstack[lfstack<0] = 0

        # if lfstack.shape[-1] > self.Nnum * 51:
        #     h00 = random.randint(0, lfstack.shape[1] - self.Nnum * 51)
        #     w00 = random.randint(0, lfstack.shape[2] - self.Nnum * 51)
        #     lfstack = lfstack[:,h00:h00+self.Nnum * 51,w00:w00+self.Nnum * 51]
        # scale = random.uniform(1, self.scale_max)
        # scale = 13 / 3
        scale = self.Nnum / self.scanning

        # lfstack_max = lfstack.max()
        lfstack_max = torch.quantile(lfstack, 0.999)


        if self.roi is not None:
            lfstack = lfstack[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        
        hw_scale = 1. if self.rand_size is None else torch.rand(1).item()*(1-self.rand_size) + self.rand_size

        if self.inp_size is not None:
            h,w = round(self.inp_size * hw_scale) , round(self.inp_size * hw_scale)
            h0 = random.randint(0, lfstack.shape[1] - h)
            w0 = random.randint(0, lfstack.shape[2] - w)

            lfstack = lfstack[:,h0:h0+h,w0:w0+w]




        # lfstack = lfstack / 10
        if self.rand_factor is not None:
            rand_factor = torch.rand(1).item()*(1-self.rand_factor) + self.rand_factor
            lfstack = lfstack * rand_factor
        if self.rand_interpolate is not None:
            rand_interpolate = torch.rand(1).item()*(1-self.rand_interpolate) + self.rand_interpolate
            lfstack = torch.nn.functional.interpolate(lfstack.unsqueeze(0), scale_factor=rand_interpolate, align_corners=False, mode='bilinear').squeeze(0)


        lfstack_clean = lfstack.clone()

        if self.RPN_noise is not None: # add noise to achieve noise robustness
            poisson_lambda = self.RPN_noise[torch.multinomial(torch.ones(len(self.RPN_noise)),1)]
            lfstack = lfstack / lfstack_max * poisson_lambda
            lfstack = torch.poisson(lfstack)
            lfstack = lfstack * lfstack_max / poisson_lambda

        if self.RGN_noise is not None: # add noise to simulate bg offset and read noise of sCMOS
            gaussian_miu = self.RGN_noise[0]
            gaussian_sigma = self.RGN_noise[1] ** 0.5
            lfstack = lfstack + gaussian_miu + torch.randn(lfstack.shape)*gaussian_sigma
            lfstack[lfstack<0] = 0
            # lfstack_clean = lfstack_clean + gaussian_miu

        shift = (self.shift / scale)# .round().int()



        inp = utils.rolldim_shiftgridsample(lfstack, shift) # [C,D,H,W]


        return {
            'inp': inp,
            'lf': lfstack_clean,
        }


    def __len__(self):
        return len(self.dataset)

@register('rlfm-vcdnet')
class RLFMVCDnet(object):
    def __init__(self, dataset, randomSeed=None, inp_size = None, augment=False, volume_depth = None, \
        sample_q = None, noise = None, RPN_noise = None,RGN_noise = None, Nnum=None, scanning=None,roi = None, normalize_mode='percentile', normalize_low=None, normalize_high=None, normalize_clamp=None, config=None):

        self.dataset = dataset
        self.randomSeed = randomSeed

        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        # self.shift = get_shift()
        self.normalize_fn = normalize_percentile if normalize_mode == 'percentile' else normalize
        self.normalize_low = normalize_low
        self.normalize_high = normalize_high
        self.normalize_clamp = normalize_clamp
        self.noise = noise
        self.RPN_noise = RPN_noise
        self.RGN_noise = RGN_noise
        self.roi = roi

        if randomSeed is not None:
            torch.manual_seed(randomSeed)
            

    def __getitem__(self, idx):

        lfstack, volume = self.dataset[idx]
        # lfstack, volume = lfstack/10, volume/10
        lfstack[torch.isnan(lfstack)] = 0
        volume[torch.isnan(volume)] = 0
        lfstack_min = 0
        volume_min = 0
        if self.normalize_low is not None:
            lfstack_min, volume_min = np.percentile(lfstack, self.normalize_low), np.percentile(volume, self.normalize_low)

        
        if self.normalize_high is not None:
            lfstack_max, volume_max = np.percentile(lfstack, self.normalize_high), np.percentile(volume, self.normalize_high)
        else:
            # lfstack_max, volume_max = lfstack.max(), volume.max()
            lfstack_max, volume_max = np.percentile(lfstack, 99.9), np.percentile(volume, 99.9)
        

        scale = volume.shape[-1] / lfstack.shape[-1]

        if self.roi is not None:
            h,w = self.roi[1]-self.roi[0], self.roi[3]-self.roi[2]
            H,W = round(h*scale), round(w*scale)
            lfstack = lfstack[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            volume = volume[:, round(self.roi[0]*scale):round(self.roi[0]*scale)+H, round(self.roi[2]*scale):round(self.roi[2]*scale)+W]

        if self.inp_size is not None:
            h,w = self.inp_size, self.inp_size
            H,W = round(h*scale), round(w*scale)
            h0 = random.randint(0, lfstack.shape[1] - h)
            w0 = random.randint(0, lfstack.shape[2] - w)

            H0,W0 = round(h0*scale), round(w0*scale)
            lfstack = lfstack[:,h0:h0+h,w0:w0+w]
            volume = volume[:,H0:H0+H,W0:W0+W]


        if self.RPN_noise is not None: # add noise to achieve noise robustness
            poisson_lambda = self.RPN_noise[torch.multinomial(torch.ones(len(self.RPN_noise)),1)]
            lfstack = lfstack / lfstack_max * poisson_lambda
            lfstack = torch.poisson(lfstack)
            lfstack = lfstack * lfstack_max / poisson_lambda

        if self.RGN_noise is not None: # add noise to simulate bg offset and read noise of sCMOS
            gaussian_miu = self.RGN_noise[0]
            gaussian_sigma = self.RGN_noise[1] ** 0.5
            lfstack = lfstack + gaussian_miu + torch.randn(lfstack.shape)*gaussian_sigma
            lfstack[lfstack<0] = 0


        # lfstack = self.normalize_fn(lfstack,0.,99.99).clamp(0,1)
        # volume = self.normalize_fn(volume,0,99.99).clamp(0,1)
        lfstack = (lfstack-lfstack_min) / (1e-3 + lfstack_max - lfstack_min)#.clamp(0,1)
        volume = (volume -volume_min) / (1e-3 + volume_max - volume_min)#.clamp(0,1)

        lfstack[torch.isnan(lfstack)] = 0
        lfstack[torch.isinf(lfstack)] = 0
        volume[torch.isnan(volume)] = 0
        volume[torch.isinf(volume)] = 0




        if self.normalize_clamp is not None:
            lfstack = lfstack.clamp(0,1)
            volume = volume.clamp(0,1)


        return {
            'inp': lfstack,
            'scale': torch.tensor([scale],dtype=torch.float32),
            'volume': volume,
        }

    def __len__(self):
        return len(self.dataset)


@register('rlnet-wrapper')
class RLN(object):
    def __init__(self, dataset, randomSeed=None, inp_size = None, volume_depth = None, augment=False,  \
        roi = None, normalize_low=None, normalize_high=None, normalize_clamp = None,\
            Nnum = 13, rand_factor = None, RPN_noise = None,RGN_noise = None, scanning = None,psfshift = None,config=None):

        self.dataset = dataset
        self.randomSeed = randomSeed

        self.inp_size = inp_size
        self.roi = roi
        self.augment = augment
        self.normalize_low = normalize_low
        self.normalize_high = normalize_high
        self.normalize_clamp = normalize_clamp
        self.volume_depth = volume_depth

        self.Nnum = Nnum
        self.scanning = scanning
        self.rand_factor = rand_factor
        self.RPN_noise = RPN_noise
        self.RGN_noise = RGN_noise
        self.shift = psfshift if config.get('shiftmode') == 'psfcenterofmass' else None
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
        lfstack[lfstack.isnan()] = 0
        lfstack[lfstack<0] = 0

        scale = self.Nnum / self.scanning

        lfstack_min = 0
        volume_min = 0
        if self.normalize_low is not None:
            lfstack_min, volume_min = np.percentile(lfstack, self.normalize_low), np.percentile(volume, self.normalize_low)

        
        if self.normalize_high is not None:
            lfstack_max, volume_max = np.percentile(lfstack, self.normalize_high), np.percentile(volume, self.normalize_high)
        else:
            # lfstack_max, volume_max = lfstack.max(), volume.max()
            lfstack_max, volume_max = np.percentile(lfstack, 99.9), np.percentile(volume, 99.9)
        

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
            rand_factor = torch.rand(1).item()*(1-self.rand_factor) + self.rand_factor
            lfstack = lfstack * rand_factor
            volume = volume * rand_factor

        if self.RPN_noise is not None: # add noise to achieve noise robustness
            poisson_lambda = self.RPN_noise[torch.multinomial(torch.ones(len(self.RPN_noise)),1)]
            # lfstack_max = lfstack.max()
            lfstack_max = np.percentile(lfstack, 99.9)
            lfstack = lfstack / lfstack_max * poisson_lambda
            lfstack = torch.poisson(lfstack)
            # lfstack = lfstack * lfstack_max / poisson_lambda
            lfstack = lfstack * 20 
            volume = volume  / lfstack_max * poisson_lambda * 20
        if self.RGN_noise is not None: # add noise to achieve noise robustness
            gaussian_miu = self.RGN_noise[0]
            gaussian_sigma = self.RGN_noise[1] ** 0.5
            lfstack = lfstack + gaussian_miu + torch.randn(lfstack.shape)*gaussian_sigma
            lfstack[lfstack<0] = 0
            volume = volume + gaussian_miu / (self.Nnum*self.Nnum)

        lfstack = (lfstack-lfstack_min) / (1e-3 + lfstack_max - lfstack_min)#.clamp(0,1)
        volume = (volume -volume_min) / (1e-3 + volume_max - volume_min)#.clamp(0,1)

        lfstack[torch.isnan(lfstack)] = 0
        lfstack[torch.isinf(lfstack)] = 0
        volume[torch.isnan(volume)] = 0
        volume[torch.isinf(volume)] = 0

        if self.normalize_clamp is not None:
            lfstack = lfstack.clamp(0,1)
            volume = volume.clamp(0,1)

        lfinp = lfstack[self.input_views, :, :]

        shift = (self.shift / scale)# .round().int()
        inp = utils.rolldim_shiftgridsample(lfinp, shift) # [C,D,H,W]
        inp = inp.mean(dim=0, keepdim=False) # [D,H,W]
        inp = torch.nn.functional.interpolate(inp.unsqueeze(0), size=(volume.shape[-2], volume.shape[-1]), mode='bilinear', align_corners=False).squeeze(0)

        return {
            'inp': inp,
            'volume': volume,
        }


    def __len__(self):
        return len(self.dataset)
    

