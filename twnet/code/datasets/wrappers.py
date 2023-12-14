import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
# from utils import to_pixel_samples
# from utils import *
import utils


@register('twnet-wrapper')
class TWNETWRAPPER(object):
    def __init__(self, dataset, randomSeed=None, inp_size = None, normalize_mode=None, Nnum = 13, scanning = 3, config=None, order = [6,7,8,5,2,1,0,3,4]):

        self.dataset = dataset
        self.randomSeed = randomSeed

        self.inp_size = inp_size
        self.Nnum = Nnum
        self.scanning = scanning

        self.normalize_fn = utils.normalize_percentile if normalize_mode == 'percentile' else None
        # self.normalize_low_high = normalize_low_high
        self.input_views = torch.tensor(config.get('input_views'))#b[weight>=1]
        self.config = config
        self.order = order

        if randomSeed is not None:
            torch.manual_seed(randomSeed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(randomSeed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            random.seed(randomSeed)
            np.random.seed(randomSeed)
            

    def __getitem__(self, idx):


        lfstack = self.dataset[idx]
        if self.normalize_fn is not None:
            lfstack_high = np.percentile(lfstack, 99.99)
            lfstack = lfstack / lfstack_high

        if self.inp_size is not None:
            h,w = min(self.inp_size, lfstack.shape[-2]), min(self.inp_size, lfstack.shape[-2])
            h0 = random.randint(0, lfstack.shape[-2] - h) // self.scanning * self.scanning
            w0 = random.randint(0, lfstack.shape[-1] - w) // self.scanning * self.scanning # to be multiple of 3

            lfstack = lfstack[:,h0:h0+h,w0:w0+w]

        lfinp = lfstack[self.input_views, :, :]

        return {
            'inp': lfinp,
        }


    def __len__(self):
        return len(self.dataset)

