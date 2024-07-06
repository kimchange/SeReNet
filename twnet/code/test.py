# (TWNet) inference script 
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physical model-driven self-supervised learning permits fast, high-resolution, robust, broadly-generalized 3D reconstruction for scanning light-field microscopy
#       In submission, 2024
# Contact: ZHI LU (luz18@mails.tsinghua.edu.cn)
# Date: 7/7/2024
import argparse
import os
import sys
import math
from functools import partial
import re

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

import datasets
# import models
# import utils
import numpy as np
# import h5py
import imageio
import tifffile
from  tifffile import imwrite
import torch.nn.functional as F
from torchvision.utils import flow_to_image

def get_all_abs_path(source_dir):
    path_list = []
    for fpathe, dirs, fs in os.walk(source_dir):
        for f in fs:
            p = os.path.join(fpathe, f)
            path_list.append(p)
    return path_list


def timeweight(X_bicubic, indexmap):
    timePoints = X_bicubic.shape[1]

    weight = indexmap ** ( torch.linspace(1-timePoints, timePoints-1, timePoints, device=X_bicubic.device).abs().reshape(1,9,1,1) /2 )

    weight = weight / weight.sum(dim=1, keepdim=True)

    ret = X_bicubic * weight
    ret = ret.sum(1, keepdim=True)

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savefolder', default="../data/")
    parser.add_argument('--model', default="../pth/twnet_pth/epoch-800.pth")
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--savebitdepth', default='16')
    parser.add_argument('--startframe', default='0')
    parser.add_argument('--savevolume', default='1')
    parser.add_argument('--order', default='0')
    parser.add_argument('--codefolder', default="./")
    # parser.add_argument('--replacestr', default='twnet/')
    parser.add_argument('--inputfile', default='../data/demo_motionWDF_input.tif')
    args = parser.parse_args()


    if not os.path.exists(args.savefolder):
        try:
            os.mkdir(args.savefolder) 
        except:
            os.makedirs(args.savefolder) 


    sys.path.append(args.codefolder)

    import models
    import utils


    print(f'using codefolder {args.codefolder}')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_save = torch.load(args.model)
    model_name = model_save['model']['name']
    model = models.make(model_save['model'], load_sd=True).cuda()
    model.eval()
    print(f'using model {args.model}')

    files = [args.inputfile]

    if int(args.order):
        try:
            files = sorted(files, key=lambda x:int(x[x.rfind('/')+8:-4]) )
        except:
            files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
    else: 
        files = sorted(files)

    files = files[int(args.startframe):]

    lfstack = torch.tensor(np.array(tifffile.imread(files[0]),dtype=np.float32)).cuda()

    order=[6,7,8,5,2,1,0,3,4] # scanning order of used system
    inv_order = [order.index(ii) for ii in range(9)]
    Nshift = int(len(order)**0.5)
    scanningPosition = (  torch.tensor([[order[ii]%Nshift,order[ii]//Nshift,] for ii in range(len(order))]) - torch.tensor([Nshift//2*1.0, Nshift//2*1.0]) ).cuda()
    H,W = lfstack.shape[-2:]
    h,w = H//3,W//3

    # original mesh grid
    coord = utils.make_coord([i for i in [H,W]],flatten=False).flip(-1).unsqueeze(0).to(lfstack.device)
    coord_shift = coord.repeat(9,1,1,1)
    init_flow0 = scanningPosition - scanningPosition[8:9, :]

    # scanning optical flow induced mesh grid
    coord_shift = coord_shift - 2*init_flow0.reshape(9,1,1,2) / torch.tensor([W, H], dtype=torch.float32, device=lfstack.device).reshape(1,1,1,2)

    for file in files:
        print(file)
        thisFile = file
        lfstack = torch.tensor(np.array(tifffile.imread(thisFile),dtype=np.float32))
        lfstack_original = lfstack

        # lfstack_high = np.percentile(lfstack, 99.99)

        # print(f'preprocess normalize percentile low is {lfstack_low} high is {lfstack_high}')

        lfstack = lfstack.cuda()
        lfstack_high = torch.quantile(lfstack.reshape(lfstack.shape[0], -1), 0.99, dim=1)

        lfstack_low = torch.quantile(lfstack.reshape(lfstack.shape[0], -1), 0.05, dim=1)
        # lfstack_low = 0 # np.percentile(lfstack, 10.)

        B, N = 1, lfstack.shape[0]
        X1 = lfstack.reshape(B*N, 1, H, W)

        t0 = time.time()

        x0_all = lfstack.reshape(B*N,  h, 3, w, 3).permute(0, 2, 4, 1, 3).reshape(B*N, 9, h, w)[:,order, :,:]

        X_bicubic  = F.grid_sample(x0_all.permute(1,0,2,3), coord_shift, mode='bicubic', align_corners=False, padding_mode="reflection").permute(1,0,2,3)

        x0_all = x0_all[:,inv_order,:,:]
        x0_all = ((x0_all -lfstack_low.reshape(-1, 1, 1, 1)) / (lfstack_high.reshape(-1, 1, 1, 1)-lfstack_low.reshape(-1, 1, 1, 1)) )

        x0_all[x0_all<0] = 0

        pred = torch.zeros(B*N, 1, H, W, device=x0_all.device)
        mask = torch.zeros(B*N, 1, H, W, device=x0_all.device)                

        with torch.no_grad():
            blksize = 13
            for aa in range((lfstack.shape[0]-1)//blksize+1):
                mask[aa*blksize:aa*blksize+blksize, :,:,:] = model(x0_all[aa*blksize:aa*blksize+blksize, :,:,:].cuda() )

        pred = mask * X1 + (1-mask) * timeweight(X_bicubic, mask)
        ret = pred.squeeze(1).cpu()
        ret[ret<0] = 0

        print(f'time weight time using {time.time() - t0} s')

        savetype = np.uint16 if int(args.savebitdepth)==16 else np.float32

        # if int(args.savevolume):
        #     imwrite(args.savefolder + thisFile[-thisFile[-1::-1].find('/'):][0:-4]+'.tif',savetype(ret), imagej=True, metadata={'axes': 'ZYX'}, compression =None)
        if int(args.savevolume):
            temp_name = args.savefolder + thisFile[-thisFile[-1::-1].find('/'):-4] + '_' + model_name + '.tif'
            imwrite(temp_name,savetype(ret), imagej=True, metadata={'axes': 'ZYX'}, compression =None)

        # mask = mask.squeeze(1).cpu()
        # imwrite(args.savefolder[0:-1] + '_cv_mask/' + thisFile[-thisFile[-1::-1].find('/'):][0:-4]+'.tif', np.float32(mask[84,:,:]), compression =None)
