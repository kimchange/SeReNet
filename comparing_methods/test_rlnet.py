# (RL=Net for light field reconstruction) inference script 
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physics-driven self-supervised learning for fast high-resolution robust 3D reconstruction of light-field microscopy
#       In submission, 2024
#       Yue Li, Yijun Su, et al., 
#       Incorporating the image formation process into deep learning improves network performance, Nature Methods 2022
# Contact: ZHI LU (luzhi@tsinghua.edu.cn)
# Date: 7/7/2024
import argparse
import os
import sys
import math
from functools import partial
import re

import yaml
import torch
import torch.nn.functional as F
from torch.fft import *
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
import numpy as np
from numpy.core.fromnumeric import sort
import h5py
import tifffile
from tifffile import imwrite
import time


def get_all_abs_path(source_dir):
    path_list = []
    for fpathe, dirs, fs in os.walk(source_dir):
        for f in fs:
            p = os.path.join(fpathe, f)
            path_list.append(p)
    return path_list

def ensure_path(filename):
    if ('/') in filename:
        folder = filename[0:filename.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder,exist_ok=True) 

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default="../pth/rlnet_pth/epoch-800.pth")
    # parser.add_argument('--model', default="../pth/fserenet_pth/epoch-800.pth")
    parser.add_argument('--resolution', default='101,1027,1027')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--inp_size', default='546')# 234
    parser.add_argument('--overlap', default='65')# 9
    parser.add_argument('--startframe', default='0')
    parser.add_argument('--psfshift', default="../psf/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-10u_zmax10u_zspacing0.2u_psfshift_49views.pt")
    parser.add_argument('--savevolume', default='1')
    parser.add_argument('--savebitdepth', default='32')
    parser.add_argument('--order', default='0')
    parser.add_argument('--config', default="./configs/train-supervised/train_rlnet_bubtub.yaml")
    parser.add_argument('--codefolder', default="./")
    parser.add_argument('--savefolder', default="../data_recon/")
    parser.add_argument('--sourcefolder', default="../data/")
    parser.add_argument('--inputfile', default='../data/brain_slice.tif')

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

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load model
    model_save = torch.load(args.model)
    model_name = model_save['model']['name']
    model = models.make(model_save['model'], load_sd=True).cuda()
    print(f'using model {args.model}')
    model.eval()

    d, h, w = list(map(int, args.resolution.split(',')))

    input_views = torch.tensor(config.get('input_views'))

    # load some system parameters from "config.yaml"
    Nnum = config.get('train_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('train_dataset')['wrapper']['args']['scanning']

    # load pre-computed psfshift (center of mass of light field phase-space psf)
    psfshift = torch.load(args.psfshift)
    print(f'using psfshift  {args.psfshift}')
    if(psfshift.shape[1]>d):
        psfshift = psfshift[:, psfshift.shape[1]//2-d//2:psfshift.shape[1]//2+d//2+1, :]
    scale = Nnum / scanning
    shift = (psfshift[:,:,:] / scale)


    # files = os.listdir(args.sourcefolder)
    # files = [os.path.join(args.sourcefolder,file) for file in files if ('.tif') in file]

    files = [args.inputfile]

    # depth decomposition meshgrid for refocusing
    lfstack = torch.tensor(np.array(tifffile.imread(files[0]),dtype=np.float32))
    grid = utils.make_coord([i for i in [lfstack.shape[-2], lfstack.shape[-1]]], flatten=False).flip(-1).unsqueeze(0)
    new_grid =  grid + shift.view(-1, 2).flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([lfstack.shape[-1], lfstack.shape[-2]], device=grid.device)
    # new_grid = new_grid.cuda()

    lfstack = lfstack[input_views,:,:]
    inp_size = int(args.inp_size)
    # inp_size = min(inp_size, lfstack.shape[-1])

    # calculate overlap patched sigmoid weight if CUDA memory is limited
    overlap = int(args.overlap)
    overlapVolume = round(overlap)
    if overlap:
        edge = torch.sigmoid((torch.arange(overlapVolume) - (overlapVolume-1)/2 ) / overlapVolume *15)
        weight = torch.cat([edge, edge.max()*torch.ones(round(inp_size) - 2*len(edge)),edge.flip(0)],dim=0).view(-1,1) @ \
            torch.cat([edge, edge.max()*torch.ones(round(inp_size) - 2*len(edge)),edge.flip(0)],dim=0).view(1,-1) + 1e-9
        # weight = weight.unsqueeze(0) # 1,h,w
        weight = weight.unsqueeze(0).cuda() # 1,h,w
        weight[:,-1,:] = 1e-9
        weight[:,:,-1] = 1e-9
        weight_cpu = weight.cpu()
        del edge
    else:
        weight = torch.ones(round(inp_size), round(inp_size))
        weight = weight.unsqueeze(0) # 1,h,w


    # inference lfs one-by-one
    for file in files:
        print(file)
        thisFile = file
        lfstack = torch.tensor(np.array(tifffile.imread(thisFile),dtype=np.float32))
        
        t0 = time.time()
        lfstack = lfstack[input_views,:,:].cuda()
        lfstack = lfstack / ( torch.quantile(lfstack,0.9999).abs() + 1e-3 )

        lfstack = lfstack.clamp(0,1)

        # begin digital refocusing
        # depth decomposition, then mean the spatial-angular views to get refocused 3D stacks.
        inp_all = utils.rolldim_gridsample(lfstack, new_grid.cuda()).unsqueeze(0) # 1, A, D, h, w
        torch.cuda.synchronize()
        inp_all = inp_all.mean(dim=1,keepdim=False) # 1, A, D, h, w
        inp_all = torch.nn.functional.interpolate(inp_all, size=(round(inp_all.shape[-2]*scale), round(inp_all.shape[-1]*scale)), mode='bilinear', align_corners=False)
        # Adapt the spatial sampling rate
        print(f'Upsampling and refocusing time using {time.time() - t0} s')
        if inp_size == inp_all.shape[-1]:
            with torch.no_grad():
                ret = model( (inp_all).cuda() )[:,1:2,:,:,:] # 0:1 is intermediate feature, just used in training process, and 1:2 is final prediction
        else: # slower, but cost less memory
            # inp_all = inp_all.cpu() # or cuda out of memory
            print('patched and sigmoid-based image fusion for overlap')
            ret = torch.zeros([d,round(inp_all.shape[-2]),round(inp_all.shape[-1])])
            base = torch.zeros_like(ret)

            for h0 in [i*(inp_size-overlap) for i in range(1+math.ceil((inp_all.shape[-2]-inp_size)/(inp_size-overlap)))]:
                for w0 in [i*(inp_size-overlap) for i in range(1+math.ceil((inp_all.shape[-1]-inp_size)/(inp_size-overlap)))]:
                    inp = inp_all[:,:,h0:h0+inp_size,w0:w0+inp_size]

                    with torch.no_grad():
                        pred = model( (inp).cuda() )[:,1:2,:,:,:]

                    pred[torch.isnan(pred)] = 0
                    pred[torch.isinf(pred)] = 0         
                    ret[:,round(h0):round((h0+inp_size)),round(w0):round((w0+inp_size))]= \
                        ret[:,round(h0):round((h0+inp_size)),round(w0):round((w0+inp_size))] + (pred * weight[:,0:pred.shape[-2],0:pred.shape[-1] ]).cpu()
                    base[:,round(h0):round((h0+inp_size)),round(w0):round((w0+inp_size))]= \
                        base[:,round(h0):round((h0+inp_size)),round(w0):round((w0+inp_size))] + weight_cpu[:,0:pred.shape[-2],0:pred.shape[-1] ]
                    pred = None
            ret = ret / base
        torch.cuda.synchronize()
        print(f'Prediction time using {time.time() - t0} s')

        ret = ret.cpu().squeeze()
        ret[ret<0] = 0

        ret = ret[20:-20, 50:-50, 50:-50] 
        savetype = np.uint16 if int(args.savebitdepth)==16 else np.float32

        if int(args.savevolume):
            # temp_name = args.savefolder + thisFile[-thisFile[-1::-1].find(filesdir[-1:0:-1] ):]
            temp_name = args.savefolder + thisFile[-thisFile[-1::-1].find('/'):-4] + '_' + model_name + '.tif'
            ensure_path(temp_name)
            # imwrite(temp_name, savetype(ret*100), imagej=True, metadata={'axes': 'ZYX'}, compression =None) # faster
            imwrite(temp_name, savetype(ret*100), imagej=True, metadata={'axes': 'ZYX'}, compression = 'zlib') # less disk space
