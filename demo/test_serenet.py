# (SeReNet) inference script 
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physics-driven self-supervised learning for fast high-resolution robust 3D reconstruction of light-field microscopy
#       In submission, 2024
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


import numpy as np
from numpy.core.fromnumeric import sort
import h5py
import imageio
import tifffile
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
            # try:
            #     os.mkdir(folder) 
            # except:
            #     os.makedirs(folder) 
    return


def preDAO(lfstack, defocus=1):

    with torch.no_grad():
        D,H,W = lfstack.shape

        ref = F.interpolate(lfstack.unsqueeze(1), size = [round(lfstack.shape[-2] / 3 *13),round(lfstack.shape[-1] /3 *13)], mode='bicubic',align_corners=False).squeeze(1)

        for iteration in range(1):
            predLF = ref[0:1,:,:].repeat(lfstack.shape[0],1,1).unsqueeze(0).to(device)
            corr = fftshift(ifft2(fft2(ifftshift(predLF)) * fft2(ifftshift(ref.to(device).flip([-2,-1]).unsqueeze(0) )))).real.squeeze(0)
            shift_lf_x = corr.max(dim=-2).values.max(dim=-1).indices
            shift_lf_y = corr.max(dim=-1).values.max(dim=-1).indices # keepdim =False
            shift_lf_x = predLF.shape[-1]//2 - shift_lf_x
            shift_lf_y = predLF.shape[-1]//2 - shift_lf_y

            # remove defocus
            if defocus:
                k = (Si * shift_lf_y + Sj * shift_lf_x).sum() / (Si * Si + Sj * Sj).sum()
                shift_lf_y = shift_lf_y - k * Si
                shift_lf_x = shift_lf_x - k * Sj
                # print( (Si * shift_lf[:,0,0,0] + Sj * shift_lf[:,0,0,1]).sum())
            shift_lf = torch.cat([shift_lf_x.reshape(D,1,1,1), shift_lf_y.reshape(D,1,1,1)], dim=-1)
            shift_lf = shift_lf - shift_lf[0,0,0,:]

            # print(shift_lf.squeeze().permute(1,0))
            del predLF, corr

    shift_lf = shift_lf * 2 / ref.shape[-1] # very important
    lfstack = F.grid_sample(lfstack.unsqueeze(1), lf_grid + shift_lf, mode='bicubic', align_corners=False).squeeze(1)
    return lfstack


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default="./pth/serenet_pth/epoch-800.pth")
    # parser.add_argument('--model', default="../pth/fserenet_pth/epoch-800.pth")
    parser.add_argument('--resolution', default='101,1027,1027')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--inp_size', default='237')# 234
    parser.add_argument('--overlap', default='0')# 9
    parser.add_argument('--startframe', default='0')
    parser.add_argument('--psfshift', default="../psf/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-10u_zmax10u_zspacing0.2u_psfshift_49views.pt")
    parser.add_argument('--savevolume', default='1')
    parser.add_argument('--savebitdepth', default='32')
    parser.add_argument('--AOstar', default='0')
    parser.add_argument('--order', default='1')
    parser.add_argument('--config', default="../save/month_202505/config_tag/code/configs/train-serenet/serenet_Nnum13_bubtub.yaml")
    parser.add_argument('--codefolder', default="../save/month_202505/config_tag/code/")
    parser.add_argument('--savefolder', default="./data_3x3_recon/")
    parser.add_argument('--sourcefolder', default="./data_3x3/")
    parser.add_argument('--inputfile', default='./data_3x3/brain_slice.tif')

    args = parser.parse_args()

    if not os.path.exists(args.savefolder):
        try:
            os.mkdir(args.savefolder) 
        except:
            os.makedirs(args.savefolder) 

    sys.path.append(args.codefolder)

    import models
    # from utils import *
    import utils

    print(f'using codefolder {args.codefolder}')

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = 'cuda:0'
    # device = 'cpu'

    model_save = torch.load(args.model)
    model_name = model_save['model']['name']
    model = models.make(model_save['model'], load_sd=True).to(device)
    print(f'using model {args.model}')
    model.eval()

    d, h, w = list(map(int, args.resolution.split(',')))

    input_views = torch.tensor(config.get('input_views'))

    # load some system parameters from "config.yaml"
    Nnum = config.get('train_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('train_dataset')['wrapper']['args']['scanning']

    # load pre-computed psfshift (center of mass of light field phase-space psf)
    psfshift = torch.load(args.psfshift)
    if len(input_views) != psfshift.shape[0]:
        psfshift = psfshift[input_views,:,:]

    print(f'using psfshift  {args.psfshift}')
    if(psfshift.shape[1]>d):
        psfshift = psfshift[:, psfshift.shape[1]//2-d//2:psfshift.shape[1]//2+d//2+1, :]
    scale = Nnum / scanning
    shift = (psfshift[:,:,:] / scale)


    # files = [args.inputfile]
    files = os.listdir(args.sourcefolder)
    files = [os.path.join(args.sourcefolder,file) for file in files if ('.tif') in file]

    # depth decomposition meshgrid
    lfstack = torch.tensor(np.array(tifffile.imread(files[0]),dtype=np.float32))
    grid = utils.make_coord([i for i in [lfstack.shape[-2], lfstack.shape[-1]]], flatten=False).flip(-1).unsqueeze(0)
    new_grid =  grid + shift.view(-1, 2).flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([lfstack.shape[-1], lfstack.shape[-2]], device=grid.device)
    new_grid = new_grid.to(device)

    lfstack = lfstack[input_views,:,:]
    inp_size = int(args.inp_size)
    inp_size = min(inp_size, lfstack.shape[-1])

    # calculate overlap patched sigmoid weight if CUDA memory is limited
    overlap = int(args.overlap)
    overlapVolume = round(overlap * scale)
    if overlap:
        edge = torch.sigmoid((torch.arange(overlapVolume) - (overlapVolume-1)/2 ) / overlapVolume *15)
        weight = torch.cat([edge, edge.max()*torch.ones(round(inp_size*scale) - 2*len(edge)),edge.flip(0)],dim=0).view(-1,1) @ \
            torch.cat([edge, edge.max()*torch.ones(round(inp_size*scale) - 2*len(edge)),edge.flip(0)],dim=0).view(1,-1) + 1e-3
        # weight = weight.unsqueeze(0) # 1,h,w
        weight = weight.unsqueeze(0).to(device) # 1,h,w
        weight_cpu = weight.cpu()
        del edge
    else:
        weight = torch.ones(round(inp_size*scale), round(inp_size*scale))
        weight = weight.unsqueeze(0) # 1,h,w

    # preDAO if needed (optional)
    if int(args.AOstar):
        global lf_grid, Si, Sj
        [Si,Sj] = torch.meshgrid(*(torch.arange(Nnum).to(device)-Nnum//2,torch.arange(Nnum).to(device)-Nnum//2),indexing='ij')
        Si = Si.reshape(-1)[input_views]
        Sj = Sj.reshape(-1)[input_views]
        lf_grid = F.affine_grid(torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.]
            ], dtype=torch.float).unsqueeze(0), lfstack.unsqueeze(0).shape, align_corners=False).repeat(lfstack.shape[0],1,1,1).to(device) # D,H,W,2 # 2 means(-1 index, -2 index)
    
    
    # inference lfs one-by-one
    for file in files:
        print(file)
        thisFile = file
        lfstack = torch.tensor(np.array(tifffile.imread(thisFile),dtype=np.float32))
        
        t0 = time.time()
        lfstack = lfstack[input_views,:,:].to(device)
        lfstack = lfstack - torch.quantile(lfstack, 0.01)

        if int(args.AOstar):
            lfstack = preDAO(lfstack, defocus=1)
            torch.cuda.synchronize()
            print(f'preDAO time using {time.time() - t0} s')
            
        # depth decomposition 
        inp_all = utils.rolldim_gridsample(lfstack, new_grid.to(device)).unsqueeze(0)
        torch.cuda.synchronize()
        print(f'Depth decomposition time using {time.time() - t0} s')
        
        if inp_size == lfstack.shape[-1]:
            with torch.no_grad():
                ret = model( (inp_all).to(device), scale)
        else: # slower, but cost less memory
            # inp_all = inp_all.cpu() # or cuda out of memory
            print('patched and sigmoid-based image fusion for overlap')
            ret = torch.zeros([d,round(scale*inp_all.shape[-2]),round(scale*inp_all.shape[-1])])
            base = torch.zeros_like(ret)

            for h0 in [i*(inp_size-overlap) for i in range(1+math.ceil((inp_all.shape[-2]-inp_size)/(inp_size-overlap)))]:
                for w0 in [i*(inp_size-overlap) for i in range(1+math.ceil((inp_all.shape[-1]-inp_size)/(inp_size-overlap)))]:
                    inp = inp_all[:,:,:,h0:h0+inp_size,w0:w0+inp_size]

                    with torch.no_grad():
                        pred = model( (inp).to(device), scale)

                    pred[torch.isnan(pred)] = 0
                    pred[torch.isinf(pred)] = 0         
                    ret[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)]= \
                        ret[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)] + (pred * weight[:,0:pred.shape[-2],0:pred.shape[-1] ]).cpu()
                    base[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)]= \
                        base[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)] + weight_cpu[:,0:pred.shape[-2],0:pred.shape[-1] ]
                    pred = None
            ret = ret / base
        torch.cuda.synchronize()
        print(f'Depth decomposition and Deblurring and Fusion time using {time.time() - t0} s')

        ret = ret.cpu().squeeze()
        ret[ret<0] = 0

        ret = ret[20:-20, 50:-50, 50:-50] 
        savetype = np.uint16 if int(args.savebitdepth)==16 else np.float32

        if int(args.savevolume):
            # temp_name = args.savefolder + thisFile[-thisFile[-1::-1].find(filesdir[-1:0:-1] ):]
            temp_name = args.savefolder + thisFile[-thisFile[-1::-1].find('/'):-4] + '_' + model_name + '.tif'
            ensure_path(temp_name)
            # imwrite(temp_name, savetype(ret*100), imagej=True, metadata={'axes': 'ZYX'}, compression =None) # faster
            tifffile.imwrite(temp_name, savetype(ret*100), imagej=True, metadata={'axes': 'ZYX'}, compression = 'zlib') # less disk space
