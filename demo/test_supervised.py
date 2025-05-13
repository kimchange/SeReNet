import argparse
import os
import sys

from PIL import Image
from numpy.core.fromnumeric import sort
import imageio
import torch
from torchvision import transforms
import time
import math

import yaml
import h5py
import numpy as np
import tifffile


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
    parser.add_argument('--model', default='./pth/vcdnet_pth/epoch-800.pth')
    # parser.add_argument('--model', default='../pth/hylfmnet_pth/epoch-800.pth')
    parser.add_argument('--resolution', default='101,1027,1027')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--order', default='0')
    parser.add_argument('--inp_size', default=126)
    parser.add_argument('--overlap', default=15)
    parser.add_argument('--savebitdepth', default='32')
    parser.add_argument('--startframe', default='0')
    parser.add_argument('--savevolume', default='1')
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
    import utils

    print(f'using codefolder {args.codefolder}')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_save = torch.load(args.model)
    model_name = model_save['model']['name']
    model = models.make(model_save['model'], load_sd=True).cuda()
    # model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    print(f'using model {args.model}')


    d, h, w = list(map(int, args.resolution.split(',')))

    files = os.listdir(args.sourcefolder)
    files = [os.path.join(args.sourcefolder,file) for file in files if ('.tif') in file]
    # files = [args.inputfile]


    if int(args.order):
        files = sorted(files,key=lambda x:int(x[x.rfind('/')+8:-4]))
    else:
        files = sorted(files)

    files = files[int(args.startframe):]

    lfstack = torch.tensor(np.array(tifffile.imread(files[0]),dtype=np.float32))
    

    Nnum = 13
    scanning = 3
    scale = Nnum / scanning

    inp_size = int(args.inp_size)
    inp_size = min(inp_size, lfstack.shape[-1])

    overlap = int(args.overlap)
    overlapVolume = round(overlap * scale)
    if overlap:
        edge = torch.sigmoid((torch.arange(overlapVolume) - overlapVolume//2 ) / overlapVolume *15)
        weight = torch.cat([edge, edge.max()*torch.ones(round(inp_size*scale) - 2*len(edge)),edge.flip(0)],dim=0).view(-1,1) @ \
            torch.cat([edge, edge.max()*torch.ones(round(inp_size*scale) - 2*len(edge)),edge.flip(0)],dim=0).view(1,-1) + 1e-3
        weight = weight.unsqueeze(0) # 1,h,w
        del edge
    else:
        weight = torch.ones(round(inp_size*scale), round(inp_size*scale))
        weight = weight.unsqueeze(0) # 1,h,w


    for file in files:
        print(file)
        thisFile = file
        lfstack = torch.tensor(np.array(tifffile.imread(thisFile),dtype=np.float32))

        lfstack[lfstack.isnan()] = 0
        lfstack[lfstack.isinf()] = 0

        t0 = time.time()

        lfstack_high = np.percentile(lfstack, 99.9)
        lfstack_low = np.percentile(lfstack, 10.)

        print(f'preprocess normalize percentile low is {lfstack_low} high is {lfstack_high}')
        lfstack = ((lfstack-lfstack_low) / (lfstack_high-lfstack_low) ).clamp(0,1)
        # lfstack = (lfstack / 250).clamp(0,1)

        inp_all = lfstack.unsqueeze(0)

        if inp_size == lfstack.shape[-1]:
            with torch.no_grad():
                # ret = model( (inp_all).cuda(), torch.tensor([scale],dtype=torch.float32)).squeeze()\
                ret = model( (inp_all).cuda(), scale).squeeze()
        else:
            print('patched and sigmoid-based image fusion for overlap')

            ret = torch.zeros([d,h,w])
            base = torch.zeros_like(ret)
            for h0 in [i*(inp_size-overlap) for i in range(1+math.ceil((inp_all.shape[-2]-inp_size)/(inp_size-overlap)))]:
                for w0 in [i*(inp_size-overlap) for i in range(1+math.ceil((inp_all.shape[-1]-inp_size)/(inp_size-overlap)))]:
                    inp = inp_all[:,:,h0:h0+inp_size,w0:w0+inp_size]

                    patch_size = [d, round(inp.shape[-2]*scale),round(inp.shape[-1]*scale)]


                    with torch.no_grad():
                        pred = model( ((inp - 0) / 1).cuda(), scale)

                    pred[pred<0] = 0
                    pred[torch.isnan(pred)] = 0
                    pred[torch.isinf(pred)] = 0
                        
                    pred = pred.cpu()

                    ret[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)]= \
                        ret[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)] + pred * weight[:,0:pred.shape[-2],0:pred.shape[-1] ]
                    base[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)]= \
                        base[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)] + weight[:,0:pred.shape[-2],0:pred.shape[-1] ]
                    pred = None
            ret = ret / (base)
        torch.cuda.synchronize()
        print(time.time()-t0)
        # ret[ret<0.1] = 0
        ret = ret.cpu()
        print(ret.shape)
        ret = ret[20:-20, 50:-50, 50:-50] 
        savetype = np.uint16 if int(args.savebitdepth)==16 else np.float32
        if int(args.savevolume):
            temp_name = args.savefolder + thisFile[-thisFile[-1::-1].find('/'):-4] + '_' + model_name + '.tif'
            ensure_path(temp_name)
            tifffile.imwrite(temp_name, savetype(ret*2000), imagej=True, metadata={'axes': 'ZYX'}, compression = 'zlib') # less disk space