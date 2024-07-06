# TWNet(Time-weighted Network) training toolbox 
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physical model-driven self-supervised learning permits fast, high-resolution, robust, broadly-generalized 3D reconstruction for scanning light-field microscopy
#       In submission, 2024
# Contact: ZHI LU (luz18@mails.tsinghua.edu.cn)
# Date: 7/7/2024
import argparse
import os

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
# from test import eval_psnr
import numpy as np
from  tifffile import imwrite
from torchvision.utils import flow_to_image
# from losses import *

# dataloader
def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset,'config':config})
    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=True, num_workers=4+5*int(tag == 'train'), pin_memory=True)
    return loader


# determine if resume to train a pretrained model (often used in training unexpectedly broken)
# determine the optimizer, learning rate
def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            for _ in range(epoch_start - 1):
                lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


# timeweight algorithm
# 0 <= indexmap <= 1
# orignially, indexmap have the save value for every position 
def timeweight(X_bicubic, indexmap):
    timePoints = X_bicubic.shape[1]

    weight = indexmap ** ( torch.linspace(1-timePoints, timePoints-1, timePoints, device=X_bicubic.device).abs().reshape(1,9,1,1) /2 )

    weight = weight / weight.sum(dim=1, keepdim=True)

    ret = X_bicubic * weight
    ret = ret.sum(1, keepdim=True)
    return ret


# train one epoch
def train(train_loader, model, optimizer):
    model.train()

    train_loss = utils.Averager()
    loss_fn = nn.L1Loss(reduction='mean') if config.get('loss_fn') is None else eval(config.get('loss_fn'))

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()


        B, N, H, W = batch['inp'].shape
        h, w= H // int(len(order) ** 0.5), W // int(len(order) ** 0.5)

        X1 = batch['inp'].reshape(B*N, 1, H, W)

        x0 = batch['inp'].reshape(B*N,  h, 3, w, 3).permute(0, 2, 4, 1, 3).reshape(B*N, 9, h, w)[:,order, :,:]


        # orignial mesh grid
        coord = utils.make_coord([i for i in [H,W]],flatten=False).flip(-1).unsqueeze(0).to(x0.device)
        coord_shift = coord.repeat(9,1,1,1)
        init_flow0 = scanningPosition - scanningPosition[8:9, :]
        
        # scanning optical flow induced mesh grid 
        coord_shift = coord_shift - 2*init_flow0.reshape(9,1,1,2) / torch.tensor([W, H], dtype=torch.float32, device=x0.device).reshape(1,1,1,2)


        X_bicubic  = F.grid_sample(x0.permute(1,0,2,3), coord_shift, mode='bicubic', align_corners=False).permute(1,0,2,3)

        X0 = X_bicubic[:,4:5,:,:]

        # a reference from itself
        target = utils.substitute_outlier(X1.clone(), X0, kernel_size=3, threshold=0.2)

        # generate timeweight coef. map
        mask = model(x0[:,inv_order, :,:])

        pred = mask * X1 + (1-mask) * timeweight(X_bicubic, mask)
        loss = loss_fn(pred, target)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None; # vpred = None
        # inp = None

    return train_loss.item()




def main(config_, save_path):
    global config, log, epoch, order, inv_order, scanningPosition


    config = config_
    log = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader = make_data_loader(config.get('train_dataset'), tag='train')

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')

    order = [6,7,8,5,2,1,0,3,4] # scanning order
    inv_order = [order.index(ii) for ii in range(9)]

    Nshift = int(len(order)**0.5)
    scanningPosition = (  torch.tensor([[order[ii]%Nshift,order[ii]//Nshift,] for ii in range(len(order))]) - torch.tensor([Nshift//2*1.0, Nshift//2*1.0]) ).cuda()

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        train_loss = train(train_loader, model, optimizer)
        if not (train_loss < 100000000):#or torch.isnan(train_loss) or  torch.isnan(train_loss):
            print(f'train_loss is {train_loss} so retrain')
            break
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }


        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))


        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))

    return train_loss
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train-twnet/twnet_config.yaml')
    parser.add_argument('--name', default='twnet')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name + '_pth'
    # if save_name is None:
    # save_name += '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('../pth', save_name)

    train_loss = main(config, save_path)
