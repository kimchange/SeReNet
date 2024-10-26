# SERENET (Self supervised Reconstruction Network for light field images) training script
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physics-driven self-supervised learning for fast high-resolution robust 3D reconstruction of light-field microscopy
#       In submission, 2024
# Contact: ZHI LU (luzhi@tsinghua.edu.cn)
# Date: 7/7/2024
import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import h5py

import datasets
import models
import utils

import numpy as np
# import taichi as ti

def make_data_loader(spec, tag='train'):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset,'psfshift':psfshift,'config':config})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=1+5*int(tag == 'train'), pin_memory=True)
    return loader


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
    log('model details: ')
    log(model)
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss(reduction='mean') if config.get('loss_fn') is None else eval(config.get('loss_fn'))
    train_loss = utils.Averager()

    volume_depth = config.get('train_dataset')['wrapper']['args']['volume_depth']


    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        rand_factor = 1. #torch.rand(1).item()
        rand_bg = torch.rand(1).item() * 0.
        batch['inp'] = batch['inp']*rand_factor + rand_bg * global_psf_sumdhw.reshape(1, -1, 1, 1, 1)
        batch['lf'] = batch['lf']*rand_factor + rand_bg * global_psf_sumdhw.reshape(1, -1, 1, 1)
        
        # pred = model(batch['inp'], Nnum / scanning)
        pred, pred_woupsample = model(batch['inp'], Nnum / scanning)

        if config.get('train_dataset')['wrapper']['args']['sample_centerview'] is not None:
            selected_index = torch.cat([torch.tensor([0]),torch.multinomial(selection_weight.view(-1), num_sample_views - 1, replacement=False, out=None)],dim=0)
        else:
            selected_index = torch.multinomial(selection_weight.view(-1), num_sample_views, replacement=False, out=None)

        if config.get('bpmode') == 'selectiviews':
            # forward projection
            psf = global_psf[selected_index,:,:,:].unsqueeze(0).cuda()
            predLF = torch.zeros(psf.shape[0:2]+pred.shape[3:], device='cuda:0')
            for blk in range(volume_depth):
                predLF = predLF + utils.imagingLFM(pred[:,:,blk*(1):(blk+1)*(1),:,:], \
                    psf[:,:,blk*(1):(blk+1)*(1),:,:])
        else:
            pass


        input_lf = batch['lf'][:, selected_index, :,:]
        if predLF.shape != input_lf.shape:
            predLF = torch.nn.functional.interpolate(predLF, size = input_lf.shape[2:], mode = 'bilinear', align_corners = False)
            # input_lf = torch.nn.functional.interpolate(input_lf, size = predLF.shape[2:], mode = 'bilinear', align_corners = False)


        loss = loss_fn(predLF,  input_lf) # + predLF.abs().mean()

        # loss = loss + 0.1*utils.tv_2d_loss(pred) + torch.nn.L1Loss()( pred, pred.abs() )
        if True:
            loss = loss + 1000*torch.nn.L1Loss()( pred_woupsample, pred_woupsample.abs() )

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        pred = None; loss = None; predLF = None; psf = None; selected_index = None
        # inp = None

    return train_loss.item()


def main(config_, save_path):
    global config,  epoch, Nnum, scanning, selection_weight, num_sample_views


    config = config_
    
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    # os.system(f'cp -r ../code {save_path}')

    train_loader = make_data_loader(spec=config.get('train_dataset'))


    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)



    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    

    Nnum = config.get('train_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('train_dataset')['wrapper']['args']['scanning']

    num_sample_views = config.get('train_dataset')['wrapper']['args']['sample_views']
    selection_weight = torch.ones(1,len(input_views)) *1.0


    if config.get('train_dataset')['wrapper']['args']['sample_centerview'] is not None:
        selection_weight[0,0] = 0


    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]


        train_loss = train(train_loader, model, optimizer)

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
    parser.add_argument('--config', default='./configs/train-serenet/serenet_config.yaml')
    parser.add_argument('--name', default='serenet')
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
    global global_psf, psfshift, input_views, log, global_psf_sumdhw

    log = utils.set_save_path(save_path)
    
    if config.get('psf').endswith('.mat'):
        f = h5py.File(config.get('psf'),'r')
        global_psf = f.get('psf')
        global_psf = np.array(global_psf)
        global_psf = torch.tensor(global_psf, dtype=torch.float32) # / 10
    elif config.get('psf').endswith('.pt'):
        global_psf = torch.load(config.get('psf'))
    else:
        raise NotImplementedError
    input_views = torch.tensor(config.get('input_views'))

    if len(global_psf.shape) == 5:
        # global_psf = global_psf / global_psf.max() / 2
        global_psf = global_psf.permute(2,1,0,4,3).reshape(global_psf.shape[1]*global_psf.shape[2], global_psf.shape[0], global_psf.shape[4], global_psf.shape[3]).contiguous()
        global_psf = global_psf[input_views,:,:,:]
    elif len(global_psf.shape) == 4:
        # input views has pre selected
        global_psf = global_psf.permute(0,1,3,2) # H,W,D,C in matlab --> C,D,W,H in pytorch --> permute to C,D,H,W
    else:
        raise NotImplementedError
    log(f'psf loaded, psf.shape = {global_psf.shape}')
    # global_psf = global_psf * 20
    if config.get('psfshift') is not None:
        psfshift = torch.load(config.get('psfshift')) # views already selected 
        log(f'using pre-computed psfshift '+config.get('psfshift'))
    else:
        psfshift = utils.get_centerofmass(global_psf)
        log(f'using psfshift center-of-mass of '+config.get('psf'))


    global_psf_sumdhw = global_psf.sum((1,2,3), keepdim=False).cuda()

    train_loss = main(config, save_path)
