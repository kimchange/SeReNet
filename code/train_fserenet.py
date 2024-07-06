# FSERENET (Finetune SeReNet) training script
# Note that FSERENET finetune process is based on a pretrained SeReNet
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
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import h5py

import datasets
import models
import utils
import numpy as np
# import taichi as ti

def make_data_loader(spec, tag=''):
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




def freeze_serenet(model): # 
    # 0, 2, 4, 6, 9, 11 select conv layers

    model.fusion[0].weight.requires_grad = False
    model.fusion[0].bias.requires_grad = False

    model.fusion[2].weight.requires_grad = False
    model.fusion[2].bias.requires_grad = False

    model.fusion[4].weight.requires_grad = False
    model.fusion[4].bias.requires_grad = False

    model.fusion[6].weight.requires_grad = False
    model.fusion[6].bias.requires_grad = False

    model.fusion[9].weight.requires_grad = False
    model.fusion[9].bias.requires_grad = False

    model.fusion[11].weight.requires_grad = False
    model.fusion[11].bias.requires_grad = False



def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()

        if config.get('freeze_serenet') is not None:
            freeze_serenet(model)
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
        if config.get('load_pretrain') is not None:
            sv_file = torch.load(config['load_pretrain'])
            model = models.make(config['model']).cuda()
            model_dict=model.state_dict()
            pretrained_dict = {k: v for k, v in sv_file['model']['sd'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            if config.get('freeze_serenet') is not None:
                freeze_serenet(model)
            epoch_start = 1
        else:
            model = models.make(config['model']).cuda()
            epoch_start = 1

        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])

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


    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)


        rand_bg = torch.rand(1).item() * 1

        pred = model(batch['inp'] + rand_bg * global_psf_sumdhw.reshape(1, -1, 1, 1, 1), Nnum / scanning)


        synthetic_volume = batch['volume'].unsqueeze(1) + rand_bg # / (Nnum*Nnum)

        loss = loss_fn(pred,  synthetic_volume) 

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        pred = None; loss = None
        # inp = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, epoch, Nnum, scanning

    config = config_

    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    # os.system(f'cp -r ../code {save_path}')

    train_loader = make_data_loader(spec=config.get('train_dataset'), tag='train')
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')    
    Nnum = config.get('train_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('train_dataset')['wrapper']['args']['scanning']


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
    parser.add_argument('--config', default='./configs/train-fserenet/fserenet_config.yaml')
    parser.add_argument('--name', default='fserenet')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # ti.init(arch=ti.gpu, kernel_profiler=True)

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
        if config.get('psfcrop') is not None:
            cropsize = config.get('psfcrop')
            global_psf = global_psf[:,cropsize['z']:global_psf.shape[1]-cropsize['z'],cropsize['x']:global_psf.shape[2]-cropsize['x'],cropsize['y']:global_psf.shape[3]-cropsize['y']].contiguous()
        # global_psf[global_psf<5e-4] = 0

        global_psf = global_psf[input_views,:,:,:]
    elif len(global_psf.shape) == 4:
        # input views has pre selected
        global_psf = global_psf.permute(0,1,3,2) # H,W,D,C in matlab --> C,D,W,H in pytorch --> permute to C,D,H,W
    else:
        raise NotImplementedError


    print(f'psf loaded, psf.shape = {global_psf.shape}')
    psfshift = utils.get_centerofmass(global_psf)


    global_psf_sumdhw = global_psf.sum((1,2,3), keepdim=False).cuda()

    if config.get('bpmode') is None:
        global_psf = None

    train_loss = main(config, save_path)
