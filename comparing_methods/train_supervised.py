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
from tifffile import imwrite
import torch.nn.functional as F
import numpy as np



def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset,'config':config})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=1+5*int(tag == 'train'), pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader



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
        if config.get('load_pretrain') is not None:
            sv_file = torch.load(config['load_pretrain'])
            model = models.make(config['model']).cuda()
            model_dict=model.state_dict()
            pretrained_dict = {k: v for k, v in sv_file['model']['sd'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

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


    for batch in tqdm(train_loader, leave=False, desc='train', ncols=0):
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)


        pred = model(batch['inp'] , Nnum / scanning)

        gt = batch['volume'].unsqueeze(1) # / (Nnum*Nnum)

        if gt.shape[-3] < pred.shape[-3]:
            pad_f, pad_b = (pred.shape[-3] - gt.shape[-3]) // 2, (1+pred.shape[-3] - gt.shape[-3]) // 2
            gt = torch.nn.functional.pad(gt, [0,0,0,0,pad_f,pad_b])


        # loss = loss_fn(pred,  gt) # avoid all-zeros output
        loss = loss_fn(pred[gt<threshold], gt[gt<threshold]) + loss_fn(pred[gt>=threshold], gt[gt>=threshold]) if epoch < epoch_threshold else loss_fn(pred, gt)


        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        pred = None; loss = None; predLF = None; psf = None; x3_slf = None; inp = None; inp2 = None; pred2 = None; selected_index = None
        # inp = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, epoch, Nnum, scanning, threshold, epoch_threshold


    config = config_
    # log, writer = utils.set_save_path(save_path)
    # with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
    #     yaml.dump(config, f, sort_keys=False)
    # os.system(f'cp -r ../comparing_methods {save_path}')

    train_loader = make_data_loader(spec=config.get('train_dataset'), tag='train')


    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')
    threshold = config.get('threshold')
    epoch_threshold = config.get('epoch_threshold')

    Nnum = config.get('train_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('train_dataset')['wrapper']['args']['scanning']

    max_val_v = -1e18

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
    parser.add_argument('--config', default='./configs/train-supervised/train_vcdnet_bubtubbead.yaml')
    # parser.add_argument('--config', default='./configs/train-supervised/train_hylfmnet_bubtubbead.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    model_name = config.get('model')['name']
    # save_name = args.name + '_pth'
    save_name = model_name + '_pth'
    # if save_name is None:
    # save_name += '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('../pth', save_name)
    global global_psf, psfshift, input_views, log, global_psf_sumdhw

    log = utils.set_save_path(save_path)


    if config.get('bpmode') is None:
        global_psf = None

    train_loss = main(config, save_path)