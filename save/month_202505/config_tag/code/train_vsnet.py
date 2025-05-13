import argparse
import os,sys

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
from tifffile import imwrite
import typing
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
import losses
from losses import gan_loss
from losses import basic_loss



def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset,'config':config})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))


    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=int(tag == 'train'), num_workers=1+5*int(tag == 'train'), pin_memory=True, persistent_workers=True)

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


def eval_psnr(loader, model, writer = None, config=None, EPOCH=0, save_path=None, tensorboard_image_writing = True):
    model.eval()

    Nnum = config.get('val_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('val_dataset')['wrapper']['args']['scanning']


    ii_max = config.get('val_dataset')['wrapper']['args']['inp_size'] // scanning
    jj_step = Nnum // scanning
    start_pos = Nnum // 2 - scanning // 2 * jj_step
    scanning_index = [start_pos+ii*13+jj*jj_step for ii in range(ii_max) for jj in range(scanning)]


    metric_fn = utils.calc_psnr


    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val', ncols=0)
    # IDX = 1
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()


        with torch.no_grad():
            # pred = model(batch['inp'], sr_scale_factor)
            pred = model(batch['inp'])


        pred[torch.isnan(pred)] = 0
        pred[torch.isinf(pred)] = 0



        # gt_lf = batch['lf'][:, selected_index, :,:]
        gt_lf = batch['gt']
        if pred.shape != gt_lf.shape:
            pred = torch.nn.functional.interpolate(pred.reshape(-1, 1, pred.shape[-2], pred.shape[-1]), size = gt_lf.shape[-2:], mode = 'bicubic', align_corners = False).reshape(gt_lf.shape)

        B,N,H,W = gt_lf.shape
        gt_maxvalue = torch.quantile(gt_lf[:,N//2,:,:].reshape(B, -1), 0.999, dim=1).reshape(B,1,1,1).abs().cpu() + 1
        inp_lf = (batch['inp'].cpu() / gt_maxvalue).clamp(0,1)
        gt_lf = (gt_lf.cpu() / gt_maxvalue).clamp(0,1)
        pred = (pred.cpu()/ gt_maxvalue).clamp(0,1)
        res = metric_fn(pred, gt_lf)
        val_res.add(res.item(), batch['gt'].shape[0])


        if 'writer' in locals().keys() and tensorboard_image_writing == True:
            tensorboard_image_writing = False

            if(EPOCH % 10 ==0 ) and save_path is not None:
                imwrite(os.path.join(save_path, 'epoch{:03d}.tiff'.format(EPOCH)), np.float32( pred[0,:,:,:].squeeze().cpu()) , imagej=True, metadata={'axes': 'ZYX'}, compression ='zlib')

            writer.add_image('val/inpLF', utils.cmap[ (inp_lf[0,N//2,:,:]*255).long()].cpu() ,dataformats='HWC',global_step=EPOCH)
            writer.add_image('val/pred', utils.cmap[ (pred[0,N//2,:,:]*255).long()].cpu() ,dataformats='HWC',global_step=EPOCH)
            writer.add_image('val/gtLF', utils.cmap[ (gt_lf[0,N//2,:,:]*255).long()].cpu() ,dataformats='HWC',global_step=EPOCH)

    return val_res.item()


def train(train_loader, model, optimizer):
    model.train()

    train_loss = utils.Averager()


    for batch in tqdm(train_loader, leave=False, desc='train',ncols=0):
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)


        pred = model(batch['inp'])
        gt_lf = batch['gt']

        if pred.shape != gt_lf.shape:
            pred = torch.nn.functional.interpolate(pred.reshape(-1, 1, pred.shape[-2], pred.shape[-1]), size = gt_lf.shape[-2:], mode = 'bicubic', align_corners = False).reshape(gt_lf.shape)


        # loss = loss_fn(pred,  gt_lf) # + pred.abs().mean()
        B,N,H,W = gt_lf.shape
        gt_maxvalue = torch.quantile(gt_lf[:,N//2,:,:].reshape(B, -1), 0.999, dim=1).abs().reshape(B,1,1,1) + 1
        gt_lf = gt_lf / gt_maxvalue
        pred = pred / gt_maxvalue

        loss = loss_fns['pixel_loss1'](pred, gt_lf) 
        
        if ('perceptual_loss') in loss_fns.keys():
            loss = loss + loss_fns['perceptual_loss'](pred.reshape(-1,1,pred.shape[-2], pred.shape[-1]) ,  gt_lf.reshape(-1,1,gt_lf.shape[-2], gt_lf.shape[-1]) )


        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        pred = None; loss = None; pred = None; inp = None; inp2 = None; pred2 = None


    return train_loss.item()



def main(config_, save_path):
    global config, log, writer, epoch, Nnum, scanning, shift, selection_weight, num_sample_views, scanning_index, epoch_consistency, loss_fns

    config = config_

    train_loader, val_loader = make_data_loaders()


    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)



    epoch_max = config.get('epoch_max')
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    

    Nnum = config.get('train_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('train_dataset')['wrapper']['args']['scanning']


    ii_max = config.get('train_dataset')['wrapper']['args']['inp_size'] // scanning
    jj_step = Nnum // scanning
    start_pos = Nnum // 2 - scanning // 2 * jj_step
    scanning_index = [start_pos+ii*Nnum+jj*jj_step for ii in range(ii_max) for jj in range(scanning)]


    loss_fns = {}
    for key in config.get('loss_fns'):
        loss_fns[key] = eval(config.get('loss_fns')[key])

    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        # if not (train_loss < 100000):#or torch.isnan(train_loss) or  torch.isnan(train_loss):
        #     print(f'train_loss is {train_loss} so retrain')
        #     break
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)


        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):

            val_res = eval_psnr(val_loader, model,
                writer=writer,
                config=config,
                EPOCH=epoch,
                save_path=save_path)

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

    return train_loss
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train-vsnet/vsnet_Nnum13_datamix.yaml')
    parser.add_argument('--name', default='202505/')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # ti.init(arch=ti.gpu, kernel_profiler=True)
    global log, writer, config#, sr_scale_factor

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    # if save_name is None:
    save_name += '/' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('../../../../save', save_name)
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    os.system(f'cp -r ../code {save_path}')
    log('current workdir is ' + os.getcwd())
    log(f'back up successfully in {save_path}')
    command = 'python ' + ' '.join(sys.argv)
    log('running command: ')
    log(command)


    train_loss = main(config, save_path)