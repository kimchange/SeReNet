# PyTorch Implementation of RL-Net (https://github.com/MeatyPlus/Richardson-Lucy-Net)
import argparse
import os, sys

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
# import taichi as ti



# used when training process, to show some intermediate results
def eval_tensorboard(loader, model, psfshift = None, global_psf = None, 
              verbose=False, writer = None, config=None, EPOCH=0, save_path=None, tensorboard_image_writing = True,IDX=0):
    model.eval()
    Nnum = config.get('val_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('val_dataset')['wrapper']['args']['scanning']
    input_views = torch.tensor(config.get('input_views'))

    metric_fn = utils.calc_psnr


    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val', ncols=0)
    # IDX = 1
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        with torch.no_grad():
            pred = model(batch['inp'])

        pred[torch.isnan(pred)] = 0
        pred[torch.isinf(pred)] = 0

        pred = pred[:,1:2,:,:,:]

        gt = batch['volume']

        if gt.shape[-3] < pred.shape[-3]:
            pad_f, pad_b = (pred.shape[-3] - gt.shape[-3]) // 2, (1+pred.shape[-3] - gt.shape[-3]) // 2
            gt = torch.nn.functional.pad(gt, [0,0,0,0,pad_f,pad_b])

        res = metric_fn(pred, gt.unsqueeze(1))

        
        val_res.add(res.item(), batch['inp'].shape[0])
        input_lf = batch['inp'].clamp(0,1)
        if 'volume' in batch.keys():
            synthetic_volume = gt[0:1,5:-5, 30:-30, 30:-30]
            synthetic_volume = utils.normalize_percentile(synthetic_volume.cpu(), low=10., high=100).clamp(0,1)


        if 'writer' in locals().keys() and tensorboard_image_writing == True:
            tensorboard_image_writing = False
            if(EPOCH % 20 ==0 ) and save_path is not None:
                # pred = pred[0,0,:,:,:] * 65535
                imwrite(os.path.join(save_path, 'epoch{:03d}.tiff'.format(EPOCH)), np.float32( pred[0,0,:,:,:].squeeze().cpu()) , imagej=True, metadata={'axes': 'ZYX'}, compression ='zlib')

            pred = pred[0:1,0:1,5:-5, 30:-30, 30:-30]
            pred = utils.normalize_percentile(pred.cpu(), low=10., high=100).clamp(0,1)
            predxyxzyz = torch.cat( [torch.cat([pred[0,0,:,:,:].max(0).values,pred[0,0,:,:,:].max(2).values.permute(1,0)],dim=1),  F.pad( pred[0,0,:,:,:].max(1).values, (0, pred.shape[2]))], dim=0).cpu()

            writer.add_image('val_pred', utils.cmap[ (   predxyxzyz         *255).long()].cpu() ,   dataformats='HWC',global_step=EPOCH)
            # writer.add_image('val_inputLF', utils.cmap[ (input_lf[0,0,:,:]*255).long()].cpu() ,dataformats='HWC',global_step=EPOCH)

            if 'synthetic_volume' in locals().keys():
                synthetic_volume_xyxzyz = torch.cat( [torch.cat([synthetic_volume[0,:,:,:].max(0).values,synthetic_volume[0,:,:,:].max(2).values.permute(1,0)],dim=1),  F.pad( synthetic_volume[0,:,:,:].max(1).values, (0, synthetic_volume.shape[-3]))], dim=0).cpu()
                writer.add_image('val_syntheticVolume', utils.cmap[ (   synthetic_volume_xyxzyz         *255).long()].cpu() ,   dataformats='HWC',global_step=EPOCH)

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()

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


def get_SSIM(dl_op, gt_label,max_val=1):
    mean_prediction =torch.mean(dl_op)
    mean_gt =torch.mean(gt_label)
    sigma_prediction=torch.mean(torch.square(torch.subtract(dl_op,mean_prediction)))
    sigma_gt=torch.mean(torch.square(torch.subtract(gt_label,mean_gt)))
    sigma_cross=torch.mean(torch.multiply(torch.subtract(dl_op,mean_prediction),
                                                    torch.subtract(gt_label,mean_gt)))
    SSIM_1=2*torch.multiply(mean_prediction,mean_gt)+1e-4*max_val*max_val
    SSIM_2=2*sigma_cross+9e-4**max_val*max_val
    SSIM_3=torch.square(mean_prediction)+torch.square(mean_gt)+1e-4**max_val*max_val
    SSIM_4=sigma_prediction+sigma_gt+9e-4**max_val*max_val
    SSIM=torch.div(torch.multiply(SSIM_1,SSIM_2),torch.multiply(SSIM_3,SSIM_4))
    return SSIM


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss(reduction='mean') if config.get('loss_fn') is None else eval(config.get('loss_fn'))
    train_loss = utils.Averager()


    for batch in tqdm(train_loader, leave=False, desc='train', ncols=0):
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        pred = model(batch['inp'])


        gt = batch['volume']

        if gt.shape[-3] < pred.shape[-3]:
            pad_f, pad_b = (pred.shape[-3] - gt.shape[-3]) // 2, (1+pred.shape[-3] - gt.shape[-3]) // 2
            gt = torch.nn.functional.pad(gt, [0,0,0,0,pad_f,pad_b])

        loss = 0.1*loss_fn(pred[:,0,:,:,:], 0.2*batch['inp'] + 0.8*gt) + 1*loss_fn(pred[:,1,:,:,:], gt) - 1.0*torch.log((1+get_SSIM(pred[:,1,:,:,:],gt))/2)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        pred = None; loss = None
        # inp = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer, epoch, Nnum, scanning


    config = config_
    # log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    os.system(f'cp -r ../code {save_path}')

    train_loader, val_loader = make_data_loaders()


    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')


    Nnum = config.get('train_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('train_dataset')['wrapper']['args']['scanning']

    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)

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

            val_res = eval_tensorboard(val_loader, model,
                writer=writer,
                config=config,
                EPOCH=epoch,
                save_path=save_path)

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)

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
    parser.add_argument('--config', default='./configs/train-rlnet/rlnet_bubtub.yaml')
    parser.add_argument('--name', default='202505/')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # ti.init(arch=ti.gpu, kernel_profiler=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    # if save_name is None:
    save_name += '/' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('../../../../save', save_name)
    global global_psf, psfshift, input_views, log, writer
    log, writer = utils.set_save_path(save_path)

    os.system(f'cp -r ../code {save_path}')
    log('current workdir is ' + os.getcwd())
    log(f'back up successfully in {save_path}')
    command = 'python ' + ' '.join(sys.argv)
    log('running command: ')
    log(command)

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

    if config.get('bpmode') is None:
        global_psf = None

    train_loss = main(config, save_path)
