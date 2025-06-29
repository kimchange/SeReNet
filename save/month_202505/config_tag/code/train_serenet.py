# SeReNet train script
# The Code is created based on the method described in the following paper
#       Zhi Lu#, Manchang Jin#, et al.,
#       Physics-driven self-supervised learning for fast high-resolution robust 3D reconstruction of light-field microscopy
#       Nature, Methods 2025
# Contact: ZHI LU (luzhi@tsinghua.edu.cn)
# Date: 7/7/2024
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
import torch.nn.functional as F
import numpy as np
import tifffile

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset,'psfshift':psfshift,'config':config})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=1+5*int(tag == 'train'), pin_memory=True, persistent_workers=True) # if using windows, set "num_workers=1+5*int(tag == 'train'), persistent_workers=False", or you may get some unexpected bugs
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


def eval_tensorboard(loader, model, psfshift = None, global_psf = None, 
              verbose=False, writer = None, config=None, EPOCH=0, save_path=None, tensorboard_image_writing = True,IDX=0):
    model.eval()
    num_sample_views = config.get('val_dataset')['wrapper']['args']['sample_views']
    selection_weight = torch.ones(1,len(input_views)) *1.0
    # selection_weight[0,0] = 0
    if config.get('val_dataset')['wrapper']['args']['sample_centerview'] is not None:
        selection_weight[0,0] = 0

 
    metric_fn = utils.calc_psnr


    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val', ncols=0)
    # IDX = 1
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        with torch.no_grad():
            # pred = model(batch['inp'], shift, Nnum / scanning, global_psf_sumhw)
            pred = model(batch['inp'], Nnum / scanning )

        pred[torch.isnan(pred)] = 0
        pred[torch.isinf(pred)] = 0

        if config.get('val_dataset')['wrapper']['args']['sample_centerview'] is not None:
            selected_index = torch.cat([torch.tensor([0]),torch.multinomial(selection_weight.view(-1), num_sample_views - 1, replacement=False, out=None)],dim=0)
        else:
            selected_index = torch.multinomial(selection_weight.view(-1), num_sample_views, replacement=False, out=None)

        input_lf = batch['lf'][:, selected_index, :,:]

        if config.get('bpmode') == 'selectiviews':
            # forward projection
            psf = global_psf[selected_index,:,:,:].unsqueeze(0).cuda()
            predLF = torch.zeros(psf.shape[0:2]+pred.shape[3:], device='cuda:0')
            for blk in range(psf.shape[-3]):
                predLF = predLF + utils.imagingLFM(pred[:,:,blk*(1):(blk+1)*(1),:,:], \
                    psf[:,:,blk*(1):(blk+1)*(1),:,:])
        else:
            raise NotImplementedError

        if predLF.shape != input_lf.shape:
            predLF = torch.nn.functional.interpolate(predLF, size = input_lf.shape[2:], mode='bicubic', align_corners = False)


        input_lf = utils.normalize_percentile(input_lf.cpu(), low=0., high=100).clamp(0,1)
        predLF = utils.normalize_percentile(predLF.cpu(), low=5., high=100).clamp(0,1)
        res = metric_fn(predLF, input_lf)
        val_res.add(res.item(), batch['inp'].shape[0])

        if 'volume' in batch.keys():
            synthetic_volume = batch['volume'][0:1,5:-5, 30:-30, 30:-30]
            synthetic_volume = utils.normalize_percentile(synthetic_volume.cpu(), low=10., high=100).clamp(0,1)


        if 'writer' in locals().keys() and tensorboard_image_writing == True:
            tensorboard_image_writing = False
            if(EPOCH % 10 ==0 ) and save_path is not None:
                tifffile.imwrite(os.path.join(save_path, 'epoch{:03d}.tiff'.format(EPOCH)), np.float32( pred[0,0,:,:,:].squeeze().cpu()) , imagej=True, metadata={'axes': 'ZYX'}, compression ='zlib')

            pred = pred[0:1,0:1,5:-5, 30:-30, 30:-30]
            pred = utils.normalize_percentile(pred.cpu(), low=10., high=100).clamp(0,1)
            predxyxzyz = torch.cat( [torch.cat([pred[0,0,:,:,:].max(0).values,pred[0,0,:,:,:].max(2).values.permute(1,0)],dim=1),  F.pad( pred[0,0,:,:,:].max(1).values, (0, pred.shape[2]))], dim=0).cpu()

            writer.add_image('val_pred', utils.cmap[ (   predxyxzyz         *255).long()].cpu() ,   dataformats='HWC',global_step=EPOCH)
            writer.add_image('val_predLF', utils.cmap[ (predLF[0,0,:,:]*255).long()].cpu() ,dataformats='HWC',global_step=EPOCH)
            writer.add_image('val_inputLF', utils.cmap[ (input_lf[0,0,:,:]*255).long()].cpu() ,dataformats='HWC',global_step=EPOCH)

            if 'synthetic_volume' in locals().keys():
                synthetic_volume_xyxzyz = torch.cat( [torch.cat([synthetic_volume[0,:,:,:].max(0).values,synthetic_volume[0,:,:,:].max(2).values.permute(1,0)],dim=1),  F.pad( synthetic_volume[0,:,:,:].max(1).values, (0, synthetic_volume.shape[-3]))], dim=0).cpu()
                writer.add_image('val_syntheticVolume', utils.cmap[ (   synthetic_volume_xyxzyz         *255).long()].cpu() ,   dataformats='HWC',global_step=EPOCH)

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()

def train(train_loader, model, optimizer):
    model.train()
    
    train_loss = utils.Averager()


    for batch in tqdm(train_loader, leave=False, desc='train', ncols=0):
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        rand_factor = 1. #torch.rand(1).item()
        rand_bg = torch.rand(1).item() * 0.
        batch['inp'] = batch['inp']*rand_factor + rand_bg * global_psf_sumdhw.reshape(1, -1, 1, 1, 1)
        batch['lf'] = batch['lf']*rand_factor + rand_bg * global_psf_sumdhw.reshape(1, -1, 1, 1)
        
        pred, pred_woupsample = model(batch['inp'], Nnum / scanning)

        if config.get('train_dataset')['wrapper']['args']['sample_centerview'] is not None:
            selected_index = torch.cat([torch.tensor([0]),torch.multinomial(selection_weight.view(-1), num_sample_views - 1, replacement=False, out=None)],dim=0)
        else:
            selected_index = torch.multinomial(selection_weight.view(-1), num_sample_views, replacement=False, out=None)

        if config.get('bpmode') == 'selectiviews':
            # forward projection
            psf = global_psf[selected_index,:,:,:].unsqueeze(0).cuda()
            predLF = torch.zeros(psf.shape[0:2]+pred.shape[3:], device='cuda:0')
            for blk in range(psf.shape[-3]):
                predLF = predLF + utils.imagingLFM(pred[:,:,blk*(1):(blk+1)*(1),:,:], \
                    psf[:,:,blk*(1):(blk+1)*(1),:,:])
        else:
            pass


        input_lf = batch['lf'][:, selected_index, :,:]
        if predLF.shape != input_lf.shape:
            # predLF = torch.nn.functional.interpolate(predLF, size = input_lf.shape[2:], mode = 'bicubic', align_corners = True)
            predLF = torch.nn.functional.interpolate(predLF, size = input_lf.shape[2:], mode = 'bilinear', align_corners = False)
            # input_lf = torch.nn.functional.interpolate(input_lf, size = predLF.shape[2:], mode = 'bilinear', align_corners = False)

        B,A,H,W = input_lf.shape
        input_lf_maxval = torch.quantile(input_lf.reshape(B,A,-1) , 0.999, dim=2).reshape(B,A,1,1).abs() + 1 # if maxval is 0.01, the division will still be stable 
        loss = loss_fns['pixel_loss1'](predLF,  input_lf) / input_lf_maxval.mean()

        # predLF, input_lf = predLF / input_lf_maxval,  input_lf / input_lf_maxval
        if ('perceptual_loss') in loss_fns.keys():
            loss = loss + loss_fns['perceptual_loss'](predLF / input_lf_maxval,  input_lf / input_lf_maxval)

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
    global config,  epoch, Nnum, scanning, shift, selection_weight, num_sample_views, loss_fns


    config = config_

    train_loader, val_loader = make_data_loaders()


    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    # loss_fn = nn.L1Loss(reduction='mean') if config.get('loss_fn') is None else eval(config.get('loss_fn'))
    loss_fns = {}
    for key in config.get('loss_fns'):
        loss_fns[key] = eval(config.get('loss_fns')[key])


    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    

    Nnum = config.get('train_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('train_dataset')['wrapper']['args']['scanning']

    shift = (psfshift / Nnum * scanning).cuda()

    num_sample_views = config.get('train_dataset')['wrapper']['args']['sample_views']
    selection_weight = torch.ones(1,len(input_views)) *1.0


    if config.get('train_dataset')['wrapper']['args']['sample_centerview'] is not None:
        selection_weight[0,0] = 0


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
                psfshift = psfshift,
                global_psf = global_psf,
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
    parser.add_argument('--config', default='./configs/train-serenet/serenet_Nnum13_bubtub.yaml')
    parser.add_argument('--name', default='202505/')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    # if save_name is None:
    save_name += '/' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('../../../../save', save_name)
    global global_psf, psfshift, input_views, log, writer, global_psf_sumdhw, global_psf_sumhw
    input_views = torch.tensor(config.get('input_views'))
    log, writer = utils.set_save_path(save_path)

    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    os.system(f'cp -r ../code {save_path}')
    log('current workdir is ' + os.getcwd())
    log(f'back up successfully in {save_path}')
    command = 'python ' + ' '.join(sys.argv)
    log('running command: ')
    log(command)

    
    if config.get('debug_mode') is not None:
        global_psf = torch.ones(len(input_views), 101, 195, 195)
    else:

        if config.get('psf').endswith('.mat'):
            f = h5py.File(config.get('psf'),'r')
            global_psf = f.get('psf')
            global_psf = np.array(global_psf)
            global_psf = torch.tensor(global_psf, dtype=torch.float32) # / 10
        elif config.get('psf').endswith('.pt'):
            global_psf = torch.load(config.get('psf'))
        else:
            raise NotImplementedError
        

        if len(global_psf.shape) == 5: # H,W,U,V,D in matlab, DVUWH in python, to   A, D, H, W
            global_psf = global_psf.permute(2,1,0,4,3).reshape(global_psf.shape[1]*global_psf.shape[2], global_psf.shape[0], global_psf.shape[4], global_psf.shape[3]).contiguous()
            global_psf = global_psf[input_views,:,:,:]
        elif len(global_psf.shape) == 4:
            # input views has pre selected
            global_psf = global_psf.permute(0,1,3,2) # H,W,D,C in matlab --> C,D,W,H in pytorch --> permute to C,D,H,W
        else:
            raise NotImplementedError

        global_psf[global_psf<0] = 0

    log(f'psf loaded, psf.shape = {global_psf.shape}')

    if config.get('psfinfo') is not None:
        psfinfo = torch.load(config.get('psfinfo')) 
        psfshift = psfinfo['psfshift']
        # psfshift = psfinfo['psfshift_linearfit'][input_views]
        # global_psf_sumhw = psfinfo['psfsum_gaussianfit'][input_views]
        # global_psf_sumhw = global_psf_sumhw.cuda().unsqueeze(0)
        log(f'using pre-computed psfinfo '+config.get('psfinfo'))
    else:
        psfshift = utils.get_centerofmass(global_psf)
        log(f'using psfshift center-of-mass of '+config.get('psf'))


    global_psf_sumdhw = global_psf.sum((1,2,3), keepdim=False).cuda()

    train_loss = main(config, save_path)