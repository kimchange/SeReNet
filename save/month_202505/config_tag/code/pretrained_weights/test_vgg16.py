import torch
import torchvision
import lpips


loss_fn_vgg = lpips.LPIPS(pretrained=True, net='vgg', version='0.1', lpips=True)
