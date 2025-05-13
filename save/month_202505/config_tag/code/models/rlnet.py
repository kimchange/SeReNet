# PyTorch Implementation of RL-Net (https://github.com/MeatyPlus/Richardson-Lucy-Net)
# this network part is just all the same

from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv3d

from models import register

class tensor_div(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a,x1):
        m=torch.nan_to_num(torch.div(a, x1+0.001), nan=0.0, posinf=0.0, neginf=0.0)
        ctx.save_for_backward(m)
        return m

    @staticmethod
    def backward(ctx, grad):
        m, = ctx.saved_tensors
        return grad,torch.mul(-torch.square(m),grad)

class tensor_mul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a,x):
        m=torch.mul(a,x)
        ctx.save_for_backward(a)
        return m

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors # "," is important
        return grad, torch.mul(a, grad)

def pad_to3d(x, stride=[1,1,1]):
    # https://www.coder.work/article/7536422
    # just match size to 2^n
    assert len(x.shape)==5, 'input shoud be 5d'
    d, h, w = x.shape[-3:]
    
    if d % stride[-3] > 0:
        new_d = d + stride[-3] - d % stride[-3]
    else:
        new_d = d

    if h % stride[-2] > 0:
        new_h = h + stride[-2] - h % stride[-2]
    else:
        new_h = h
    if w % stride[-1] > 0:
        new_w = w + stride[-1] - w % stride[-1]
    else:
        new_w = w

    ld, ud = int((new_d-d) / 2), int(new_d-d) - int((new_d-d) / 2)
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh, ld, ud)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad3d(x, pad):
    assert (len(pad) + 1) % 2, 'pad length should be an even number'
    x = x[:,...,pad[4]:x.shape[-3]-pad[5], pad[2]:x.shape[-2]-pad[3], pad[0]:x.shape[-1]-pad[1]  ]
    # if pad[2]+pad[3] > 0:
    #     x = x[:,:,pad[2]:-pad[3],:]
    # if pad[0]+pad[1] > 0:
    #     x = x[:,:,:,pad[0]:-pad[1]]
    return x

@register('rlnet')
class RLNET(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Sequential(*[
            nn.Conv3d(1,4,3,1,1),
            nn.BatchNorm3d(4,),
            nn.Softplus(),
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv3d(4,4,3,1,1),
            nn.BatchNorm3d(4,),
            nn.Softplus(),
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv3d(8,4,3,1,1),
            nn.BatchNorm3d(4,),
            nn.Softplus(),
        ])
        self.t1 = nn.Sequential(*[
            nn.BatchNorm3d(1,),
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv3d(1,8,3,1,1),
            nn.BatchNorm3d(8,),
            nn.Softplus(),
        ])
        self.conv5 = nn.Sequential(*[
            nn.Conv3d(8,8,3,1,1),
            nn.BatchNorm3d(8,),
            nn.Softplus(),
        ])
        self.conv6 = nn.Sequential(*[
            nn.Conv3d(16,8,3,1,1),
            nn.BatchNorm3d(8,),
            nn.Softplus(),
        ])
        self.conv7 = nn.Sequential(*[
            nn.ConvTranspose3d(8,4,2,2),
            nn.BatchNorm3d(4,),
            nn.Softplus(),
        ])
        self.conv8 = nn.Sequential(*[
            nn.Conv3d(4,4,3,1,1),
            nn.BatchNorm3d(4,),
            nn.Softplus(),
        ])
        self.t2 = nn.Sequential(*[
            nn.BatchNorm3d(1,),
            nn.Softplus(),
        ])

        self.conv9 = nn.Sequential(*[
            nn.Conv3d(1,4,3,1,1),
            nn.BatchNorm3d(4,),
            nn.Softplus(),
        ])
        self.conv10 = nn.Sequential(*[
            nn.Conv3d(4,4,3,1,1),
            nn.BatchNorm3d(4,),
            nn.Softplus(),
        ])

        self.t3 = nn.Sequential(*[
            nn.BatchNorm3d(1,),
        ])

        self.conv11 = nn.Sequential(*[
            nn.Conv3d(1,8,3,1,1),
            nn.BatchNorm3d(8,),
            nn.Softplus(),
        ])
        self.conv12 = nn.Sequential(*[
            nn.Conv3d(8,8,3,1,1),
            nn.BatchNorm3d(8,),
        ])
        self.t4 = nn.Sequential(*[
            nn.BatchNorm3d(1,),
            nn.Softplus(),
        ])

        self.conv13 = nn.Sequential(*[
            nn.Conv3d(1,8,3,1,1),
            # nn.BatchNorm3d(8,),
            nn.Softplus(),
        ])
        self.conv14 = nn.Sequential(*[
            nn.Conv3d(10,8,3,1,1),
            # nn.BatchNorm3d(8,),
            nn.Softplus(),
        ])

        self.conv15 = nn.Sequential(*[
            nn.Conv3d(16,8,3,1,1),
            nn.BatchNorm3d(8,),
            nn.Softplus(),
        ])
           

    def forward(self,x):
        x = x.unsqueeze(1)
        x, pads = pad_to3d(x, [2**1, 2**1, 2**1] )
        normed_batch_t_down = F.avg_pool3d(x, kernel_size=[2,2,2], stride=[2,2,2]) # B, 1, D, H, W
        normed_batch_t_down_m=F.softplus(normed_batch_t_down.repeat(1,4,1,1,1), beta=1,) # B, 4, D, H, W
        result_prelu_3 = self.conv1(normed_batch_t_down)
        result_conv_9 = self.conv2(result_prelu_3)
        result_prelu_9 = self.conv3(torch.cat([result_prelu_3, result_conv_9], dim = 1)) + normed_batch_t_down_m

        ave_9=result_prelu_9.mean(dim=1,keepdim=True)
        temp_layer =tensor_div.apply(normed_batch_t_down, ave_9)
        temp_layer=self.t1(temp_layer)

        result_prelu_15 = self.conv4(temp_layer)
        result_prelu_10 = self.conv5(result_prelu_15)

        result_prelu_12 = self.conv6(torch.cat([result_prelu_10, result_prelu_15], dim =1))
        result_conv_12_u = self.conv7(result_prelu_12)

        result_prelu_12_u2 = self.conv8(result_conv_12_u)

        ave_result_prelu_12_u2 = result_prelu_12_u2.mean(dim=1,keepdim=True)
        temp2=tensor_mul.apply(x,ave_result_prelu_12_u2)

        result_prelu_12_1 = self.t2(temp2)

        # update
        normed_batch_m=F.softplus(x.repeat(1,4,1,1,1), beta=1,)
        result_conv_1_b = self.conv9(x)
        result_prelu_1 = self.conv10(result_conv_1_b) + normed_batch_m
        ave_1 = result_prelu_1.mean(dim=1,keepdim=True)
        EST=tensor_div.apply(x, ave_1)
        EST = self.t3(EST)
        result_conv_2_b = self.conv11(EST)
        normed_batch_2 = self.conv12(result_conv_2_b)
        act_2 = F.softplus(normed_batch_2) + torch.ones_like(normed_batch_2)
        ave_2 = act_2.mean(dim=1,keepdim=True)
        Estimation1=tensor_mul.apply(result_prelu_12_1,ave_2)
        Estimation=F.softplus(self.t4(Estimation1))
        result_prelu_2 =Estimation
        result_prelu_2_1=Estimation
        # Estimation_tile=Estimation.repeat(1,8,1,1,1)
        result_conv_2_fined = self.conv13(result_prelu_2_1)

        result_conv_2_fine1_c=torch.cat([result_conv_2_fined,result_prelu_2_1,result_prelu_12_1], dim =1)
        act_2_fine = self.conv14(result_conv_2_fine1_c)

        Merge=torch.cat([result_conv_2_fined,act_2_fine],1)
        result_prelu_13 = self.conv15(Merge)
        self.prediction=result_prelu_13.mean(dim=1,keepdim=True)
        self.prediction_log=self.prediction
        self.e=temp2
        # self.e2=Estimation1
        self.first=result_prelu_13.mean(dim=1,keepdim=True)

        self.e = unpad3d(self.e,pads)
        self.prediction_log = unpad3d(self.prediction_log,pads)

        return torch.cat([self.e, self.prediction_log],dim=1)

