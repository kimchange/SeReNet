import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
import math
from models import register


"""
Script of Vs-Net model.
Virtual-scanning light-field microscopy for robust snapshot high-resolution volumetric imaging. Nat Methods 20, 735–746 (2023).
Lu, Z., Liu, Y., Jin, M. et al. 

"""

@register('vsnet')
class VsNet(nn.Module):
    def __init__(self, angRes, K=4, n_block=4, channels=64, upscale_factor=3):
        super(VsNet, self).__init__()
        """ 
        Parameters:
            angRes (int)       -- the number of angular views in input LF images
            K (int)            -- the number of cascades of the interaction module
            n_block (int)      -- the number of interaction blocks in the interaction module
            channels (int)     -- the number of channels in the network
            upscale_factor(int)-- the upscale factor in the network     
        """


        self.angRes = angRes
        self.nf = channels
        self.upscale_factor = upscale_factor

        # Feature Extraction  AM_FE:the Angular-mixed feature | LF_FE:the light-field feature | SA-FE: the spatial-angular feature
        self.AM_FE = nn.Conv2d(1, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False)
        self.LF_FE = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                               bias=False)
        self.SA_FE = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Feature interaction and fusion
        self.cas_interaction = Cascaded_InterGroups(angRes, K, n_block, channels)
        self.fusion = Fusion(angRes, K, channels)

        # Upsampling
        self.upsampling = Upsampling(angRes, channels, upscale_factor)


    def forward(self, x):

        # Here the input x is the spatial-angular image with the size of [B,u*v,h,w] ([Batchsize, angle number, height, width])
        # u, v, B, H, W = self.angRes, self.angRes, x_lf.shape[0], x_lf.shape[2], x_lf.shape[3]

        # am: the Angular-mixed feature
        # lf: the light-field feature
        # sa: the spatial-angular feature
        input_sa = x.view(-1,1,x.shape[-2],x.shape[-1])   # Size of [B*169,1,h,w]

        # Realignment from spatial-angular domain to light-field domain
        input_lf = SA_to_LF_input(x, self.angRes)         # Size of [B,1,h*u,w*v]

        # Feature extraction to obatin x_am, x_lf, x_sa feature
        fea_am = self.AM_FE(input_lf)                     # Size of [B,nf,h,w]
        fea_lf = self.LF_FE(input_lf)                     # Size of [B,nf,h*u,w*v]
        fea_sa = self.SA_FE(input_sa)                     # Size of [B*u*v,nf,h,w]
        
        # Interaction and Fusion
        cas_inter_am, cas_inter_lf, cas_inter_sa = self.cas_interaction(fea_am, fea_lf, fea_sa)
        fusion = self.fusion(cas_inter_am, cas_inter_lf, cas_inter_sa)
        
        # Upsampling 
        out_sr = self.upsampling(fusion)

        # Global residual connection
        x_sr = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic',
                                    align_corners=False)  # Size of [B, angRes*angRes,H*3, W*3]
        out = out_sr + x_sr

        return out.unsqueeze(1)



class InterBlock(nn.Module):
    '''
    Including 3 modelues:
    1. angular-mixed feature interaction
    2. spatial-angular feature interaction
    3. light-field feature interaction
    '''
    def __init__(self, angRes, channels):
        super(InterBlock, self).__init__()

        self.angRes = angRes
        self.nf = channels
        
        self.sa_Conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.sa_Conv2 = nn.Conv2d(2 * channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.am_Conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.am_Conv2 = nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
            
        self.lf_Conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                 padding=int(angRes), bias=False)
        self.lf_Conv2 = nn.Conv2d(3 * channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                  padding=int(angRes), bias=False)

        # Convolution to transform the feature from the light-field feature into the angular-mixed feature
        self.lf_am_conv = nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), padding=0,
                                    bias=False)
        # Convolution to transform the feature from the angular-mixed feature into light-field feature
        self.am_lf_conv = nn.Sequential(
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x_am, x_lf, x_sa):

        # Angular-mixed feature interaction
        inter_am1 = self.am_Conv(x_am)
        inter_am2 = self.lrelu(self.lf_am_conv(x_lf))
        inter_am = torch.cat((inter_am1, inter_am2), 1)
        out_am = self.lrelu(self.am_Conv2(inter_am)) + x_am
        
        # Light-field feature interaction
        inter_lf1 = self.lf_Conv(x_lf)
        inter_lf2 = self.am_lf_conv(x_am)
        inter_lf3 = SA_to_LF(x_sa, self.angRes)      # Realignment from spatial-angular domain to light-field domain
        inter_lf3 = self.lf_Conv(inter_lf3)
        inter_lf = torch.cat((inter_lf1, inter_lf2, inter_lf3), 1)
        out_lf = self.lrelu(self.lf_Conv2(inter_lf)) + x_lf

        # Spatial-angular feature interaction
        inter_sa1 = self.sa_Conv(x_sa)
        inter_sa2 = LF_to_SA(x_lf, self.angRes)  # Realignment from light-field domain to spatial-angular domain
        inter_sa2 = self.sa_Conv(inter_sa2)
        inter_sa = torch.cat((inter_sa1, inter_sa2), 1)
        out_sa = self.lrelu(self.sa_Conv2(inter_sa)) + x_sa
        return out_am, out_lf, out_sa


class InterGroup(nn.Module):
    def __init__(self, angRes, n_block, channels):
        # Use n interaction block in one Interaction group
        super(InterGroup, self).__init__()
        modules = []
        self.n_block = n_block
        for i in range(n_block):
            modules.append(InterBlock(angRes, channels))
        self.chained_blocks = nn.Sequential(*modules)

    def forward(self, x_am, x_lf, x_sa):
        inter_am = x_am
        inter_lf = x_lf
        inter_sa = x_sa
        for i in range(self.n_block):
           inter_am, inter_lf, inter_sa = self.chained_blocks[i](inter_am, inter_lf, inter_sa)
        out_am = inter_am
        out_lf = inter_lf
        out_sa = inter_sa
        return out_am, out_lf, out_sa


class Cascaded_InterGroups(nn.Module):
    def __init__(self, angRes, K, n_block, channels):
        '''
        Perform K cascades of the interaction modules, and return the K concatenated cascaded interacted features
        Input:  the interaction features (inter_am, inter_lf, inter_sa) with the channel number of C
        Output: the K cascaded features (cas_am, cas_lf, cas_sa) with the channel number of K*C
        Parameters:
            K:       the number of cascades
            n_block: the number of interaction blocks in each interation module
        '''

        super(Cascaded_InterGroups, self).__init__()
        self.K = K
        body = []
        for i in range(K):
            body.append(InterGroup(angRes, n_block, channels))
        self.body = nn.Sequential(*body)

    def forward(self,inter_am, inter_lf, inter_sa):
        cas_am = []
        cas_lf = []
        cas_sa = []
        for i in range(self.K):
            inter_am,inter_lf, inter_sa = self.body[i](inter_am, inter_lf, inter_sa)
            cas_am.append(inter_am)
            cas_lf.append(inter_lf)
            cas_sa.append(inter_sa)
        return torch.cat(cas_am, 1), torch.cat(cas_lf, 1), torch.cat(cas_sa,1)


class Fusion(nn.Module):
    def __init__(self, angRes, K, channels):
        super(Fusion, self).__init__()
        '''
        Fuse the K concatenated cascaded interacted features in 3 domains
        Input: the K cascaded features (cas_am, cas_lf, cas_sa) with the channel number of K*C
        Output: the fused features of 3-domain features
        Parameters: 
           K: the number of cascades
       '''

        self.angRes = angRes
        self.sa_fusion = nn.Sequential(
            nn.Conv2d(K * channels, channels, kernel_size=3, stride=1,padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.am_fusion = nn.Sequential(
            nn.Conv2d(K * channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.final_fusion = nn.Sequential(
            nn.Conv2d((K+2)*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),padding=int(angRes), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, cas_am, cas_lf, cas_sa):       
        fu_am = self.am_fusion(cas_am)
        fu_sa = SA_to_LF(self.sa_fusion(cas_sa), self.angRes)
        out = torch.cat((fu_am, cas_lf, fu_sa), 1)
        out = self.final_fusion(out)
        return out


class Upsampling(nn.Module):
    # Upsampling module to imrpve image resolution accroding to upscale_factor

    def __init__(self, angRes, channels, upscale_factor):
        super(Upsampling, self).__init__()       
        self.PreConv = nn.Conv2d(channels, channels * upscale_factor ** 2, kernel_size=3, stride=1,
                                 dilation=int(angRes), padding=int(angRes), bias=False)
        self.PixelShuffle = nn.PixelShuffle(upscale_factor)
        self.FinalConv = nn.Conv2d(int(channels), 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.angRes = angRes

    def forward(self, x):
        out_LR = self.PreConv(x)
        out_sav_LR = LF_to_SAM(out_LR, self.angRes) 
        out_sav_SR = self.PixelShuffle(out_sav_LR)
        out_SR = self.FinalConv(out_sav_SR)
        out_SR = SAM_to_SA(out_SR, self.angRes) 
        return out_SR


def SA_to_LF_input(img, angRes=13):
    # Realignment from spatial-angular domain to light-field domain (size of [B, 169, h, w] --> [B, 1, h*13, w*13])
    u, v, h, w = int(angRes), int(angRes), img.shape[-2] , img.shape[-1]
    if img.ndimension() == 4:
        img = img.view(img.shape[0],u, v, h, w) \
            .permute(0, 3,1,4,2) \
            .contiguous() \
            .view(img.shape[0], 1, h * u, w * v)
    return img


def LF_to_SA(img,angRes=13):
    # realignment the light field feature to spatial-angular feature (size of [B,nf, h*13, w*13] --> [B*169, nf,h,w])
    u,v,h,w = int(angRes), int(angRes), img.shape[2]//int(angRes), img.shape[3]//int(angRes)
    if img.ndimension() == 4:
        img = img.view(img.shape[0],img.shape[1], h,u,w,v) \
              .permute(0,3,5,1,2,4) \
              .contiguous() \
              .view(-1,img.shape[1], h, w)
    return img

def SA_to_LF(x,angRes):
    # Realignment the spatial-angular domain to light-field domain (Size of [B*169, nf, h, w] --> [B,nf, h*13, w*13])
    u, v, nf, h, w = angRes, angRes, x.shape[1], x.shape[2], x.shape[3]
    x = x.contiguous().view(-1, u * v, nf, h, w)

    out = x.view(-1, u, v, nf, h, w) \
        .permute(0, 3, 4, 1, 5, 2) \
        .contiguous() \
        .view(-1, nf, u * h, v * w)
    return out

def LF_to_SAM(img,angRes=13):
    # Realignment from light-field domain to spatial-angular montage (size of [B, nf, h*13, w*13] --> [B, nf, 13*h, 13*w])
    u,v,h,w = int(angRes), int(angRes), img.shape[2]//int(angRes), img.shape[3]//int(angRes)
    if img.ndimension() == 4:
        img = img.view(img.shape[0], img.shape[1], h,u,w,v) \
              .permute(0, 1, 3, 2, 5, 4) \
              .contiguous() \
              .view(img.shape[0], img.shape[1], u*h, v*w)
    return img

def SAM_to_SA(img,angRes=13):
    # Realignment from spatial-angular montage to spatial-angualr domain (Size of [B,1, 13*h, 13*w] --> [B, 169,h, w])
    u, v, h, w = int(angRes), int(angRes), img.shape[2] // int(angRes), img.shape[3] // int(angRes)
    if img.ndimension() == 4:
        img = img.view(img.shape[0],  u, h, v, w) \
            .permute(0, 1, 3, 2, 4) \
            .contiguous() \
            .view(img.shape[0], u * v, h, w)
    return img


def SA_to_SAM(img,angRes=13):
    # Realignment from spatial-angular domain to spatial-angular montage (Size of [B, 169,h, w] --> [B,1, 13*h, 13*w])
    u, v,  h, w = angRes, angRes, img.shape[2], img.shape[3]
    img = img.view(-1, u, v, h, w) \
        .permute(0, 1, 3, 2, 4) \
        .contiguous() \
        .view(-1, 1, u * h, v * w)
    return img

