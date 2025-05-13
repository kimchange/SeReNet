import torch
from torch import nn
import torchvision
from collections import namedtuple

class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_name, model_path, reduction='mean'):
        super(PerceptualLoss, self).__init__()

        vgg16_first1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(inplace=True)
        )

        vgg16_first2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(inplace=True)
        )

        vgg16_first3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(inplace=True),
        )

        vgg19_first12 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(inplace=True)
        )

        if model_name == 'vgg16_1':
            self.model = vgg16_first1
        if model_name == 'vgg16_2':
            self.model = vgg16_first2
        if model_name == 'vgg16_3':
            self.model = vgg16_first3
        if model_name == 'vgg19_12':
            self.model = vgg19_first12

        self.model.load_state_dict(torch.load(model_path))
        self.model.to(torch.device('cuda:0'))
        self.reduction = reduction

    def forward(self, input, target):
        """
        input : pred_lf
        target : ground truth
        """

        # input = normalize_percentile(input, mode=1)
        # target = normalize_percentile(target, mode=1)

        input = torch.permute(input, (1, 0, 2, 3))
        input = input.repeat((1, 3, 1, 1))
        target = torch.permute(target, (1, 0, 2, 3))
        target = target.repeat((1, 3, 1, 1))

        input_pc = self.model(input)
        target_pc = self.model(target)

        l2 = torch.square(input_pc - target_pc)
        if self.reduction == 'sum':
            return torch.sum(l2)
        if self.reduction == 'mean':
            return torch.mean(l2)



# Learned perceptual metric
class LPIPS_vgg16_single_channel(nn.Module):
    def __init__(self, pretrained=True, net='vgg', version='0.1', lpips=True, spatial=False, 
        pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        """ Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(LPIPS_vgg16_single_channel, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'%
                ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()


        if(self.pnet_type in ['vgg','vgg16']):
            # net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        else:
            raise NotImplementedError
        # elif(self.pnet_type=='alex'):
        #     net_type = pn.alexnet
        #     self.chns = [64,192,384,256,256]
        # elif(self.pnet_type=='squeeze'):
        #     net_type = pn.squeezenet
        #     self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns)

        self.net = vgg16_singlechannel(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if(pretrained):
                if(model_path is None):
                    import inspect
                    import os
                    model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'lpips_weights/v%s/%s.pth'%(version,net)))

                if(verbose):
                    print('Loading model from: %s'%model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)          

        if(eval_mode):
            self.eval()

    def forward(self, in0, in1, retPerLayer=False):
        # input shound have value range [0,1]


        # if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
        #     in0 = 2 * in0  - 1
        #     in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val
        
def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()

        # value range [-1,1]
        # self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        # self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

        # value range [0, 1]
        self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None])
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class vgg16_singlechannel(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, lpips=True, use_dropout=False):
        super(vgg16_singlechannel, self).__init__()
        # vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        vgg_pretrained_features = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        # vgg_pretrained_features = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        self.slice1.add_module(str(0), torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1) ))

        self.slice1[0].weight = nn.Parameter( vgg_pretrained_features[0].weight.sum(dim=1,keepdim=True) )
        self.slice1[0].bias = nn.Parameter( vgg_pretrained_features[0].bias )


        for x in range(1, 4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

def extract_vgg16_singlechannel():
    vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

    param_list = []
    for param in vgg16.parameters():
        param_list.append(param)

    vgg16_first1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True)
    )

    for i, layer in enumerate([0]):
        vgg16_first1[layer].weight = param_list[2 * i]
        vgg16_first1[layer].bias = param_list[2 * i + 1]

    torch.save(vgg16_first1.state_dict(), 'vgg16_first1.pth')
    torch.save(vgg16_first1, 'vgg16_first1.pt')


def extract_vgg16_1():
    vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

    param_list = []
    for param in vgg16.parameters():
        param_list.append(param)

    vgg16_first1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True)
    )

    for i, layer in enumerate([0]):
        vgg16_first1[layer].weight = param_list[2 * i]
        vgg16_first1[layer].bias = param_list[2 * i + 1]

    torch.save(vgg16_first1.state_dict(), 'vgg16_first1.pth')
    torch.save(vgg16_first1, 'vgg16_first1.pt')


def extract_vgg16_2():
    vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

    param_list = []
    for param in vgg16.parameters():
        param_list.append(param)

    vgg16_first2 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True)
    )

    for i, layer in enumerate([0, 2]):
        vgg16_first2[layer].weight = param_list[2 * i]
        vgg16_first2[layer].bias = param_list[2 * i + 1]

    torch.save(vgg16_first2.state_dict(), 'vgg16_first2.pth')
    torch.save(vgg16_first2, 'vgg16_first2.pt')


def extract_vgg16_3():
    vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

    param_list = []
    for param in vgg16.parameters():
        #     print(param.shape)
        param_list.append(param)

    vgg16_first3 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
    )

    for i, layer in enumerate([0, 2, 5]):
        vgg16_first3[layer].weight = param_list[2 * i]
        vgg16_first3[layer].bias = param_list[2 * i + 1]

    torch.save(vgg16_first3.state_dict(), 'vgg16_first3.pth')
    torch.save(vgg16_first3, 'vgg16_first3.pt')


def extract_vgg19_12():
    vgg19 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

    param_list = []
    for param in vgg19.parameters():
        param_list.append(param)

    vgg19_first12 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True)
    )

    conv_list = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25]
    for i, layer in enumerate(conv_list):
        vgg19_first12[layer].weight = param_list[2 * i]
        vgg19_first12[layer].bias = param_list[2 * i + 1]

    torch.save(vgg19_first12.state_dict(), 'vgg19_first12.pth')
    torch.save(vgg19_first12, 'vgg19_first12.pt')


if __name__ == '__main__':
    # extract_vgg16_3()
    # extract_vgg19_12()
    extract_vgg16_1()
    # extract_vgg16_2()
