import numpy as np
import h5py
import torch
import sys
sys.path.append('../code')
import utils

def get_centerofmass(psf):
    """ calculate psf center of mass.(psf>0 == True)
    input: [C,D,H,W]
    return:[C,D,2]
    """
    C, D, H, W = psf.shape# [-2], psf.shape[-1]
    # psf = psf.view(-1, H, W).unsqueeze(-1)
    psf = psf.reshape(C, D, H, W, 1)
    hwgrid = torch.stack(torch.meshgrid(torch.arange(H) - (H-1)/2,torch.arange(W) - (W-1)/2,indexing='ij'), dim=-1).unsqueeze(0) # 1, H, W, 2
    psf_centerofmass = torch.zeros(C, D, 2)
    for c in range(C):
        psf_centerofmass[c, :,:] = (  psf[c, :,:,:,:] * hwgrid / (psf[c, :,:,:,:].sum((-3,-2),keepdim =True) + 1e-9)  ).sum((-3,-2))

    # return (psf*hwgrid/(psf.sum((-3,-2),keepdim =True) + 1e-9)).sum((-3,-2)).view(C, D, 2)
    return psf_centerofmass

psfName = 'Ideal_PhazeSpacePSF_M63_NA1.4_zmin-10u_zmax10u_zspacing0.2u.mat'
f = h5py.File('./' + psfName, 'r')

global_psf = f.get('psf')
global_psf = np.array(global_psf)

global_psf = torch.tensor(global_psf, dtype=torch.float32)
# torch.save(global_psf,psfName[0:-4]+'.pt')
# global_psf = torch.load(psfName[0:-4]+'.pt')
global_psf = global_psf.permute(2,1,0,4,3).reshape(global_psf.shape[1]*global_psf.shape[2], global_psf.shape[0], global_psf.shape[4], global_psf.shape[3]).contiguous()
input_views = [84, 85, 98, 97, 96, 83, 70, 71, 72, 73, 86, 99, 112, 111, 110, 109, 108, 95, 82, 69, 56, 57, 58, 59, 60, 61, 74, 87, 100, 113, 125, 124, 123, 122, 121, 107, 94, 81, 68, 55, 43, 44, 45, 46, 47, 88, 136, 80, 32]

psfshift = get_centerofmass(global_psf)
print(psfshift.shape)
torch.save(psfshift,psfName[0:-4]+'_psfshift_all.pt')
psfshift = psfshift[input_views]
print(psfshift.shape)
torch.save(psfshift, psfName[0:-4]+'_psfshift_'+'%d'%len(input_views)+'views.pt')