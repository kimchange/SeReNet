import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import *
import tifffile
import re

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self,  root_path, tag = None, split_file=None, split_key=None, first_k=None, last_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        filenames = os.listdir(root_path)
        filenames = [filename for filename in filenames if ( ('.tif') in filename )]

        if split_file is None:
            if tag == 'beads':
                filenames = sorted(filenames,key=lambda x:int(x[5:-8]))
                # print('beads ok')
            elif tag == 'test_No':
                # filenames = sorted(filenames,key=lambda x:int(x[7:-4]))
                filenames = sorted(filenames, key=lambda x: int(re.search(r'\d+', x).group()))
            else:
                filenames = sorted(filenames,key=lambda x:(x[:-4]))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]
        if last_k is not None:
            filenames = filenames[-last_k:]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                # self.files.append(torch.tensor(imread3dtiff(file), dtype=torch.float32))
                # self.files.append(torch.tensor(np.array(imageio.volread(file),dtype=np.float32)))
                self.files.append(torch.tensor(np.array(tifffile.imread(file),dtype=np.float32)))
                # self.files.append(torch.tensor())
                # self.files.append(transforms.ToTensor()(
                #     Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        # if self.cache == 'none':
        #     return transforms.ToTensor()(Image.open(x).convert('RGB'))
        if self.cache == 'none':
            # return torch.tensor(imread3dtiff(x), dtype=torch.float32)
            return torch.tensor(np.array(imageio.volread(x),dtype=np.float32))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
