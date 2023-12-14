import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MMFloodDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        self.sar_path = os.path.join(root, 'sar')
        self.dem_path = os.path.join(root, 'dem')
        self.mask_path = os.path.join(root, 'mask')
        self.itemslist = sorted(os.listdir(self.sar_path))

    def __len__(self):
        return len(self.itemslist)
    
    def read_tif(self, path, channels_first=True):
        with rasterio.open(str(path), mode="r", driver="GTiff") as src:
            image = src.read()
        image = image if channels_first else image.transpose(1, 2, 0)
        return image

    def __getitem__(self, idx):
        item = self.itemslist[idx]
        sar = self.read_tif(os.path.join(self.sar_path, item))
        dem = self.read_tif(os.path.join(self.dem_path, item))
        mask = self.read_tif(os.path.join(self.mask_path, item))

        if self.transform:
            sar = self.transform(sar)
            dem = self.transform(dem)
            mask = self.transform(mask)

        return sar, dem, mask
    
class TemporalMMFloodDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        self.itemslist = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.itemslist)
    
    def read_tif(self, path, channels_first=False):
        with rasterio.open(str(path), mode="r", driver="GTiff") as src:
            image = src.read()
        image = image if channels_first else image.transpose(1, 2, 0)
        return image
    
    def read_tifs_in_file(self, path, channels_first=True, return_first=False):
        files = sorted(os.listdir(path))

        if len(files) == 0:
            return None

        if return_first:
            return transforms.ToTensor()(self.read_tif(os.path.join(path, files[0])))
        
        tifs = []
        for f in files:
            tifs.append(transforms.ToTensor()(self.read_tif(os.path.join(path, f))))
        return torch.stack(tifs)

    def __getitem__(self, idx):
        item = self.itemslist[idx]
        s1 = self.read_tifs_in_file(os.path.join(self.root_dir, item, 's1', item+'-0'))
        s2 = self.read_tifs_in_file(os.path.join(self.root_dir, item, 's2', item+'-0'))
        dem = self.read_tifs_in_file(os.path.join(self.root_dir, item, 'DEM'), return_first=True)
        mask = self.read_tifs_in_file(os.path.join(self.root_dir, item, 'mask'), return_first=True)

        if self.transform:
            s1 = self.transform(s1)
            s2 = self.transform(s2)
            dem = self.transform(dem)
            mask = self.transform(mask)

        return s1, s2, dem, mask
    
    def get_sar_by_id(self, id):
        '''
        Retrieve a scene time series by its mmflod id (e.g. 417)
        '''
        s1 = self.read_tifs_in_file(os.path.join(self.root_dir, 's1', id))

        if self.transform:
            s1 = self.transform(s1)


        return s1

