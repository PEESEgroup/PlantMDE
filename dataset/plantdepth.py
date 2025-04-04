import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
import numpy as np
import os

class PLANTDEPTH_MIXED(Dataset):
    def __init__(self, root_dir, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        self.filelist = []
        self.root_dir = root_dir
        for path in filelist_path:
            with open(self.root_dir + path, 'r') as f:
                if mode == 'test':
                    self.filelist.extend(f.read().splitlines()[:100])  # Limiting the number for testing
                else:
                    self.filelist.extend(f.read().splitlines())
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    
    def __getitem__(self, item):
        img_path = self.root_dir + self.filelist[item].split(' ')[0]
        depth_path = self.root_dir + self.filelist[item].split(' ')[1]
        seg_path = self.root_dir + self.filelist[item].split(' ')[2]
        dataset_name = img_path.split('/')[-3]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
        segmentation = cv2.imread(seg_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
        
        if dataset_name in ['PlantStereo', 'ETH_BeetRoot', 'WUR_DwarfTomato']:
            rgbmask = cv2.imread(self.root_dir + self.filelist[item].split(' ')[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            image[rgbmask == 0] = [70/255,70/255,70/255]

        sample = self.transform({'image': image, 'depth': depth, 'segmentation': segmentation, 'mask': (depth > 0)})                
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['segmentation'] = torch.from_numpy(sample['segmentation'])
        sample['mask'] = torch.from_numpy(sample['mask'])

        if dataset_name not in ['PlantStereo', 'ETH_BeetRoot']: #Depth info in PlantStereo and ETHs is alredy in disparity space
            sample['disparity'] = torch.where(sample['depth'] > 0, 1 / sample['depth'], torch.zeros_like(sample['depth']))
        else:
            sample['disparity'] = torch.clone(sample['depth'])
            sample['depth'] = torch.where(sample['depth'] > 0, 1 / sample['depth'], torch.zeros_like(sample['depth']))
            
        sample['dataset_name'] = dataset_name
        sample['img_path'] = img_path
        return sample

    def __len__(self):
        return len(self.filelist)
