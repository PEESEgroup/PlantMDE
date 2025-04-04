import numpy as np
import cv2



def get_samples(img_path, read_rgbmask=False):
    depth_path = img_path.replace('rgb', 'depth')
    seg_path = img_path.replace('rgb', 'segmentation')
    dataset_name = img_path.split('/')[-3]
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
    segmentation = cv2.imread(seg_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
    sample = {}

    if read_rgbmask:
        mask = cv2.imread(img_path.replace('rgb', 'mask'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        sample['rgb_mask'] = mask
    
    sample.update({'image': image, 'depth': depth, 'segmentation': segmentation, 'mask': (depth > 0)})
    sample['depth'][sample['mask'] == 0] = np.nan

    if dataset_name not in ['PlantStereo', 'ETH_BeetRoot']: #Depth info in PlantStereo and ETHs is alredy in disparity space
        sample['disparity'] =  1 / sample['depth']
    else:
        sample['disparity'] = np.copy(sample['depth'])
        sample['depth'] = 1 / sample['depth']

    sample['dataset_name'] = dataset_name
    return sample

def filer_sample_noise(pred, valid_mask, low_percentile=0.05, high_percentile=1):
    outlier_filtered = (pred >= np.nanpercentile(pred[valid_mask], low_percentile * 100)) & ( pred <= np.nanpercentile(pred[valid_mask], high_percentile * 100))
    valid_mask_filter = (valid_mask) & outlier_filtered
    valid_mask_filter = valid_mask_filter ==1
    return valid_mask_filter