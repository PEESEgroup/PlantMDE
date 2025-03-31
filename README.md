# PlantMDE
The repo for ``3D Plant Phenotyping from a Single Image: Generalizable Plant Monocular Depth Estimation via Dataset Mixing''.

*We first release the PlantDepth benchmark dataset. The code for PlantMDE (Monocular Depth Estimation) is coming soon.*

---

## 🌱 PlantDepth Benchmark Dataset 
The first benchmark dataset designed for **plant depth estimation** and **3D reconstruction tasks**. It renders RGB-D plant images from the mainstream publicly available plant 3D datasets. The dataset can be downloaded by the [link](https://drive.google.com/file/d/1XbDwjUn16dVl7F6uGNvpgtFU-uLm1jeb/view?usp=drive_link).


### 📂 Dataset Structure
```
PlantDepth/
├── Crops3D/ # Sub-dataset
    ├── depth/                # Depth maps 
    ├── rgb/                  # RGB images 
    ├── segmentation/         # Segmentation labels 
    ├── test_file_list.txt    # List of test files used for evaluation
    ├── train_file_list.txt   # List of train files used for model training
├── ETH_BeetRoot/
    ├── ...
    ├── mask/ # Binary masks for the RGB-D datasets with backgrounds
├── GScatter/
├── PLANest/
├── Plant3D/
├── PlantStereo/
├── Soybeanmvs/
└── WUR_DwarfTomato/
```
### 🔗 Usage
We provide pytorch dataloader in `dataset/plantdepth.py`.
The dataset can be loaded by:
```
from dataset.plantdepth import PLANTDEPTH_MIXED
mix_set = ['GScatter','Crops3D','PLANest','Soybeanmvs','Plant3D'] # The sub-datasets you want load
root_dir = '.../PlantDepth' # The path for PlantDepth
trainset = PLANTDEPTH_MIXED(root_dir, [f'/{d}/train_file_list.txt' for d in mix_set], 'train', size=(518,518))
trainloader = DataLoader(trainset)
for i, sample in enumerate(trainloader):
    # Load RGB, Depth in disparity space, mask for background, and organ segmentation
    img, disparity, valid_mask, segmentation = sample['image'], sample['disparity'], sample['mask'], sample['segmentation']
    ...
    ...
```

## 📢 Citation
> Please cite the paper when using PlantMDE or PlantDepth in your work.
"""
