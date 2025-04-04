# PlantMDE
The repo for ``3D Plant Phenotyping from a Single Image: Generalizable Plant Monocular Depth Estimation via Dataset Mixing''.
![Graphical_Abstract](assets/Graphical_Abstract.png)
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
## PlantMDE model
Trained PlantMDE checkpoint can be downloaded in the [link](https://drive.google.com/file/d/1DL8ER3Tl2bZzm0CNcrFrrOoiIn_tTnuh/view?usp=drive_link).
As a comparion, Depth Anything v2 checkpoint can be downloaded in the [repo](https://github.com/DepthAnything/Depth-Anything-V2) with [link](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true).

`MDE_organ_wise.ipynb` provides a tuition for depth estiamtion on spinach image. The Pearson correlation coefficient to the GT depth at each organ is computed. The notebook compares PlantMDE vs. Depth Anything. 
## 📢 Citation
> Please cite the paper when using PlantMDE or PlantDepth in your work.
"""
