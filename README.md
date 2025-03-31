# PlantMDE
The repo for ``3D Plant Phenotyping from a Single Image: Generalizable Plant Monocular Depth Estimation via Dataset Mixing''
## 🌱 PlantDepth Benchmark Dataset 
Tthe first benchmark dataset designed for **plant depth estimation** and **3D reconstruction tasks**. It renders RGB-D plant images from the mainstream publicly available plant 3D datasets. The dataset can be downloaded by the [link](https://drive.google.com/file/d/1XbDwjUn16dVl7F6uGNvpgtFU-uLm1jeb/view?usp=drive_link).


---

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
├── GScatter/
├── PLANest/
├── Plant3D/
├── PlantStereo/
├── Soybeanmvs/
└── WUR_DwarfTomato/
```
