# A Plug-and-Play Knowledge Distillation Method based on Decoupling and Standardization

**Shao Hu**  
National Taiwan University of Science and Technology (NTUST)

This is the official repository for our paper: *(URL will be updated upon system upload)*

This repository is based on:
- [mdistiller](https://github.com/megvii-research/mdistiller)
- [RLD](https://github.com/zju-SWJ/RLD)

## Overview

Our work introduces a plug-and-play knowledge distillation method that combines scale decoupling distillation (SDD) with logit standardization. The method can be seamlessly integrated with existing knowledge distillation techniques to improve student model performance.

## Environment Setup

We recommend creating the environment using conda. Our environment has been tested with:
- **Python**: 3.8.20
- **PyTorch**: 1.9.0+cu111  
- **Torchvision**: 0.10.0+cu111
- **CUDA**: 11.1 (with NVCC 9.1 compatible)
- **Hardware**: RTX 3090 (CIFAR-100), RTX 4090 (ImageNet)

### Option 1: Using environment.yml (Recommended)
```bash
conda env create -f environment.yml
conda activate DSKD
```

### Option 2: Manual Installation
```bash
conda create -n DSKD python=3.8.20
conda activate DSKD
conda install cudatoolkit=10.2.89 -c pytorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardX==2.6.2.2 yacs==0.1.8 wandb==0.19.6 tqdm==4.67.1
pip install scipy==1.10.1 numpy==1.24.4 matplotlib==3.7.5 scikit-learn==1.3.2
```

### Option 3: Using requirements.txt
```bash
conda create -n DSKD python=3.8.20
conda activate DSKD
conda install cudatoolkit=10.2.89 -c pytorch
pip install -r requirements.txt
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3090/4090)
- **CUDA**: Compatible with CUDA 10.2+ and NVCC 9.1+
- **Memory**: Minimum 8GB GPU memory for CIFAR-100, 16GB+ recommended for ImageNet

## Datasets

### CIFAR-100
CIFAR-100 will be automatically downloaded when you first run the training scripts. The dataset will be saved to `./data/cifar100/` directory.

### ImageNet
Download the ImageNet dataset from [https://image-net.org/](https://image-net.org/) and organize it as follows:
```
./data/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## Pre-trained Teachers

### Option 1: Using our download script (Recommended)
```bash
chmod +x fetch_pretrained_teachers.sh
./fetch_pretrained_teachers.sh
```

### Option 2: Manual download
Pre-trained teacher models can be downloaded from [Decoupled Knowledge Distillation (CVPR 2022)](https://github.com/megvii-research/mdistiller).

1. Download `cifar_teachers.tar`
2. Extract to the checkpoint directory:
```bash
tar xvf cifar_teachers.tar
mv cifar_teachers ./download_ckpts/
```

The directory structure should be:
```
./save/models/
├── resnet56_vanilla/
├── resnet110_vanilla/
├── resnet32x4_vanilla/
├── ResNet50_vanilla/
├── wrn_40_2_vanilla/
└── vgg13_vanilla/
```

## Training Scripts

We provide a convenient shell script to run all experiments for each distillation method.

### Quick Start with Training Script

1. **Set execute permission**:
```bash
chmod +x run_cifar100_train.sh
```

2. **Edit the script parameters**:
```bash
# Open the script and modify these parameters at the top:
METHOD="dkd"                          # Change to: kd, dkd, or rld
MODEL="wrn_40_2_wrn_16_2"             # Your model configuration
GPU_ID="0"                            # GPU ID

# For RLD method only:
BASE_TEMP="2"                         # 2 for hetero models, 5 for homo models  
KD_WEIGHT="9"                         # 9 for hetero models, 6 for homo models
```

3. **Run the script**:
```bash
./run_cifar100_train.sh
```

### What the Script Does

The script automatically runs **all** experiments for the selected method:

**For KD and DKD:**
- Original method: `[1]` and `[1] + logit-stand`
- SDD variants: `[1]`, `[1,2]`, `[1,2,4]`
- SDD + logit standardization: `[1]`, `[1,2]`, `[1,2,4]` with `--logit-stand`

**For RLD:**
- Same as above, but includes `--base-temp` and `--kd-weight` parameters

### Examples

```bash
# Run all DKD experiments on WRN-40-2 -> WRN-16-2
# Edit: METHOD="dkd", MODEL="wrn_40_2_wrn_16_2"
./run_cifar100_train.sh

# Run all KD experiments on ResNet32x4 -> ResNet8x4
# Edit: METHOD="kd", MODEL="resnet32x4_resnet8x4"
./run_cifar100_train.sh

# Run all RLD experiments with custom parameters
# Edit: METHOD="rld", BASE_TEMP="5", KD_WEIGHT="6"
./run_cifar100_train.sh
```

## Manual Training Examples

### CIFAR-100

**Knowledge Distillation (KD)**
```bash
# Base KD
python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --M [1] --gpu 0

# KD with logit standardization
python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --M [1] --logit-stand --gpu 0

# KD with SDD (homogeneous models use [1,2], heterogeneous models use [1,2,4])
python tools/train.py --cfg configs/cifar100/sdd_kd/resnet32x4_resnet8x4.yaml --M [1,2] --gpu 0

# KD with ours (DSKD)
python tools/train.py --cfg configs/cifar100/sdd_kd/resnet32x4_resnet8x4.yaml --M [1,2] --logit-stand --gpu 0
```

**Decoupled Knowledge Distillation (DKD)**
```bash
# Base DKD
python tools/train.py --cfg configs/cifar100/dkd/wrn_40_2_wrn_16_2.yaml --M [1] --gpu 0

# DKD with logit standardization
python tools/train.py --cfg configs/cifar100/dkd/wrn_40_2_wrn_16_2.yaml --M [1] --logit-stand --gpu 0

# DKD with SDD (heterogeneous models use [1,2,4])
python tools/train.py --cfg configs/cifar100/sdd_dkd/wrn_40_2_wrn_16_2.yaml --M [1,2,4] --gpu 0

# DKD with ours (DSKD)
python tools/train.py --cfg configs/cifar100/sdd_dkd/wrn_40_2_wrn_16_2.yaml --M [1,2,4] --logit-stand --gpu 0
```

**Refined Logit Distillation (RLD)**
```bash
# Base RLD (heterogeneous models: base-temp=2, kd-weight=9)
python tools/train.py --cfg configs/cifar100/rld/wrn_40_2_resnet20.yaml --M [1] --base-temp 2 --kd-weight 9 --gpu 0

# RLD with logit standardization
python tools/train.py --cfg configs/cifar100/rld/wrn_40_2_resnet20.yaml --M [1] --logit-stand --base-temp 2 --kd-weight 9 --gpu 0

# RLD with SDD (heterogeneous models use [1,2,4])
python tools/train.py --cfg configs/cifar100/sdd_rld/wrn_40_2_resnet20.yaml --M [1,2,4] --base-temp 2 --kd-weight 9 --gpu 0

# RLD with ours (DSKD)
python tools/train.py --cfg configs/cifar100/sdd_rld/wrn_40_2_resnet20.yaml --M [1,2,4] --logit-stand --base-temp 2 --kd-weight 9 --gpu 0

# For homogeneous models (e.g., ResNet56 -> ResNet20): use base-temp=5, kd-weight=6, scales=[1,2]
python tools/train.py --cfg configs/cifar100/sdd_rld/resnet56_resnet20.yaml --M [1,2] --logit-stand --base-temp 5 --kd-weight 6 --gpu 0
```

### ImageNet

**Knowledge Distillation (KD)**
```bash
# Base KD
python tools/train.py --cfg configs/imagenet/kd/ResNet34_ResNet18.yaml --M [1] --gpu 0

# KD with ours (DSKD)
python tools/train.py --cfg configs/imagenet/sdd_kd/ResNet34_ResNet18.yaml --M [1,2] --logit-stand --gpu 0
```

**Decoupled Knowledge Distillation (DKD)**
```bash
# Base DKD
python tools/train.py --cfg configs/imagenet/dkd/ResNet34_ResNet18.yaml --M [1] --gpu 0

# DKD with ours (DSKD)
python tools/train.py --cfg configs/imagenet/sdd_dkd/ResNet34_ResNet18.yaml --M [1,2] --logit-stand --gpu 0
```

**Refined Logit Distillation (RLD)**
```bash
# Base RLD
python tools/train.py --cfg configs/imagenet/rld/ResNet34_ResNet18.yaml --M [1] --base-temp 2 --kd-weight 9 --gpu 0

# RLD with ours (DSKD)
python tools/train.py --cfg configs/imagenet/sdd_rld/ResNet34_ResNet18.yaml --M [1,2] --logit-stand --base-temp 2 --kd-weight 9 --gpu 0
```

## Supported Distillation Methods

Our framework supports various knowledge distillation methods:
- **KD**: Vanilla Knowledge Distillation
- **DKD**: Decoupled Knowledge Distillation
- **RLD**: Refined Logit Distillation
- **AT**: Attention Transfer
- **FitNet**: Hints for Thin Deep Nets
- **ReviewKD**: Reviewing Knowledge Distillation
- **CRD**: Contrastive Representation Distillation
- And more...

Each method can be enhanced with:
- **SDD**: Scale Decoupling Distillation (`--M '[1,2,4]'`)
- **Logit Standardization**: (`--logit-stand`)

## Key Parameters

- `--M`: Scale levels for SDD (options: `'[1]'`, `'[1,2]'`, `'[1,2,4]'`)
- `--logit-stand`: Enable logit standardization
- `--aug`: Enable data augmentation
- `--base-temp`: Base temperature for knowledge distillation (default: 2.0)
- `--kd-weight`: Weight for KD loss (default: 9.0)

## Configuration Files

Configuration files are located in `configs/`:
- `configs/cifar100/`: CIFAR-100 experiments
- `configs/imagenet/`: ImageNet experiments

Each distillation method has its own configuration directory with teacher-student pairs.

## Evaluation

To evaluate a trained model:
```bash
python tools/eval.py -m resnet8x4 -c path/to/checkpoint.pth -d cifar100
```

## Results and Logs

We put the training logs in `./logs` and hyper-linked below. The name of each log file is formatted with `KD_TYPE,TEACHER,STUDENT,BASE_TEMPERATURE,KD_WEIGHT.txt`. The possible third value for DKD is the value of BETA. Due to average operation and randomness, there may be slight differences between the reported results and the logged results.

### CIFAR-100 Results

**Teacher and student have identical structures:**

| Teacher<br>Student | ResNet32x4<br>ResNet8x4 | VGG13<br>VGG8 | WRN-40-2<br>WRN-40-1 | WRN-40-2<br>WRN-16-2 | ResNet56<br>ResNet20 | ResNet110<br>ResNet32 | ResNet110<br>ResNet20 |
|-------------------|-------------------------|---------------|----------------------|----------------------|---------------------|----------------------|---------------------|
| **KD** | 73.33 | 72.98 | 73.54 | 74.92 | 70.66 | 73.08 | 70.67 |
| **KD+Ours** | **77.62** | **74.49** | **75.28** | **76.14** | **71.57** | **73.71** | **71.86** |
| | | | | | | | |
| **DKD** | 76.32 | 74.68 | 74.81 | 76.24 | 71.97 | 74.11 | 71.06 |
| **DKD+Ours** | **76.94** | **75.17** | **74.90** | **76.31** | **72.14** | **74.30** | **71.90** |
| | | | | | | | |
| **RLD** | 76.64 | 74.93 | 74.88 | 76.02 | 72.00 | 74.02 | 71.67 |
| **RLD+Ours** | **76.99** | **74.90** | **75.20** | **76.05** | **71.50** | **73.85** | **71.49** |

**Teacher and student have distinct structures:**

| Teacher<br>Student | ResNet32x4<br>ShuffleNet-V2 | ResNet32x4<br>WRN-16-2 | ResNet32x4<br>WRN-40-2 | WRN-40-2<br>ResNet8x4 | WRN-40-2<br>MobileNet-V2 | VGG13<br>MobileNet-V2 | ResNet50<br>MobileNet-V2 |
|-------------------|------------------------------|-------------------------|-------------------------|----------------------|---------------------------|-------------------------|---------------------------|
| **KD** | 74.45 | 74.90 | 77.70 | 73.97 | 68.36 | 67.37 | 67.35 |
| **KD+Ours** | **78.50** | **76.46** | **79.52** | **76.83** | **70.61** | **68.77** | **71.18** |
| | | | | | | | |
| **DKD** | 77.07 | 75.70 | 78.46 | 75.56 | 69.38 | 69.71 | 70.35 |
| **DKD+Ours** | **78.84** | **76.94** | **79.41** | **76.33** | **70.55** | **70.76** | **72.12** |
| | | | | | | | |
| **RLD** | 77.56 | 76.14 | 78.91 | 76.12 | 69.75 | 69.97 | 70.76 |
| **RLD+Ours** | **77.99** | **76.42** | **79.46** | **76.40** | **71.03** | **70.25** | **71.78** |

## Acknowledgments

This work is based on the excellent frameworks:
- [mdistiller](https://github.com/megvii-research/mdistiller) by Megvii Research
- [RLD](https://github.com/zju-SWJ/RLD) by Zhejiang University
- [Logit Standardization KD](https://github.com/sunshangquan/logit-standardization-KD)
- [SDD](https://github.com/shicaiwei123/SDD-CVPR2024)

## License

This project is licensed under the MIT License - see the LICENSE file for details.