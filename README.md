# Adaptive Multi-Scale Feature Aggregation Transformer Network for Single Remote Sensing Image Super-Resolution

## Authors
Official PyTorch implementation of the paper “Adaptive Multi-Scale Feature Aggregation Transformer Network for Single Remote Sensing Image Super-Resolution” (**Accepted by Geo-spatial Information Science**).  
**Authors:** Zhiqi Zhang, Qi Sun, Zhiwei Ye, Chuang Liu, Mi Wang

## Abstract
Remote sensing image super-resolution (RSISR) plays a key role in recovering spatial detail and improving image quality from satellite imagery. In recent years, Transformer-based methods have shown excellent performance in RSISR tasks. However, despite the higher computational efficiency of the local self-attention calculation compared to the global self-attention calculation, its limited receptive field restricts the model from effectively modelling the complex scale diversity and long-range dependencies of ground observation targets. Moreover, the intermediate features of existing methods contain blocking artifacts, leading to different degrees of feature edge distortion and texture detail loss. To address the above issues, this paper proposes the Adaptive Multi-scale Feature Aggregation Transformer Network (AMFAT), which improves the feature representation capability through dynamic weighting and cross-window interacting. Specifically, the Adaptive Context Channel Attention (ACCA) is designed to fuse multi-branch features using dynamic weights for object-guided context adaptation. In addition, the Mixed-Scale Token Attention (MSTA) is constructed to eliminate blocking artifacts through cross-window interaction. Meanwhile, simple gating units with spatial enhancement operations are introduced into the feed-forward network (FFN) to optimize local feature aggregation. We conducted extensive experiments on four publicly available remote sensing datasets, and the results show that, compared to other methods, AMFAT exhibits excellent performance and adaptability both in terms of quantitative metrics and visual quality.

## Network
![AMFAT Network](amfat.png)

## Installation & Requirements
This project follows the environment setup and usage of **HAT**. Please refer to the HAT repository for detailed installation instructions (Python, PyTorch, dependencies, dataset preparation, and training/testing pipelines): <https://github.com/XPixelGroup/HAT>.  
Our implementation is integrated in the same style (configs in `*.yml`, architecture file `amfat_arch.py`, pretrained weights `amfat_x4.pth`).

## Datasets
- **AID** (Aerial Image Dataset): <https://captain-whu.github.io/AID/>
- **UC Merced Land Use Dataset**: <https://weegee.vision.ucmerced.edu/datasets/landuse.html>
- **NWPU-RESISC45**: <https://gcheng-nwpu.github.io/>
- **RSSCN7**: <https://sites.google.com/view/zhouwx/dataset> *(mirror: <https://figshare.com/articles/dataset/RSSCN7_Image_dataset/7006946>)*

## How To Train
```bash
python hat/train.py -opt train_x4.yml
```

## How To Test
```bash
python hat/test.py -opt test_x4.yml
```

## Acknowledgements
This code is built on **HAT** (<https://github.com/XPixelGroup/HAT>) and **BasicSR** (<https://github.com/XPixelGroup/BasicSR>). Thanks for these excellent open-source works!
