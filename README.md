# Fine-Grained Representation Learning and Recognition by Exploiting Hierarchical Semantic Embedding

# Introduction
This repository contains the pytorch codes, trained models, and datasets described in the paper "Fine-Grained Representation Learning and Recognition by Exploiting Hierarchical Semantic Embedding" [[PDF coming soon]()].

**Results**

- Accuracy on Caltech UCSD Birds

|        | order | family | genus | class |
| :----: | :---: | :----: | :---: | :---: |
|baseline| 98.8  |  95.0  |  91.5 |  85.2 |
|HSE(ours)| 98.8 |  95.7  |  92.7 |  88.1 |



- Accuracy on Butterfly200

|        | family | subfamily | genus | species |
| :----: | :----: | :-------: | :---: | :-----: |
|baseline|  98.9  |   97.6    |  94.8 |  85.1   |
|HSE(ours)| 98.9  |   97.7    |  95.4 |  86.1   |

- Accuracy on Vegfru

|           |  sup  |  sub  |
| :-------: | :---: | :---: |
|  baseline | 90.0  |  87.1 |
| HSE(ours) | 90.0  |  89.4 |


# Installation

**Requirement**

- pytorch, tested on [v0.4.0](http://download.pytorch.org/whl/cu80/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl)
- CUDA, tested on v8.0
- Language: Python 2.7


## 1. Clone the repository
Clone the Hierarchical Semantic Embedding project by:
```
git clone https://github.com/HCPLab-SYSU/HSE.git
```
and we denote the folder `hse-mm2018` as `$HSE_ROOT`.

**Note that**, the correct structure of `$HSE_ROOT` is like:

```
hse-mm2018
.
├── code
│   ├── Butterfly200
│   │   ├── baseline
│   │   └── HSE
│   ├── CUB_200_2011
│   │   ├── baseline
│   │   └── HSE
│   └── Vegfru
│       ├── baseline
│       └── HSE
├── data
│   ├── Butterfly200
│   │   └── images
│   ├── CUB_200_2011
│   │   └── images
│   └── Vegfru
│       └── images
├── models
│   ├── Butterfly200
│   ├── CUB_200_2011
│   └── Vegfru
└── scripts

```

## 2. Download datasets

[Caltech UCSD Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html) originally covers 200 classes of birds, and we  extend this dataset with a [four-level category hierarchy](https://www.dropbox.com/sh/kugj7vogy2no795/AABJWUxM6rXWOeNbCUPj269ua?dl=0) .

[Butterfly 200](https://www.dropbox.com/sh/3p4x1oc5efknd69/AABwnyoH2EKi6H9Emcyd0pXCa?dl=0) is constructed in our paper, it also cover four-level categories.

[Vegfru](https://github.com/ustc-vim/vegfru) is proposed by [Hou et al., ICCV2017](http://home.ustc.edu.cn/~saihui/project/vegfru/iccv17_vegfru.pdf), and it covers two-level categories.

**Note th**

## 2. Download trained models
The models can be downloaded from [BaiduyunDisk (all)](https://pan.baidu.com/s/1WWalFQFiNCCrWr30pvEA6A):

- Caltech UCSD Birds

[all models](https://pan.baidu.com/s/1LO_31tJe76DJFXXfc21STQ)

HSE: [HSE_4levels](https://pan.baidu.com/s/1GIGQQFzCy9GVhdwgDTPVdg)

baseline: [ordel](https://pan.baidu.com/s/1tOmXoiA1RESF1JMnUpbq9w), [family](https://pan.baidu.com/s/1FBYyPvb5eputq6BVysXURQ), [genus](https://pan.baidu.com/s/1tNb9pjI-Irwyqynott27ag), [class](https://pan.baidu.com/s/1R0VghAkl7zzQODjyitWxsA)

- Butterfly200

[all models](https://pan.baidu.com/s/1tOmXoiA1RESF1JMnUpbq9w)

HSE: [HSE_4levels](https://pan.baidu.com/s/1-a734IsAJYtjH2INCUxkvQ)

baseline: [family](https://pan.baidu.com/s/1UO2q5XMUtALm1C2apNW6nA), [subfamily](https://pan.baidu.com/s/1hYSDbMtbDO9N48zmmlWs1w), [genus](https://pan.baidu.com/s/1DrkUwIMnphO-R7Fru0FG_A), [species](https://pan.baidu.com/s/1EFTqiwVb_Y0Bfj9bcu9eIA)

- Vegfru

[all models](https://pan.baidu.com/s/1uYEZFFX6dxvwam4XTMH_XQ)

HSE: [HSE_2levels](https://pan.baidu.com/s/1u6WlCVXEgC189xfcRRbmjg)

baseline: [sup](https://pan.baidu.com/s/17GfovAVzBwH_6gLplzjbJw), [sub](https://pan.baidu.com/s/16D6Sm8aJNMVu7T7BIlArZg)

or Download from [Dropbox (coming soon)]().

## 3. Deployment
Firstly, make sure the working directory is `$HSE_ROOT`, or
```
cd $HSE_ROOT
```
then, run the deployment script:
## deploy HSE
```
./scripts/deploy_hse.sh [GPU_ID] [DATASET]
----
GPU_ID: required, 0-based int, 
DATASET: required, 'CUB_200_2011' or 'Butterfly200' or 'Vegfru'
```
## deploy baseline
```
./scripts/deploy_baseline.sh [GPU_ID] [DATASET] [LEVEL]
----
GPU_ID: required, 0-based int, 
DATASET: required, 'CUB_200_2011' or 'Butterfly200' or 'Vegfru'
LEVEL: require, 
    CUB_200_2011: LEVEL is chosen in ['order', 'family', 'genus', 'class']
    Butterfly200: LEVEL is chosen in ['family', 'subfamily', 'genus', 'species']
    Vegfru: LEVEL is chosen in ['sup', 'sub']
```

# License
The code is released under the SYSU License (refer to the LICENSE file for details).
The [Human Cyber Physical Intelligence Integration Lab](http://www.sysu-hcp.net/home/) owns this project.

# Citing
```
@inproceedings{chen2018fine,
    Author = {Tianshui Chen, Wenxi Wu, Yuefang Gao, Le Dong, Xiaonan Luo, Liang Lin},
    Title = {Fine-Grained Representation Learning and Recognition by Exploiting Hierarchical Semantic Embedding},
    Booktitle = {Proc. of ACM International Conference on Multimedia (ACM MM)},
    Year = {2018}
} 
```

# Contributing
For any questions, feel free to open an issue or contact us. ([tianshuichen@gmail.com]() or [ngmanhei@foxmail.com]())
