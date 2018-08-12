# Fine-Grained Representation Learning and Recognition by Exploiting Hierarchical Semantic Embedding

# Installation
This repository contains the pytorch codes for deployment and the trained models described in the paper "Fine-Grained Representation Learning and Recognition by Exploiting Hierarchical Semantic Embedding"[PDF coming soon]().
**Results**

- Caltech UCSD Birds

|  |order|family|genus|class|
|----｜:----:|:----:|:----:|:----:|
|baseline｜98.8|95.3|91.8|85.7|
|HSE(ours)｜98.8|95.7|92.7|88.1|


- Butterfly200

|  |family|subfamily|genus|class|
|----｜:----:|:----:|:----:|:----:|
|baseline｜99.0|97.6|94.8|85.1|
|HSE(ours)｜99.0|97.7|95.4|86.1|

- Vegfru

|  |sup|sub|
|----｜:----:|:----:|
|baseline｜90.0|88.2|
|HSE(ours)｜90.0|89.4|

# Deployment

**Requirement**

- pytorch, tested on [v0.4.0](http://download.pytorch.org/whl/cu80/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl)
- scipy
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
|-- data
|   |-- CUB_2011_200
|   |-- Butterfly200
|   |-- Verfru
|   |-- ...
|-- models
|   |-- CUB_2011_200
|   |-- Butterfly200
|   |-- Verfru
|   |-- ...        
|-- code
|   |-- CUB_2011_200
|   |-- Butterfly200
|   |-- Verfru
|   |-- ...
|-- scripts
|   |-- CUB_2011_200
|   |-- Butterfly200
|   |-- Verfru
|   |-- ...
```

## 2. Download datasets
The datasets can be downloaded here:

[Caltech UCSD Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html) , and its [hierarchical category annotations](https://www.dropbox.com/sh/kugj7vogy2no795/AABJWUxM6rXWOeNbCUPj269ua?dl=0) provided by us.

[Butterfly 200](https://www.dropbox.com/sh/3p4x1oc5efknd69/AABwnyoH2EKi6H9Emcyd0pXCa?dl=0) (including its hierarchical category annotation)

[Vegfru](https://github.com/ustc-vim/vegfru) 

## 2. Download trained models
The models can be downloaded via [BaiduyunDisk]() or [Dropbox(coming soon)]():

- Caltech UCSD Birds

HSE: [HSE_4levels]()

baseline: [ordel](), [family](), [genus](), [class]()

- Butterfly200

HSE: [HSE_4levels]()

baseline: [family](), [subfamily](), [genus](), [class]()

- Vegfru

HSE: [HSE_2levels]()

baseline: [sup](), [sub]()

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

The above scripts deploy the prediction for the whole testset, of which the results are report on the MM2018 paper.


# License
HSE is released under the SYSU License (refer to the LICENSE file for details).
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