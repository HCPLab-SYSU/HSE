# Disclaimer

# Installation
**requirement**

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

[Caltech UCSD Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html) , and its [hierarchical category annotations]() provided by us.

[Butterfly 200]() (including its hierarchical category annotation)

[Vegfru](https://github.com/ustc-vim/vegfru) 

## 2. Download trained models
The models can be downloaded here:

- Caltech UCSD Birds
baseline: [ordel](), [family](), [genus](), [class]()
HSE: [HSE_4levels]()

- Butterfly200
baseline: [family](), [subfamily](), [genus](), [class]()
HSE: [HSE_4levels]()

- Vegfru
baseline: [sup](), [sub]()
HSE: [HSE_2levels]()


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
@inproceedings{tsMM2018HSE,
    Author = {Tianshui Chen, Wenxi Wu, Yuefang Gao, Le Dong, Xiaonan Luo, Liang Lin},
    Title = {Fine-Grained Representation Learning and Recognition by Exploiting Hierarchical Semantic Embedding},
    Booktitle = {Proc. of ACM International Conference on Multimedia (ACM MM)},
    Year = {2018}
} 
```