<p align="center">
  <h1 align="center"> CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection</h1>
  <p align="center">
<a href="https://sunzc-sunny.github.io/">Zhichao Sun</a>, Huazhang Hu, Yidong Ma, Gang Liu, Nemo Chen, Xu Tang, Yao Hu, Yongchao Xu</a>
  </p>
  <h4 align="center"> Wuhan university;     Xiaohongshu Inc.
  <h3 align="center"><a href="https://arxiv.org/pdf/2503.18430">Paper</a>
  <!-- <div align="center"></div> -->
</p>

## Overview
We propose CQ-DINO, a category query-based object detection framework for vast vocabulary object detection.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model](#model)
- [Citation](#Citation)

## Installation

Please follow the guide to install and set up of the mmdetection.
```
conda create --name openmmlab python=3.10.6 -y
conda activate openmmlab

pip install -U openmim
mim install mmengine
mim install "mmcv==2.2.0"
```

```
git clone git@github.com:RedAIGC/CQ-DINO.git
cd CQ-DINO
pip install -v -e .
```


## Dataset
### V3Det

Please download and prepare V3Det Dataset at [V3Det Homepage](https://v3det.openxlab.org.cn/) and [V3Det Github](https://github.com/V3Det/V3Det). After downloading and unzipping, place the dataset or create a symbolic link to it in the data/v3det directory, with the following directory structure:
```
CQ-DINO
├── configs
├── data
│   ├── v3det
│   │   ├── annotations
│   │   |   ├── v3det_2023_v1_train.json
│   │   ├── images
│   │   │   ├── a00000066
│   │   │   │   ├── xxx.jpg
│   │   │   ├── ...
```
Then use [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) to convert it into the ODVG format required for training:
```shell
python tools/dataset_converters/coco2odvg.py data/v3det/annotations/v3det_2023_v1_train.json -d v3det
```
After the program has run, two new files `v3det_2023_v1_train_od.json` and `v3det_2023_v1_label_map.json` will be created in the `data/v3det/annotations` directory, with the complete structure as follows:
```text
CQ-DINO
├── configs
├── data
│   ├── v3det
│   │   ├── annotations
│   │   |   ├── v3det_2023_v1_train.json
│   │   |   ├── v3det_2023_v1_train_od.json
│   │   |   ├── v3det_2023_v1_label_map.json
│   │   ├── images
│   │   │   ├── a00000066
│   │   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

### COCO 2017
Please download it from the [COCO](https://cocodataset.org/) official website or from [opendatalab](https://opendatalab.com/OpenDataLab/COCO_2017).
After downloading and unzipping, place the dataset or create a symbolic link to the `data/coco` directory. The directory structure is as follows:
```text
CQ-DINO
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```
Then use [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) to convert it into the ODVG format required for training:
```shell
python tools/dataset_converters/coco2odvg.py data/coco/annotations/instances_train2017.json -d coco
```
This will generate new files, `instances_train2017_od.json` and `coco2017_label_map.json`, in the `data/coco/annotations/` directory. The complete dataset structure is as follows:
```text
CQ-DINO
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_train2017_od.json
│   │   │   ├── coco2017_label_map.json
│   │   │   ├── instances_val2017.json
│   │   ├── train2017
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val2017
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
```

## Usage
Download the first stage parametres from [Google drive]() in the directory [stage1](/stage1/).  
Download the category embeddings from [Google drive]().\
The complete  structure is as follows:
``` text
CQ-DINO
├── configs
├── stage1
│   ├── cqdino_swinb1k_v3det_stage1.pth
│   ├── cqdino_swinb22k_v3det_stage1.pth
│   ├── cqdino_swinl_coco_stage1.pth
│   ├── cqdino_swinl_v3det_stage1.pth
├── v3det_clip_embeddings.pth
├── coco_clip_embeddings.pth
```

### Train

```bash
# single gpu
python tools/train.py configs/cqdino/cqdino_tree_swinb22k_v3det.py 

# multi gpu
bash tools/dist_train.sh configs/cqdino/cqdino_tree_swinb22k_v3det.py  NUM_GPUs
```

### Inference
Download the checkpoint from [Google drive]()

```bash
# single gpu
python tools/test.py configs/cqdino/cqdino_tree_swinb22k_v3det.py cqdino_swinb22k_v3det.pth

# multi gpu
bash tools/dist_test.sh configs/cqdino/cqdino_tree_swinb22k_v3det.py cqdino_swinb22k_v3det.pth NUM_GPUs
```

## Citation
```bibtex
@article{sun2025cq,
         title={{CQ-DINO}: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection},
         author={Sun, Zhichao and Hu, Huazhang and Ma, Yidong and Liu, Gang and Chen, Nemo and Tang, Xu and Xu, Yongchao},
         journal={arXiv preprint arXiv:2503.18430},
         year={2025}
}
```
