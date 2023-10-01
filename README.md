# M3DETR Optimization

This project is an optimization effort for the [M3DETR](https://github.com/rayguan97/M3DETR) 3D Object Detector. Please see 
the original [website](https://openaccess.thecvf.com/content/WACV2022/html/Guan_M3DETR_Multi-Representation_Multi-Scale_Mutual-Relation_3D_Object_Detection_With_Transformers_WACV_2022_paper.html)
and [paper](https://arxiv.org/pdf/2104.11896.pdf) for detailed explanations and attribution. The following assumes familiarity with the original project.

## Overview

The M3DETR architecture has shown impressive results for 3D Object Detection against a number of open dataset challenges, including the Waymo Open Dataset and the KITTI 3D Detection Benchmark.
This project seeks to optimize the architecture to support inference and training on devices with less compute power than required by the current architecture. Note that we deliberately
eschew improvements to the _performance_ metrics employed in these benchmarks and challenges. We seek to maintain the advertised performance metrics while reducing a different set of metrics:

## Metrics

Metric | Explanation
--- | ---
Model Size (GB) | Total size of the resulting model (.pth).
Training VRAM (GB) | Amount of GPU memory required during training.
Inference VRAM (GB) | Amount of GPU memory required during inference.
Training Time (h) | Amount of time required to train a new model from scratch.
Inference Rate (hz) | Rate at which inference can be run.

## Baseline Reproduction

In order to reproduce the baseline results outlined in the original project we generally followed the steps outlined in their [README](https://github.com/rayguan97/M3DETR). However, as a few
small changes were required we reiterate the development environment setup here.

#### Development Environment
 
These instructions are tailored for Ubuntu 22.04 and require a decent NVIDIA GPU. The development environment requires [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html). With that installed, set up a `conda` environment and build the package:

```bash
conda create -n m3detr python=3.6 -y
conda activate m3detr
conda install pytorch=1.9.1 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install spconv-cu113	
pip install pyyaml numba llvmlite tensorboardX SharedArray easydict tqdm scipy scikit-image imageio
git clone https://github.com/danielmohansahu/M3DETR.git
cd M3DETR
python setup.py develop
```

#### Dataset Configuration

The instructions for each dataset differ slightly, but are essentially duplicative of the original [OpenPCDNet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) instructions.

<table><tr><td> KITTI </td> <td> WAYMO </td></tr>
<tr><td> 

```bash
# 0. Enter conda environment, if not already active.
conda activate m3detr

# 1. Register and download official KITTI 3D Object Detection dataset:
# http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

# Expected folder structure:
tree M3DETR
M3DETR
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── ...

# 2. process data and generate infos
cd M3DETR
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

</td>
<td>

```bash
# 0. Enter conda environment, if not already active.
conda activate m3detr

# 1. Install waymo dataset configuration package
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-1-0

# 2. Register and download official Waymo Open Dataset: https://waymo.com/open/download/
# Get version v1.4.2 and download the archived files.
# Extract all *.tar files to the 'data/waymo/raw_data' directory.

# Expected folder structure:
tree M3DETR
M3DETR
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
|   |   |── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1_multiframe_-4_to_0.pkl
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0_global.np
├── pcdet
├── ...

# 3. process data and generate infos
# N.B. This is _extremely_ slow, and could take several days.
cd M3DETR
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

</td></tr></table>

#### Testing Instructions

The original authors maintain a model zoo of baseline results and trained models available for download in the [M3DETR Model Zoo](MODEL_ZOO.md).
To evaluate these models (or ones trained locally), run the following:

<table><tr><td> KITTI </td> <td> WAYMO </td></tr>
<tr><td> 

```bash
# Enter conda environment, if not already active.
conda activate m3detr

# execute test script (e.g. for the pre-trained Kitti model)
cd tools/
python -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
    --cfg_file ./cfgs/m3detr_models/m3detr_kitti.yaml --workers 1 \
    --ckpt {PATH_TO_MODEL} --eval_tag evaluation --batch_size 1
```

</td>
<td>

```bash
# Enter conda environment, if not already active.
conda activate m3detr

# execute test script (e.g. for the pre-trained Waymo 1500 epoch model)
cd tools/
python -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
    --cfg_file ./cfgs/m3detr_models/m3detr_waymo_1500.yaml --workers 1 \
    --ckpt {PATH_TO_MODEL} --eval_tag evaluation --batch_size 1
```

</td></tr></table>

#### Training Instructions

To train a model from scratch using an existing dataset config:

<table><tr><td> KITTI </td> <td> WAYMO </td></tr>
<tr><td> 

```bash
# Enter conda environment, if not already active.
conda activate m3detr

# execute train script
cd tools/
python -m torch.distributed.launch --nproc_per_node=1 train.py --launcher pytorch \
    --cfg_file ./cfgs/m3detr_models/m3detr_kitti.yaml --workers 1
```

</td>
<td>

```bash
# Enter conda environment, if not already active.
conda activate m3detr

# execute train script
cd tools/
python -m torch.distributed.launch --nproc_per_node=1 train.py --launcher pytorch \
    --cfg_file ./cfgs/m3detr_models/m3detr_waymo_1500.yaml --workers 1
```

</td></tr></table>

## References
 - [M3DETR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers](https://arxiv.org/pdf/2104.11896.pdf)
 - [PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](https://arxiv.org/abs/1912.13192)
 - [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
 - [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396)
 - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
 - [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
 - [LineFormer: Rethinking Line Chart Data Extraction as Instance Segmentation](https://arxiv.org/abs/2305.01837)


## Citation
Please cite the original authors work if you found it useful,

```
@InProceedings{Guan_2022_WACV,
    author    = {Guan, Tianrui and Wang, Jun and Lan, Shiyi and Chandra, Rohan and Wu, Zuxuan and Davis, Larry and Manocha, Dinesh},
    title     = {M3DETR: Multi-Representation, Multi-Scale, Mutual-Relation 3D Object Detection With Transformers},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {772-782}
}
```
