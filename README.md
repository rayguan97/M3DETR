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
Model Size (MB) | Total size of the resulting model (.pth).
Training VRAM (GB) | Amount of GPU memory reserved during training.
Training Time per Epoch (min) | Amount of time required to train a new model from scratch.
Inference VRAM (GB) | Amount of GPU memory reserved during inference.
Inference Time (min) | Time to run inference against the full validation set.

## Results

The following results were collected on a system with an NVIDIA RTX 3500 and 13th Gen Intel(R) Core(TM) i7-13850HX against the KITTI dataset. Training and validation used a batch size of 3 and training ran for 40 epochs.

#### Metric Results

Metric | Baseline (`main`) | FPS (`farthest-point-sampling`) | Keypoints (`keypoints`)
--- | --- | --- | ---
Training Time per Epoch (min) | 17.3 | 24.4 | 12.9
Training Reserved VRAM (GB) | 9.83 | 10.6 | 10.5
Model Size (MB) | 158 | ??? | 158
Inference Reserved VRAM (GB) | 2.89 | ??? | 10.5
Inference Time (min) | 7.4 | ??? | 4.0

#### Performance Results

<table><tr><td> Baseline </td> <td> FPS </td> <td> Keypoints </td></tr>
<tr><td> 

```
Car AP@0.70, 0.70, 0.70:
bbox AP:95.2200, 89.4991, 89.0343
bev  AP:90.4530, 88.0725, 87.4612
3d   AP:89.5997, 79.3243, 78.7122
aos  AP:95.17, 89.37, 88.80
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.1255, 94.1004, 91.8833
bev  AP:95.0918, 90.4623, 88.4262
3d   AP:92.1273, 82.8361, 81.9990
aos  AP:98.08, 93.92, 91.62
Car AP@0.70, 0.50, 0.50:
bbox AP:95.2200, 89.4991, 89.0343
bev  AP:95.2486, 89.4786, 89.0652
3d   AP:95.2084, 89.4525, 89.0269
aos  AP:95.17, 89.37, 88.80
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.1255, 94.1004, 91.8833
bev  AP:98.1435, 94.2184, 93.9731
3d   AP:98.0500, 94.1572, 93.8313
aos  AP:98.08, 93.92, 91.62
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:69.7248, 63.4667, 60.7511
bev  AP:61.7324, 54.8331, 50.1869
3d   AP:59.4383, 51.3683, 46.8981
aos  AP:64.03, 57.42, 54.81
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:70.5469, 64.0751, 60.7393
bev  AP:61.8852, 53.3970, 48.9260
3d   AP:58.8968, 50.2518, 44.9334
aos  AP:64.78, 58.25, 54.85
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:69.7248, 63.4667, 60.7511
bev  AP:73.7789, 68.0925, 64.6074
3d   AP:73.7772, 67.9958, 64.4526
aos  AP:64.03, 57.42, 54.81
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:70.5469, 64.0751, 60.7393
bev  AP:75.4135, 68.4483, 64.9348
3d   AP:75.4108, 68.3430, 64.8181
aos  AP:64.78, 58.25, 54.85
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:87.8234, 77.8562, 75.7233
bev  AP:82.2665, 69.7047, 65.3805
3d   AP:81.7396, 67.9010, 62.2965
aos  AP:87.72, 76.70, 74.55
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:92.6190, 80.4736, 76.6395
bev  AP:84.5956, 70.3986, 65.9891
3d   AP:84.0590, 68.0948, 63.4930
aos  AP:92.51, 79.21, 75.35
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:87.8234, 77.8562, 75.7233
bev  AP:84.3974, 72.3394, 70.1912
3d   AP:84.3974, 72.3394, 70.1912
aos  AP:87.72, 76.70, 74.55
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:92.6190, 80.4736, 76.6395
bev  AP:88.9538, 74.8053, 71.1938
3d   AP:88.9538, 74.8053, 71.1938
aos  AP:92.51, 79.21, 75.35
```

</td>
<td>

???

</td>
<td>

```
Car AP@0.70, 0.70, 0.70:
bbox AP:94.9234, 89.4703, 88.9115
bev  AP:90.1308, 87.7842, 86.7742
3d   AP:89.0632, 78.9549, 78.0183
aos  AP:94.83, 89.25, 88.57
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.0193, 93.9798, 91.7316
bev  AP:94.1815, 88.5378, 87.9349
3d   AP:91.7143, 82.4454, 79.6741
aos  AP:97.92, 93.70, 91.33
Car AP@0.70, 0.50, 0.50:
bbox AP:94.9234, 89.4703, 88.9115
bev  AP:94.9774, 89.5529, 89.0708
3d   AP:94.9209, 89.5122, 88.9926
aos  AP:94.83, 89.25, 88.57
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.0193, 93.9798, 91.7316
bev  AP:98.0297, 94.3425, 93.9074
3d   AP:98.0043, 94.2364, 93.6941
aos  AP:97.92, 93.70, 91.33
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:70.8399, 65.6474, 62.6417
bev  AP:63.2657, 55.2871, 51.9859
3d   AP:58.6724, 52.2804, 46.9396
aos  AP:66.16, 60.43, 57.27
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:71.7397, 66.3749, 62.6131
bev  AP:62.9394, 54.7850, 50.1815
3d   AP:58.0137, 50.4408, 45.5564
aos  AP:66.43, 60.37, 56.49
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:70.8399, 65.6474, 62.6417
bev  AP:76.5036, 72.0186, 68.1575
3d   AP:76.4965, 71.9064, 68.0216
aos  AP:66.16, 60.43, 57.27
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:71.7397, 66.3749, 62.6131
bev  AP:78.0136, 72.7060, 68.5944
3d   AP:77.9877, 72.5817, 68.4592
aos  AP:66.43, 60.37, 56.49
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:88.3821, 81.1925, 75.9476
bev  AP:84.7436, 73.1939, 68.9529
3d   AP:83.8723, 68.8823, 65.8029
aos  AP:87.06, 79.42, 74.12
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:92.7343, 82.7965, 78.6086
bev  AP:88.1289, 73.8403, 69.5300
3d   AP:86.7491, 69.4110, 65.2071
aos  AP:91.19, 80.80, 76.60
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:88.3821, 81.1925, 75.9476
bev  AP:86.8161, 77.1106, 74.3826
3d   AP:86.8161, 77.1106, 74.3826
aos  AP:87.06, 79.42, 74.12
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:92.7343, 82.7965, 78.6086
bev  AP:90.9542, 78.1351, 74.9714
3d   AP:90.9542, 78.1351, 74.9714
aos  AP:91.19, 80.80, 76.60
```

</td></tr></table>

## Baseline Reproduction

In order to reproduce the baseline results outlined in the original project we generally followed the steps outlined in their [README](https://github.com/rayguan97/M3DETR). However, as a few
small changes were required we reiterate the development environment setup here.

#### Development Environment
 
These instructions are tailored for Ubuntu 22.04 and require a decent NVIDIA GPU. The development environment requires [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html). With that installed, set up a `conda` environment and build the package:

```bash
conda create -n m3detr python=3.10 -y
conda activate m3detr
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install spconv-cu120 cython numpy==1.25
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
│   │   │   ├──calib & velodyne & label_2 & image_2
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── ...

# 2. process data and generate infos
cd M3DETR
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \ 
        tools/cfgs/dataset_configs/kitti_dataset.yaml
```

</td>
<td>

```bash
# 0. Enter conda environment, if not already active.
conda activate m3detr

# 1. Install waymo dataset configuration package
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-1-0

# 2. Register and download official Waymo Open Dataset: 
# https://waymo.com/open/download/
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
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
        --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
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
python -m torch.distributed.run --nproc_per_node=1 test.py \
    --launcher pytorch --cfg_file ./cfgs/kitti_models/M3DETR.yaml \
    --workers 1 --ckpt {PATH_TO_MODEL} --eval_tag evaluation --batch_size 1
```

</td>
<td>

```bash
# Enter conda environment, if not already active.
conda activate m3detr

# execute test script (e.g. for the pre-trained Waymo 1500 epoch model)
cd tools/
python -m torch.distributed.run --nproc_per_node=1 test.py \
    --launcher pytorch --cfg_file ./cfgs/waymo_models/M3DETR_1500.yaml \
    --workers 1 --ckpt {PATH_TO_MODEL} --eval_tag evaluation --batch_size 1
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
python -m torch.distributed.run --nproc_per_node=1 train.py \
    --launcher pytorch --cfg_file ./cfgs/kitti_models/M3DETR.yaml \
    --workers 1
```

</td>
<td>

```bash
# Enter conda environment, if not already active.
conda activate m3detr

# execute train script
cd tools/
python -m torch.distributed.run --nproc_per_node=1 train.py \
    --launcher pytorch --cfg_file ./cfgs/waymo_models/M3DETR_1500.yaml \
    --workers 1
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
