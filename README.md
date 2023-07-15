# M3DETR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m3detr-multi-representation-multi-scale/3d-object-detection-on-kitti-cars-hard-val)](https://paperswithcode.com/sota/3d-object-detection-on-kitti-cars-hard-val?p=m3detr-multi-representation-multi-scale) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m3detr-multi-representation-multi-scale/3d-object-detection-on-kitti-cars-hard)](https://paperswithcode.com/sota/3d-object-detection-on-kitti-cars-hard?p=m3detr-multi-representation-multi-scale) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m3detr-multi-representation-multi-scale/3d-object-detection-on-kitti-cyclist-hard-val)](https://paperswithcode.com/sota/3d-object-detection-on-kitti-cyclist-hard-val?p=m3detr-multi-representation-multi-scale) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m3detr-multi-representation-multi-scale/3d-object-detection-on-kitti-cyclists-hard)](https://paperswithcode.com/sota/3d-object-detection-on-kitti-cyclists-hard?p=m3detr-multi-representation-multi-scale) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m3detr-multi-representation-multi-scale/3d-object-detection-on-kitti-pedestrian-hard)](https://paperswithcode.com/sota/3d-object-detection-on-kitti-pedestrian-hard?p=m3detr-multi-representation-multi-scale) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m3detr-multi-representation-multi-scale/3d-object-detection-on-kitti-pedestrians-hard)](https://paperswithcode.com/sota/3d-object-detection-on-kitti-pedestrians-hard?p=m3detr-multi-representation-multi-scale) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m3detr-multi-representation-multi-scale/3d-object-detection-on-waymo-vehicle)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle?p=m3detr-multi-representation-multi-scale) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m3detr-multi-representation-multi-scale/3d-object-detection-on-waymo-cyclist)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-cyclist?p=m3detr-multi-representation-multi-scale) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m3detr-multi-representation-multi-scale/3d-object-detection-on-waymo-pedestrian)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-pedestrian?p=m3detr-multi-representation-multi-scale) \
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/rayguan97/M3DETR/.github%2Fworkflows%2Fpython-package-conda.yml)
[![GitHub Repo stars](https://img.shields.io/github/stars/rayguan97/M3DETR)](https://github.com/rayguan97/M3DETR/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/rayguan97/M3DETR)](https://github.com/rayguan97/M3DETR/network)
[![GitHub issues](https://img.shields.io/github/issues/rayguan97/M3DETR)](https://github.com/rayguan97/M3DETR/issues)
[![GitHub](https://img.shields.io/github/license/rayguan97/M3DETR)](https://github.com/rayguan97/M3DETR/blob/main/LICENSE)


<!--- ![Codacy Badge](https://api.codacy.com/project/badge/Grade/63847d9328f64fce9c137b03fcafcc27) -->


The code base for [M3DETR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers](https://openaccess.thecvf.com/content/WACV2022/html/Guan_M3DETR_Multi-Representation_Multi-Scale_Mutual-Relation_3D_Object_Detection_With_Transformers_WACV_2022_paper.html)
<br>**Tianrui Guan***, **Jun Wang***, Shiyi Lan, Rohan Chandra, Zuxuan Wu, Larry Davis, Dinesh Manocha


## Abstract
<div style="text-align: justify">We present a novel architecture for 3D object detection, M3DETR, which combines different point cloud representations (raw, voxels, bird-eye view) with different feature scales based on multi-scale feature pyramids. M3DETR is the first approach that unifies multiple point cloud representations, feature scales, as well as models mutual relationships between point clouds simultaneously using transformers. We perform extensive ablation experiments that highlight the benefits of fusing representation and scale, and modeling the relationships. Our method achieves state-of-the-art performance on the KITTI 3D object detection dataset and Waymo Open Dataset. Results show that M3DETR improves the baseline significantly by 1.48% mAP for all classes on Waymo Open Dataset. In particular, our approach ranks 1st on the well-known KITTI 3D Detection Benchmark for both car and cyclist classes, and ranks 1st on Waymo Open Dataset with single frame point cloud input. </div>

<p>&nbsp;</p>

<img src="https://obj.umiacs.umd.edu/acmmm2021/coverpic-1.png" width="600">


### Features
* A unified architecture for 3D object detection with transformers that accounts for multi-representation, multi-scale, mutual-relation models of point clouds in an end-to-end manner.
* Support major 3D object detection datasets: Waymo Open Dataset, KITTI.

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Getting Started with M3DETR](GETTING_STARTED.md).


## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [M3DETR Model Zoo](MODEL_ZOO.md).


## Citation
Please cite our work if you found it useful,

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

# License

This project is released under the [Apache 2.0 license](LICENSE).

# Acknowledgement

The source code of M3DETR is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 
