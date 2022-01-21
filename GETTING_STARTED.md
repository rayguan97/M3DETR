# Getting Started
Please see [Getting Started with OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) for full usage.



## Preparing Datasets for M3DETR

### Expected dataset structure for Waymo Open Dataset
```
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
├── pcdet
├── tools
```

The dataset configs are located within [tools/cfgs/dataset_configs](https://github.com/rayguan97/M3DETR/blob/main/tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs/m3detr_models](https://github.com/rayguan97/M3DETR/blob/main/tools/cfgs/m3detr_models) for different datasets. 


### Inference with Pre-trained Models

1. Pick a configuration file and its corresponding model from
  [model zoo](MODEL_ZOO.md),
  for example, the config file of ./tools/cfgs/m3detr_models/m3detr_waymo_1500.yaml.

2. We provide `./scripts/dist_test.sh` that is able to test with a pretrained model. Run it with:
```
cd tools/
sh ./scripts/dist_test.sh 1 --cfg_file ./tools/cfgs/m3detr_models/m3detr_waymo_1500.yaml --workers 4 --ckpt /path/to/checkpoint_file --eval_tag test_out --batch_size 8 --save_to_file [--other-options]
```

### Train a Model
We provide `./scripts/dist_train.sh` that is able to train a model with the specified config file. Run it with:
```
cd tools/
sh ./scripts/dist_train.sh 1 --cfg_file ./tools/cfgs/m3detr_models/m3detr_waymo_1500.yaml --workers 4
```
For more options, see ./scripts/dist_train.sh.