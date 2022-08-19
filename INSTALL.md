## Installation

### Requirements
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.1. Please check PyTorch version matches that is required by OpenPCDet.
- OpenPCDet: follow [OpenPCDet installation instructions](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md).


### Example conda environment setup
```bash
conda create -n m3detr python=3.6 -y
conda activate m3detr
conda install pytorch=1.6.0 torchvision cudatoolkit=10.2 -c pytorch
pip install spconv-cu102
pip install tqdm scipy scikit-image
git clone https://github.com/rayguan97/M3DETR.git
cd M3DETR
python setup.py develop
```
