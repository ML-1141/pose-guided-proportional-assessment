
# Pose-Guided Proportional Structure Assessment for 2D Illustrated Characters


## Requirements

To install requirements:

```setup
pip install torch
pip install torch vision
pip install opencv-python
pip install grad-cam
pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
```

## Dataset

Download the datasets from [\_dataset_](https://drive.google.com/drive/folders/1bXjlnU0Q90I6w5YoS63aVochboCR5jVW?usp=drive_link) and unzip them.

## Training

To train the model(s) in the paper, run this command:

```train
python train.py <root_dir> <save_path> <out_root> --use_stn true # for STN-integrated method
python train.py <root_dir> <save_path> <out_root> --use_stn false # for deterministic method
```
 * <root_dir>: path to dataset root
 * <save_path>: path to save checkpoints
 * <out_root>: path to save inference results

>📋  The inference will be done automatically right after training.
