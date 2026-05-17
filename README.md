# CMVPR
## Introduction
CMVPR：一种跨模态领域自适应的视觉位置识别方法。
尽管3D感知在性能与鲁棒性上通常优于2D感知，但3D传感器的引入也给需要轻量级操作的系统带来了额外的挑战。
为平衡这一矛盾，本文采用知识蒸馏策略，通过多种具有不同空间曲率的流形探索更丰富的特征关系，增强了特征关系的多样性。使仅依赖相机的学生模型能够从使用激光雷达的教师模型中学习。 
同时，借助对抗训练策略，有效缩小了源域与目标域之间的特征分布差异，实现从激光雷达到相机的知识迁移。
## Install

- Python  3.10(ubuntu22.04)
- CUDA  12.1
```

cd work3
conda create -n CMVPR python=3.8
conda activate CMVPR
pip install -r requirements.txt
```
## Data Preparation
- Please download the offical [nuScenes dataset](https://www.nuscenes.org/nuscenes).
- Generate the infos and range data needed to run the code.
```bash
cd tools
python gen_info.py
python gen_index.py
python gen_range.py
cd ..
```
- The final data structure should be like:
```
nuScenes
├─ samples
│    ├─ CAM_BACK
│    ├─ CAM_BACK_LEFT
│    ├─ CAM_BACK_RIGHT
│    ├─ CAM_FRONT
│    ├─ CAM_FRONT_LEFT
│    ├─ CAM_FRONT_RIGHT
│    ├─ LIDAR_TOP
│    ├─ RANGE_DATA
├─ sweeps
│    ├─ ...
├─ maps
│    ├─ ...
├─ v1.0-test
│    ├─ attribute.json
│    ├─ calibrated_sensor.json
│    ├─ ...
├─ v1.0-trainval
│    ├─ attribute.json
│    ├─ calibrated_sensor.json
│    ├─ ...
├─ nuscenes_infos-bs.pkl
├─ nuscenes_infos-shv.pkl
├─ nuscenes_infos-son.pkl
├─ nuscenes_infos-sq.pkl
├─ bs_db.npy
├─ bs_test_query.npy
├─ bs_train_query.npy
├─ bs_val_query.npy
├─ shv_db.npy
├─ shv_query.npy
├─ son_db.npy
├─ son_query.npy
├─ sq_db.npy
├─ sq_test_query.npy
└─ sq_train_query.npy
```

## Training
First you need to set the file paths in `config/config.yaml`. Then, run the following script to train the model:
```bash
python train.py
```

## Evaluation
Set the model path that you need to load in `test.py`. Then run the script:
```bash
python test.py
```
