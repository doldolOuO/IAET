# Multi-Modal Point Cloud Completion with Interleaved Attention Enhanced Transformer (IJCAI 2025)

## Datasets
Use the code in  ``dataloader.py`` to load the dataset. 

### ShapeNet-ViPC
First, please download the [ShapeNetViPC-Dataset](https://pan.baidu.com/s/1NJKPiOsfRsDfYDU_5MH28A) (143GB, code: **ar8l**). Then run ``cat ShapeNetViPC-Dataset.tar.gz* | tar zx``, you will get ``ShapeNetViPC-Dataset`` contains three folders: ``ShapeNetViPC-Partial``, ``ShapeNetViPC-GT`` and ``ShapeNetViPC-View``. 

For each object, the dataset includes partial point clouds (``ShapeNetViPC-Patial``), complete point clouds (``ShapeNetViPC-GT``) and corresponding images (``ShapeNetViPC-View``) from 24 different views. You can find the detail of 24 cameras view in ``/ShapeNetViPC-View/category/object_name/rendering/rendering_metadata.txt``.

### KITTI
The KITTI dataset used in this work is sourced from the [Cross-PCC](https://github.com/ltwu6/cross-pcc).

## Requirements
- Ubuntu: 18.04 and above
- CUDA: 11.3 and above
- PyTorch: 1.10.1 and above

## Training
The file ``config_vipc.py`` and ``config_3depn.py`` contain the configuration for all the training parameters.

To train the models in the paper, run this command:

```train
python train_vipc.py
```

```or
python train_3depn.py 
```

## Evaluation
- [ShapeNet-ViPC]()

- [KITTI]()

To evaluate the models (select the specific category in ``config_vipc.py``):

```eval
python eval_vipc.py 
```

## Pre-trained models

## Acknowledgements
Some of the code of this repo is borrowed from:

- [XMFNet](https://github.com/diegovalsesia/XMFnet)

- [Cross-PCC](https://github.com/ltwu6/cross-pcc)

- [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet)

- [ChamferDistance](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)

- [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)

## Cite this work

```
@inproceedings{fang2025iaet,
  title={Multi-Modal Point Cloud Completion with Interleaved Attention Enhanced Transformer},
  author={Chenghao Fang and Jianqing Liang and Jiye Liang and Hangkun Wang and Kaixuan Yao and Feilong Cao},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025}
}
```

## License

This project is open sourced under MIT license.
