# COCO Object Detection and Instance Segmentation with SHViT

The codebase implements the object detection and instance segmentation framework with [MMDetection](https://github.com/open-mmlab/mmdetection), using SHViT as the backbone.

## Results and Models

### RetinaNet Object Detection
|Model | Pretrain | Lr Schd | Box AP | AP@50 | AP@75 | Config | Model | 
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|SHViT-S4 | ImageNet-1k   | 1x | 38.8  | 59.8   | 41.1  | [config](./configs/retinanet_shvit_s4_fpn_1x_coco.py) | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/retinanet_shvit_s4_1x_coco.pth) |


### Mask R-CNN Instance Segmentation
|Model | Pretrain | Lr Schd | Box AP | Mask AP | Config | Model | 
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|SHViT-S4 | ImageNet-1k   |1x| 39.0 | 35.9 | [config](./configs/mask_rcnn_shvit_s4_fpn_1x_coco.py) | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/maskrcnn_shvit_s4_1x_coco.pth) |

## Setup

After setting up dependency for [Image Classification](https://github.com/ysj9909/SHViT), install the following packages
```
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet==2.25.0
```

## Data preparation

Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).
The dataset should be organized as 
```
downstream
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Evaluation

Below are the instructions for evaluating the models on MSCOCO 2017 val set:

<details>

<summary>
RetinaNet Object Detection
</summary>

To evaluate the RetinaNet model with SHViT-S4 as backbone, run:

```bash
bash ./dist_test.sh configs/retinanet_shvit_s4_fpn_1x_coco.py ./retinanet_shvit_s4_1x_coco.pth 8 --eval bbox
```
</details>

<details>

<summary>
Mask R-CNN Instance Segmentation
</summary>

To evaluate the Mask R-CNN model with SHViT-S4 as backbone, run:

```bash
bash ./dist_test.sh configs/mask_rcnn_shvit_s4_fpn_1x_coco.py ./maskrcnn_shvit_s4_1x_coco.pth 8 --eval bbox segm
```

</details>


## Training

Below are the instructions for training the models on MSCOCO 2017 train set:

<details>

<summary>
RetinaNet Object Detection
</summary>

To train the RetinaNet model with SHViT-S4 as backbone on a single machine using multi-GPUs, run:

```bash
bash ./dist_train.sh configs/retinanet_shvit_s4_fpn_1x_coco.py 8 --cfg-options model.backbone.pretrained=$PATH_TO_IMGNET_PRETRAIN_MODEL
```

</details>


<details>

<summary>
Mask R-CNN Instance Segmentation
</summary>

To train the Mask R-CNN model with SHViT-S4 as backbone on a single machine using multi-GPUs, run:

```bash
bash ./dist_train.sh configs/mask_rcnn_shvit_s4_fpn_1x_coco.py 8 --cfg-options model.backbone.pretrained=$PATH_TO_IMGNET_PRETRAIN_MODEL
```


</details>


## Citation
If our work or code help your work, please cite our paper:
```
@inproceedings{yun2024shvit,
  author = {Yun, Seokju and Ro, Youngmin},
  title = {SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024}
}
```

## Acknowledge

The downstream task implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[MMDetection](https://github.com/open-mmlab/mmdetection), [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), [PoolFormer](https://github.com/sail-sg/poolformer/tree/main/detection), [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT).
