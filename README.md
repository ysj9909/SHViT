# SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design

This is the official repository of 

[**SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design**](https://arxiv.org/abs/2401.16456)
*Seokju Yun, Youngmin Ro.* CVPR 2024

![SHViT Performance](acc_vs_thro.png)

## Setup
```bash
conda create -n shvit python=3.9
conda activate shvit
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```


## Citation

```
@inproceedings{yun2024shvit,
  author = {Yun, Seokju and Ro, Youngmin},
  title = {SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024}
}
```
