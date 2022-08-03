# UP-ViT
This is an official implementation for "[A Unified Pruning Framework for Vision Transformers](https://arxiv.org/pdf/2111.15127.pdf)".


## Main Results on ImageNet-1K with Pretrained Models
ImageNet-1K Pretrained UP-DeiT Models

| Model  | Top-1 |  #Param.(M) |  Throughputs (img/s) |
| ------------- | ------------- |    ------------- |  ------------- | 
| [UP-DeiT-T](https://github.com/yuhao318/UP-ViT/releases/download/v1.0.0/UP_deit_tiny_patch16_224.pth)   | 75.94% | 5.7 |  1408.5 |
| [UP-DeiT-S](https://github.com/yuhao318/UP-ViT/releases/download/v1.0.0/UP_deit_small_patch16_224.pth)  | 81.56% | 	22.1 |  603.1 |

ImageNet-1K Pretrained UP-PVTv2 Models

| Model  | Top-1 |  #Param.(M) |  Throughputs (img/s) |
| ------------- | ------------- |    ------------- |  ------------- | 
| [UP-PVTv2-B0](https://github.com/yuhao318/UP-ViT/releases/download/v1.0.0/UP_pvt_v2_b0.pth)   | 75.30% | 3.67 |  139.9 |
| [UP-PVTv2-B1](https://github.com/yuhao318/UP-ViT/releases/download/v1.0.0/UP_pvt_v2_b1.pth)  | 79.48% | 	 14.01 |  249.9 |

**Note: UP-DeiT and UP-PVTv2 have the same architecture as the original DeiT and PVTv2, but with higher accuracy. See [our paper](https://arxiv.org/pdf/2111.15127.pdf) for more results.**


## Citation

```
@article{yu2021unified,
  title={A unified pruning framework for vision transformers},
  author={Yu, Hao and Wu, Jianxin},
  journal={arXiv preprint arXiv:2111.15127},
  year={2021}
}
```


## Contacts
If you have any question about our work, please do not hesitate to contact us by emails provided in the [paper](https://arxiv.org/pdf/2111.15127.pdf).

