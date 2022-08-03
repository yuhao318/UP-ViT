# UP-ViT
This is an official implementation for "[A Unified Pruning Framework for Vision Transformers](https://arxiv.org/pdf/2111.15127.pdf)".

## Getting Started
For UP-DeiT on the image classification task, please see get_started.md for detailed instructions.

## Main Results on ImageNet-1K with Pretrained Models
ImageNet-1K Pretrained UP-DeiT Models

| Model  | Top-1 |  #Param. |  Throughputs |
| ------------- | ------------- |    ------------- |  ------------- | 
| [UP-DeiT-T](https://github.com/yuhao318/UP-ViT/releases/download/v1.0.0/UP_deit_tiny_patch16_224.pth)   | 75.94% | 5.7M |  1408.5  |
| [UP-DeiT-S](https://github.com/yuhao318/UP-ViT/releases/download/v1.0.0/UP_deit_small_patch16_224.pth)  | 81.56% | 	22.1M |  603.1 |

ImageNet-1K Pretrained UP-PVTv2 Models

| Model  | Top-1 |  #Param. |  Throughputs |
| ------------- | ------------- |    ------------- |  ------------- | 
| [UP-PVTv2-B0](https://github.com/yuhao318/UP-ViT/releases/download/v1.0.0/UP_pvt_v2_b0.pth)   | 75.30% | 3.7M |  139.9 |
| [UP-PVTv2-B1](https://github.com/yuhao318/UP-ViT/releases/download/v1.0.0/UP_pvt_v2_b1.pth)  | 79.48% | 	 14.0M |  249.9 |

**Note: Test throughput on a Titan XP GPU with a fixed 32 mini-batch size.**

**Note: UP-DeiT and UP-PVTv2 have the same architecture as the original DeiT and PVTv2, but with higher accuracy. See [our paper](https://arxiv.org/pdf/2111.15127.pdf) for more results.**

## Main Results on WikiText-103 with Pretrained Models

[WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) Pretrained Neural Language Modeling Model with [Adaptive Inputs](https://arxiv.org/abs/1809.10853). Our model is based on [Fairseq](https://github.com/facebookresearch/fairseq/tree/v0.10.2).


| Model  | Perplexity |  #Param. |   
| ------------- | ------------- |   ------------- | 
| [Original Model](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2)   | 19.00 | 291M |
| [UP-Transformer](https://drive.google.com/file/d/1HhxpJYvcxer7iCVfS2cJYyK-n58ymOBT/view?usp=sharing)   | 19.88 | 95M |


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

