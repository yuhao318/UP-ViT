# UP-DeiT implementation

This folder contains the implementation of compressing DeiT-B into UP-DeiT-T.

## Main requirements

```
python >= 3.7
torch >= 1.4.0
torchvision >= 0.5.0
```
We provide the detailed requirements in requirements.txt. You can run `conda install --yes --file requirements.txt` to create the same running environment as ours.

## Uasge

### Data preparation

We use standard ImageNet dataset. You can download it from [http://image-net.org/](http://image-net.org/). 

To generate the proxy dataset, assume the ImageNet dataset is saved on `/mnt/ramdisk`, run:
```
python cp_image.py
```

### Importance Calculation 

Download the pretrained model [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) into the `pretrainmodel` floader, then run:
```
./run_importance.sh
```
**Note: We have provided the importance of each channel of DeiT-B on the importance folder, including the proxy size of 2k and 5k.**


### Sample Sub-model 

After generating importance, run:
```
python sample_sub_model.py
```

### Fine-tune Sub-model

To train a `UP-DeiT-T` on ImageNet from scratch, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_distill.py --dist-url 'tcp://127.0.0.1:12342' --dist-backend   'nccl' --multiprocessing-distributed  --world-size 1 --rank 0  -j <worker-size>  --learning-rate <learning-rate> --name <save-name> --wd <weight-decay>   -b  <batch-size> --alpha <cumix-ratio> --epochs <epoch-number> --resume <pretrainde-sub-model> <imagenet-path> 
```