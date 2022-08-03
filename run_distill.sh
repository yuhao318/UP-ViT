#! /bin/sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_distill.py --dist-url 'tcp://127.0.0.1:12342' --dist-backend   'nccl' --multiprocessing-distributed  --world-size 1 --rank 0  -j=64  --learning-rate 1e-3 --name Deit_tiny_distill --wd 1e-3    -b 256 --alpha 0.5  /mnt/ramdisk/ImageNet/ --epochs 200