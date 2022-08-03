'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import logging
import argparse
import os
import random
import csv
import numpy as np
import time

import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from utils.pred_utils import ProgressMeter, accuracy, AverageMeter

from models.model import VisionTransformer
from models.attn_importance_score_model import VisionTransformer as attn_VisionTransformer

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--img_size", default=256, type=int,
                    help="Resolution size")
parser.add_argument("--batch_size", default=128, type=int,
                    help="Total batch size for training.")
parser.add_argument("--block_ind", default=-1, type=int,
                    help="Total batch size for training.")

args = parser.parse_args()


use_cuda = torch.cuda.is_available()
batch_size = args.batch_size
if use_cuda:
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu

print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


testset = torchvision.datasets.ImageFolder(root='/mnt/ramdisk/ImageNet/fewshot2_train/', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=8)
print('==> Resuming from checkpoint..')
checkpoint = torch.load("pretrainmodel/deit_base_patch16_224-b5f2ef4d.pth", map_location='cpu')
teacher_model = VisionTransformer(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6))
model_dict = teacher_model.state_dict()
new_dict  = {}
cnt = 1
for k, v in checkpoint['model'].items():
    cnt += 1
    new_dict[k] = v
model_dict.update(new_dict)
teacher_model.load_state_dict(model_dict)
teacher_model.cuda()
teacher_model = torch.nn.DataParallel(teacher_model)
print("=> loaded teacher checkpoint")

checkpoint = torch.load("pretrainmodel/deit_base_patch16_224-b5f2ef4d.pth", map_location='cpu')
candidate_index = range(768)
results1 = []
results5 = []
kls = []
for delete_ind in candidate_index:
    net = attn_VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), reduce = delete_ind, ind=args.block_ind)

    net.cuda()
    net = torch.nn.DataParallel(net)

    model_dict = net.state_dict()

    new_dict  = {}
    cnt = 1
    for k, v in checkpoint['model'].items():
        # print(k,end= ", ")
        if "blocks." + str(args.block_ind) +  ".attn.qkv.bias" in k:
            interval = v.size(0) // 3
            new_index = [i not in [delete_ind,delete_ind+interval,delete_ind + 2* interval] for i in torch.arange(v.size(0))]
            new_v = v[new_index]
            # print(new_v.shape)
            new_dict["module." + k] = new_v
        elif "blocks." + str(args.block_ind) + ".attn.qkv.weight" in k:
            interval = v.size(0) // 3
            new_index = [i not in [delete_ind,delete_ind+interval,delete_ind + 2* interval] for i in torch.arange(v.size(0))]
            new_v = v[new_index,:]
            # print(new_v.shape)
            new_dict["module." + k] = new_v
        elif "blocks." + str(args.block_ind) +  ".attn.proj.weight" in k:
            new_v = v[:,torch.arange(v.size(1))!=delete_ind]
            # print(new_v.shape)
            new_dict["module." + k] = new_v   
        else:
            # print(v.shape)
            new_dict["module." + k] = v
    model_dict.update(new_dict)
    net.load_state_dict(model_dict)


    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    kl = AverageMeter('KL', ':6.3f')
    cos = AverageMeter('Cosine', ':6.3f')

    progress = ProgressMeter(
        len(testloader),
        [batch_time, losses, top1, top5, kl],
        prefix='Test: ')

    evaluate = True
    if evaluate:
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            net.eval()

            end = time.time()
            for i, (images, target) in enumerate(testloader):
                with autocast():

                    images = images.cuda( non_blocking=True)
                    target = target.cuda( non_blocking=True)

                    # compute output
                    output = net(images)
                    with torch.no_grad():
                        teacher_output, teacher_feature, teacher_patch_output = teacher_model(images)
                    logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
                    softmax = torch.nn.Softmax(dim=1).cuda()
                    distil_loss = torch.sum(
                        torch.sum(softmax(teacher_output) * (logsoftmax(teacher_output)-logsoftmax(output)), dim=1))

                    kl.add(distil_loss,images.size(0))
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()


            print(kl.sum.item())
        kls.append(kl.sum.item())

with open("importance/Deit_base_12_attn_768_kl_" +str(args.block_ind)+ "_1k.txt", 'w') as f:
    for s in kls:
        f.write(str(s) + '\n')
