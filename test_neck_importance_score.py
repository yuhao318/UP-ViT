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
from tqdm import  tqdm
from utils.pred_utils import ProgressMeter, accuracy, AverageMeter

from models.model import VisionTransformer
from models.neck_importance_score_model import VisionTransformer as neck_VisionTransformer

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--resume_sess', default='vit_default',type=str, help='session id')
parser.add_argument("--img_size", default=256, type=int,
                    help="Resolution size")
parser.add_argument("--batch_size", default=128, type=int,
                    help="Total batch size for training.")
parser.add_argument("--delete_ind", default=-1, type=int,
                    help="The index of delete neck.")

args = parser.parse_args()


use_cuda = torch.cuda.is_available()
batch_size = args.batch_size
if use_cuda:
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu

# print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# testset = torchvision.datasets.ImageFolder(root='../dataset/ImageNet/val/', transform=transform_test)
# testset = torchvision.datasets.ImageFolder(root='/mnt/ramdisk/ImageNet/fewshot_val/', transform=transform_test)

testset = torchvision.datasets.ImageFolder(root='/mnt/ramdisk/ImageNet/fewshot2_train/', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=8)

# print('==> Resuming from checkpoint..')

checkpoint = torch.load("pretrainmodel/deit_base_patch16_224-b5f2ef4d.pth", map_location='cpu')
# for k,v in model_dict.items():
#     print(k,v.size())
teacher_model = VisionTransformer(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6))

model_dict = teacher_model.state_dict()
new_dict  = {}
cnt = 1
for k, v in checkpoint['model'].items():
    if k in model_dict and v.size()==model_dict[k].size():
        # print('update teacher cnt {} : {}'.format(cnt, k))
        cnt += 1
        new_dict[k] = v
model_dict.update(new_dict)
teacher_model.load_state_dict(model_dict)
teacher_model.cuda()
teacher_model = torch.nn.DataParallel(teacher_model)
cudnn.benchmark = True
# print("=> loaded teacher checkpoint")

# candidate_index = range(768)
results1 = []
results5 = []
kls = []
coss = []
net = neck_VisionTransformer(patch_size=16, embed_dim=767, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6))
model_dict = net.state_dict()
new_dict  = {}
cnt = 1

candidate_index = range(768)
for delete_ind in candidate_index:
    for k, v in checkpoint['model'].items():
        # print(k,end= ", ")
        if "qkv.weight" in k:
            new_v = v[:,torch.arange(v.size(1))!=delete_ind]
            # print(new_v.shape)
            new_dict[ k] = new_v
        elif "cls_token" in k or "pos_embed" in k:
            new_v = v[:,:,torch.arange(v.size(2))!=delete_ind]
            # print(new_v.shape)
            new_dict[ k] = new_v
        elif "patch_embed" in k or "norm" in k  or "fc2" in k or "attn.proj" in k:
            new_v = v[torch.arange(v.size(0))!=delete_ind]
            # print(new_v.shape)
            new_dict[ k] = new_v
        elif "head.weight" in k or "mlp.fc1.weight" in k:
            new_v = v[:,torch.arange(v.size(1))!=delete_ind]
            # print(new_v.shape)
            new_dict[ k] = new_v
        else:
            # print(v.shape)
            new_dict[ k] = v

    model_dict.update(new_dict)

    net.load_state_dict(model_dict)
    net.cuda()
    net = torch.nn.DataParallel(net)


    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    kl = AverageMeter('KL', ':6.3f')
    cos = AverageMeter('Cosine', ':6.3f')

    progress = ProgressMeter(
        len(testloader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    evaluate = True
    if evaluate:
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            net.eval()
            teacher_model.eval()
            end = time.time()
            for i, (images, target) in enumerate(testloader):
                with autocast():

                    images = images.cuda( non_blocking=True)
                    target = target.cuda( non_blocking=True)

                    # compute output
                    output = net(images)
                    # loss = criterion(output, target)
                    with torch.no_grad():
                        teacher_output, teacher_feature, teacher_patch_output = teacher_model(images)
                    # cosine_similarity = F.cosine_similarity(output, teacher_output)
                    # cosine_similarity = torch.sum(cosine_similarity)
                    # distil_loss = F.mse_loss(feature, teacher_feature)
                    logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
                    softmax = torch.nn.Softmax(dim=1).cuda()
                    distil_loss = torch.sum(
                        torch.sum(softmax(teacher_output) * (logsoftmax(teacher_output) - logsoftmax(output)), dim=1))
                    # measure accuracy and record loss
                    # acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    # losses.update(loss.item(), images.size(0))
                    # top1.update(acc1[0], images.size(0))
                    # top5.update(acc5[0], images.size(0))
                    kl.add(distil_loss,images.size(0))
                    # cos.add(cosine_similarity, images.size(0))
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                # if i % 10 == 0:
                #     progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #     .format(top1=top1, top5=top5))
        #     print(cos.avg.item())
        # coss.append(cos.avg.item())
        # results1.append(top1.avg.item())
        # results5.append(top5.avg.item())
        print(kl.sum.item())
        kls.append(kl.sum.item())
with open("importance/Deit_base_12_neck_768_kl_2k.txt", 'a') as f:
    for s in kls:
        f.write(str(s) + '\n')