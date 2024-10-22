"""
Perform pushing of prototypes for an already trained model.
"""

import os
import shutil
import copy

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re
from types import SimpleNamespace
import logging

from CGProtoPNet.utils.helpers import makedir
import model
import CGProtoPNet.utils.push as push
import prune
import train_and_test as tnt
import CGProtoPNet.utils.save as save
# from log import create_logger
from CGProtoPNet.utils.preprocess import preprocess_input_function, mean, std 

from datasets.CUB_dataset import get_CUB_dataloaders
from datasets.AwA2_dataset import get_AwA_dataloaders
from fastprogress import progress_bar
import wandb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import CGProtoPNet.utils.find_nearest as find_nearest

config = SimpleNamespace(
    seed = 25,

    data_path = '.',
    train_path = '.',
    test_path = '.',

    train_batch_size = 80,
    test_batch_size = 100,
    train_push_batch_size = 75,

    attr_index = None,
    img_aug = False,

    base_architecture = 'vgg16',
    img_size = 224,
    prototype_shape = (1700, 128, 1, 1), 
    num_classes = 85,
    prototype_activation_function = 'log',
    add_on_layers_type = 'regular',
)

# Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs=1, type=str, default='0')
parser.add_argument('--modeldir', nargs=1, type=str)
parser.add_argument('--model', nargs=1, type=str)
#parser.add_argument('-dataset', nargs=1, type=str, default='cub200')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
load_model_dir = args.modeldir[0]
load_model_name = args.model[0]

load_model_path = os.path.join(load_model_dir, load_model_name)
img_dir = os.path.join(load_model_dir, 'img')
# load the model 
print('load model from ' + load_model_path)

ppnet = model.construct_PPNet(base_architecture=config.base_architecture,
                        pretrained=True, img_size=config.img_size,
                        prototype_shape=config.prototype_shape,
                        num_classes=config.num_classes,
                        prototype_activation_function=config.prototype_activation_function,
                        add_on_layers_type=config.add_on_layers_type)

checkpoint = torch.load(load_model_path)
ppnet.load_state_dict(checkpoint['model_state_dict'])
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet_multi.module.img_size
config.img_size = img_size

train_loader, val_loader, test_loader, train_push_loader = get_AwA_dataloaders(
        classes_file=os.path.join(config.data_path, 'Animals_with_Attributes2/classes.txt'),
        data_path=config.data_path, 
        train_batch_size=config.train_batch_size, 
        test_batch_size=config.test_batch_size, 
        train_push_batch_size=config.train_push_batch_size,
        preload=True
    )

prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=True,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir, #img_dir, # if not None, prototypes will be saved here
                epoch_number=50, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True)