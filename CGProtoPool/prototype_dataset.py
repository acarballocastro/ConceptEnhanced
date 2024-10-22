"""
Calculate the concept prototype dataset given a trained model.
"""
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2
import matplotlib.pyplot as plt

import re
import os

from CGProtoPool.utils.utils import makedir, preprocess_input_function
import model
import CGProtoPool.utils.find_nearest as find_nearest
# import train_and_test as tnt

import argparse
from types import SimpleNamespace
from datasets.CUB_dataset import get_CUB_dataloaders
from datasets.AwA2_dataset import get_AwA_dataloaders
from model import PrototypeChooser

config = SimpleNamespace(
    seed = 25,

    data_path = '.',
    train_path = '.',
    test_path = '.',

    batch_size=80,
    img_size=224,

    attr_index = None,
    img_aug = False,

    num_descriptive=10, 
    num_prototypes=400,
    num_classes=85,

    use_thresh=True,
    arch='resnet50',
    add_on_layers_type='log',
    prototype_activation_function='log',
    proto_depth=256,
    pretrained=True,
    last_layer=True,
    inat=False,

    data_type='awa'
)

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs=1, type=str, default='0')
parser.add_argument('--modeldir', nargs=1, type=str)
parser.add_argument('--model', nargs=1, type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
load_model_dir = args.modeldir[0]
load_model_name = args.model[0]

load_model_path = os.path.join(load_model_dir, load_model_name)

# load the model
print('load model from ' + load_model_path)
model = PrototypeChooser(
        num_prototypes=config.num_prototypes,
        num_descriptive=config.num_descriptive,
        num_classes=config.num_classes,
        use_thresh=config.use_thresh,
        arch=config.arch,
        pretrained=config.pretrained,
        add_on_layers_type=config.add_on_layers_type,
        prototype_activation_function=config.prototype_activation_function,
        proto_depth=config.proto_depth,
        use_last_layer=config.last_layer,
        inat=config.inat,
    )

checkpoint = torch.load(load_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
model_multi = torch.nn.DataParallel(model)

# prepare dataloaders
if config.data_type == 'birds':
    train_loader, val_loader, test_loader, train_push_loader = get_CUB_dataloaders(args)
elif config.data_type == 'awa':
    train_loader, val_loader, test_loader, train_push_loader = get_AwA_dataloaders(
        classes_file=os.path.join(config.data_path, 'Animals_with_Attributes2/classes.txt'),
        data_path=config.data_path, 
        batch_size=config.batch_size, 
        preload=True
    )

# prepare directories
root_dir_for_saving_train_images = os.path.join(load_model_dir, 'nearest_train')
root_dir_for_saving_test_images = os.path.join(load_model_dir, 'nearest_test')
makedir(root_dir_for_saving_train_images)
makedir(root_dir_for_saving_test_images)

load_img_dir = os.path.join(load_model_dir, 'push') # 'img_proto'
prototype_info = np.load(os.path.join(load_img_dir, 'bb.npy'))

def save_prototype_original_img_with_bbox(fname, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(fname, p_img_rgb)

print('Creating directories...')
assigned_prototypes = []
for j in range(model_multi.module.num_prototypes):
    if os.path.exists(os.path.join(load_img_dir, 'prototype-img-original'+str(j)+'.png')):
        assigned_prototypes += [j]
        makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
        makedir(os.path.join(root_dir_for_saving_test_images, str(j)))
        save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_train_images, str(j),
                                                                'prototype_in_original_pimg.png'),
                                            index=j,
                                            bbox_height_start=prototype_info[j][1],
                                            bbox_height_end=prototype_info[j][2],
                                            bbox_width_start=prototype_info[j][3],
                                            bbox_width_end=prototype_info[j][4],
                                            color=(0, 255, 255))
        save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_test_images, str(j),
                                                                'prototype_in_original_pimg.png'),
                                            index=j,
                                            bbox_height_start=prototype_info[j][1],
                                            bbox_height_end=prototype_info[j][2],
                                            bbox_width_start=prototype_info[j][3],
                                            bbox_width_end=prototype_info[j][4],
                                            color=(0, 255, 255))
print('Directories ready')

k = 49

find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network=model, 
        prototype_network_parallel=model_multi, # pytorch network with prototype_vectors
        assigned_prototypes=assigned_prototypes,
        k=k+1,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        log=print)

find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network=model, 
        prototype_network_parallel=model_multi, # pytorch network with prototype_vectors
        assigned_prototypes=assigned_prototypes,
        k=k,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_test_images,
        log=print)