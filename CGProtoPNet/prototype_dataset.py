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

from CGProtoPNet.utils.helpers import makedir
import model
import CGProtoPNet.utils.find_nearest as find_nearest
import train_and_test as tnt

from CGProtoPNet.utils.preprocess import preprocess_input_function
import argparse
from types import SimpleNamespace
from datasets.CUB_dataset import get_CUB_dataloaders
from datasets.AwA2_dataset import get_AwA_dataloaders

config = SimpleNamespace(
    seed = 25,
    data_path = '.',
    train_path = '.',
    test_path = '.',
    val_path = '.',

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

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs=1, type=str, default='0')
parser.add_argument('--modeldir', nargs=1, type=str)
parser.add_argument('--model', nargs=1, type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
load_model_dir = args.modeldir[0]
load_model_name = args.model[0]

load_model_path = os.path.join(load_model_dir, load_model_name)
start_epoch_number = 50 

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

# load the data
# must use unaugmented (original) dataset
train_loader, test_loader, _, train_push_loader = get_AwA_dataloaders(
        classes_file=os.path.join(config.data_path, 'Animals_with_Attributes2/classes.txt'),
        data_path=config.data_path, 
        train_batch_size=config.train_batch_size, 
        test_batch_size=config.test_batch_size, 
        train_push_batch_size=config.train_push_batch_size,
        preload=True,
        img_aug=config.img_aug
    )

# prepare for saving
root_dir_for_saving_train_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_train')
root_dir_for_saving_test_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_test')
makedir(root_dir_for_saving_train_images)
makedir(root_dir_for_saving_test_images)

# save prototypes in original images
load_img_dir = os.path.join(load_model_dir, 'img')
prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(start_epoch_number), 'bb'+str(start_epoch_number)+'.npy'))
def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    plt.imsave(fname, p_img_rgb)

print('Creating directories...')
for j in range(ppnet.num_prototypes):
    makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
    makedir(os.path.join(root_dir_for_saving_test_images, str(j)))
    save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_train_images, str(j),
                                                             'prototype_in_original_pimg.png'),
                                          epoch=start_epoch_number,
                                          index=j,
                                          bbox_height_start=prototype_info[j][1],
                                          bbox_height_end=prototype_info[j][2],
                                          bbox_width_start=prototype_info[j][3],
                                          bbox_width_end=prototype_info[j][4],
                                          color=(0, 255, 255))
    save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_test_images, str(j),
                                                             'prototype_in_original_pimg.png'),
                                          epoch=start_epoch_number,
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
        prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
        k=k+1,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        log=print)

find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
        k=k,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_test_images,
        log=print)
