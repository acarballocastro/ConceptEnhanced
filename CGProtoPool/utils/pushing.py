"""
Perform pushing of prototypes for an already trained model.
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
from datasets.AwA2_dataset import get_AwA_dataloaders2
from model import PrototypeChooser

from fastprogress import progress_bar

from CGProtoPool.utils.utils import mixup_data, find_high_activation_crop
import os
import matplotlib.pyplot as plt
import cv2

from CGProtoPool.utils.utils import mixup_data, compute_proto_layer_rf_info_v2, compute_rf_prototype

from datasets.CUB_dataset import get_CUB_dataloaders
from datasets.AwA2_dataset import get_AwA_dataloaders
from fastprogress import progress_bar
import wandb
import pickle
from types import SimpleNamespace

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

# load pre-trained model
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

# Data loaders
# images should not be transformed (img_aug=False)
if config.data_type == 'birds':
        train_loader, val_loader, test_loader, train_push_loader = get_CUB_dataloaders(args)
elif config.data_type == 'awa':
    train_loader, val_loader, test_loader, train_push_loader = get_AwA_dataloaders2(
        classes_file=os.path.join(config.data_path, 'Animals_with_Attributes2/classes.txt'),
        data_path=config.data_path, 
        batch_size=config.batch_size, 
        preload=True,
        img_aug=config.img_aug,
    )
else:
    raise ValueError

# Prototype pushing
def update_prototypes_on_batch(search_batch_input, start_index_of_search_batch,
                               model,
                               global_min_proto_dist,  # this will be updated
                               global_min_fmap_patches,  # this will be updated
                               proto_rf_boxes,  # this will be updated
                               proto_bound_boxes,  # this will be updated
                               class_specific=True,
                               search_y=None,  # required if class_specific == True
                               num_classes=None,  # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None
                               ):
    model.eval()
    search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        proto_dist_torch = model.prototype_distances(search_batch)
        protoL_input_torch = model.conv_features(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    prototype_shape = model.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    if class_specific:
        map_class_to_prototypes = model.get_map_class_to_prototypes()

        # Getting prototype to concept class dictionary
        num_elements = map_class_to_prototypes.shape[1]
        class_indices = np.arange(num_classes*2).reshape(-1, 1)
        repeated_class_indices = np.repeat(class_indices, num_elements, axis=1)

        flat_elements = map_class_to_prototypes.flatten()
        flat_class_indices = repeated_class_indices.flatten()

        prototype_to_concept_dict = {i: [] for i in range(n_prototypes)}
        for element, class_idx in zip(flat_elements, flat_class_indices):
            prototype_to_concept_dict[element].append(class_idx)

        if dir_for_saving_prototypes is not None:
            with open(dir_for_saving_prototypes.rsplit('/', 1)[0] + '/prototype_to_concept_dict.pkl', 'wb') as f:
                pickle.dump(prototype_to_concept_dict, f)

        # Getting prototype to image dictionary
        prototype_to_img_index_dict = {key: [] for key in range(n_prototypes)}

        for img_index, img_y in enumerate(progress_bar(search_y, leave=False)):
            for concept_index, concept in enumerate(img_y):
                if concept.item() == 1:
                    [prototype_to_img_index_dict[prototype].append(img_index) for prototype in map_class_to_prototypes[concept_index]]
                else:
                    [prototype_to_img_index_dict[prototype].append(img_index) for prototype in map_class_to_prototypes[concept_index+num_classes]]

        # Since an image has n_classes concepts and some of these concepts can have been assigned to the same prototype,
        # it is possible that there are repeated image ids for a given prototype in the dictionary
        # Here we remove repeated indexes:
        prototype_to_img_index_dict = {key: sorted(set(value)) for key, value in prototype_to_img_index_dict.items()}

    for j in range(n_prototypes):
        if class_specific:
            # target_class is the class of the class_specific prototype
            # if there is not images of the target_class from this batch we go on to the next prototype
            if len(prototype_to_img_index_dict[j]) == 0:
                continue
            proto_dist_j = proto_dist_[prototype_to_img_index_dict[j]][:, j]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j]

        batch_min_proto_dist_j = np.amin(proto_dist_j)

        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''

                batch_argmin_proto_dist_j[0] = prototype_to_img_index_dict[j][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * \
                prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * \
                prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

           # get the receptive field boundary of the image patch
            # that generates the representation
            # protoL_rf_info = model.proto_layer_rf_info
            layer_filter_sizes, layer_strides, layer_paddings = model.features.conv_info()
            protoL_rf_info = compute_proto_layer_rf_info_v2(224, layer_filter_sizes, layer_strides, layer_paddings,
                                           prototype_kernel_size=1)
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            original_img_j = (original_img_j - np.min(original_img_j)) / np.max(original_img_j - np.min(original_img_j))
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if model.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + model.epsilon))
            elif model.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    makedir(os.path.join(dir_for_saving_prototypes, 'push/'))
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         'push/' + prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes, 
                                            'push/' + prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            'push/' + prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                'push/' + prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                'push/' + prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            'push/' + prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)

global_min_proto_dist = np.full(model_multi.module.num_prototypes, np.inf)
global_min_fmap_patches = np.zeros(
    [model_multi.module.num_prototypes,
        model_multi.module.prototype_shape[1],
        model_multi.module.prototype_shape[2],
        model_multi.module.prototype_shape[3]])

proto_rf_boxes = np.full(shape=[model.num_prototypes, 7],
                            fill_value=-1)
proto_bound_boxes = np.full(shape=[model.num_prototypes, 7],
                                    fill_value=-1)

search_batch_size = train_push_loader.batch_size     

for push_iter, (search_batch_input, search_y) in enumerate(progress_bar(train_push_loader, leave=False)):
    '''
    start_index_of_search keeps track of the index of the image
    assigned to serve as prototype
    '''

    start_index_of_search_batch = push_iter * search_batch_size

    update_prototypes_on_batch(search_batch_input=search_batch_input, 
                                start_index_of_search_batch=start_index_of_search_batch,
                                model=model_multi.module,
                                global_min_proto_dist=global_min_proto_dist,
                                global_min_fmap_patches=global_min_fmap_patches,
                                proto_rf_boxes=proto_rf_boxes,
                                proto_bound_boxes=proto_bound_boxes,
                                class_specific=True,
                                search_y=search_y,
                                num_classes=model.num_classes,
                                prototype_layer_stride=1,
                                dir_for_saving_prototypes=load_model_dir, #proto_img_dir,
                                prototype_img_filename_prefix='prototype-img',
                                prototype_self_act_filename_prefix='prototype-self-act',
                                prototype_activation_function_in_numpy=None)

np.save(os.path.join(load_model_dir, 'push/' + 'bb-receptive_field.npy'), proto_rf_boxes)
np.save(os.path.join(load_model_dir, 'push/' + 'bb.npy'), proto_bound_boxes)