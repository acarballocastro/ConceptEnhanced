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

from CGProtoPNet.utils.helpers import makedir, save_model_w_condition
import model
import CGProtoPNet.utils.push as push
import train_and_test as tnt
# from log import create_logger
from CGProtoPNet.utils.preprocess import preprocess_input_function, mean, std 

from datasets.CUB_dataset import get_CUB_dataloaders
from datasets.AwA2_dataset import get_AwA_dataloaders
from fastprogress import progress_bar
import wandb

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%d-%m %I:%M:%S")

# Config parameters
config = SimpleNamespace(
    seed = 25,

    base_architecture = 'vgg19',
    img_size = 224,
    prototype_shape = (1700, 512, 1, 1), # 10 concepts per class for AwA2
    # prototype_shape = (2240, 128, 1, 1), # 10 concepts per class for CUB
    num_classes = 85, #112 for CUB
    prototype_activation_function = 'log',
    add_on_layers_type = 'regular',

    run_name = 'test',
    data_path = '.',
    train_path = '.',
    test_path = '.',
    val_path = '.',

    train_batch_size = 80,
    test_batch_size = 100,
    train_push_batch_size = 75,

    joint_optimizer_lrs_features = 1e-4,
    joint_optimizer_lrs_add_on_layers = 3e-3,
    joint_optimizer_prototype_vectors = 3e-3,
    joint_lr_step_size = 5,

    warm_optimizer_lrs_add_on_layers = 3e-3,
    warm_optimizer_prototype_vectors = 3e-3,

    last_layer_optimizer_lr = 1e-4,

    coefs_crs_ent = 1,
    coefs_clst = 0.8,
    coefs_sep = -0.08,
    coefs_l1 = 1e-4,

    num_train_epochs = 50,
    num_warm_epochs = 5,

    push_start = 10, # every 10 epochs, every 2 times we reduce learning rate (joint_lr_step_size = 5)
    push_epochs = None,

    gpuid = 0,
    attr_index = None,
    img_aug = True
)

def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')

    parser.add_argument('--base_architecture', type=str, default=config.base_architecture, help='base architecture: vgg, resnet, densenet + number')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--prototype_shape', type=int, default=config.prototype_shape, help='protoype shape')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--prototype_activation_function', type=str, default=config.prototype_activation_function, help='activation function')
    parser.add_argument('--add_on_layers_type', type=str, default=config.add_on_layers_type, help='add on layers type')

    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--data_path', type=str, default=config.data_path, help='data path')
    parser.add_argument('--train_path', type=str, default=config.train_path, help='train directory')
    parser.add_argument('--test_path', type=str, default=config.test_path, help='test directory')

    parser.add_argument('--train_batch_size', type=int, default=config.train_batch_size, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, default=config.test_batch_size, help='test batch size')
    parser.add_argument('--train_push_batch_size', type=int, default=config.train_push_batch_size, help='train push batch size')

    parser.add_argument('--joint_optimizer_lrs_features', type=float, default=config.joint_optimizer_lrs_features, help='joint stage: learning rate pretrained layers')
    parser.add_argument('--joint_optimizer_lrs_add_on_layers', type=float, default=config.joint_optimizer_lrs_add_on_layers, help='joint stage: learning rate add-on convolutional layers')
    parser.add_argument('--joint_optimizer_prototype_vectors', type=float, default=config.joint_optimizer_prototype_vectors, help='joint stage: learning rate prototype layer')
    parser.add_argument('--joint_lr_step_size', type=int, default=config.joint_lr_step_size, help='joint optimizer step size')

    parser.add_argument('--warm_optimizer_lrs_add_on_layers', type=float, default=config.warm_optimizer_lrs_add_on_layers, help='warm-up stage: learning rate add-on convolutional layers')
    parser.add_argument('--warm_optimizer_prototype_vectors', type=float, default=config.warm_optimizer_prototype_vectors, help='warm-up stage: learning rate prototype layer')
    parser.add_argument('--last_layer_optimizer_lr', type=float, default=config.last_layer_optimizer_lr, help='last layer learning rate')

    parser.add_argument('--coefs_crs_ent', type=float, default=config.coefs_crs_ent, help='coefficients: cross entropy')
    parser.add_argument('--coefs_clst', type=float, default=config.coefs_clst, help='coefficients: cluster cost')
    parser.add_argument('--coefs_sep', type=float, default=config.coefs_sep, help='coefficients: separation cost')
    parser.add_argument('--coefs_l1', type=float, default=config.coefs_l1, help='coefficients: l1 regularization term')

    parser.add_argument('--num_train_epochs', type=int, default=config.num_train_epochs, help='number of training epochs')
    parser.add_argument('--num_warm_epochs', type=int, default=config.num_warm_epochs, help='number of warm epochs')
 
    parser.add_argument('--push_start', type=int, default=config.push_start, help='push start')
    parser.add_argument('--push_epochs', type=int, default=config.push_epochs, help='push epochs')

    parser.add_argument('-gpuid', nargs=1, type=str, default='0', help='GPU device ID(s) you want to use')
    parser.add_argument('--attr_index', type=int, default=config.attr_index, help='concept to use as label')

    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

def train(config):
    """
    Train and test routine
    """

    # Setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Additional info when using cuda
    if device.type == "cuda":
        logging.info(f"Using {torch.cuda.get_device_name(0)}")
    else:
        logging.info("No GPU available")

    # Configuring directories
    model_dir = config.base_architecture + '/' + config.run_name + '_seed' + str(config.seed) + '/'
    makedir(model_dir)
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # ---------------------------------
    #       Prepare parameters
    # ---------------------------------

    config.push_epochs = [i for i in range(config.num_train_epochs) if i % 10 == 0] # stores every 10 epochs (starting from epoch push_start)

    config.joint_optimizer_lrs = {'features': config.joint_optimizer_lrs_features,
                                'add_on_layers': config.joint_optimizer_lrs_add_on_layers,
                                'prototype_vectors': config.joint_optimizer_prototype_vectors}

    config.warm_optimizer_lrs = {'add_on_layers': config.warm_optimizer_lrs_add_on_layers,
                                'prototype_vectors': config.warm_optimizer_prototype_vectors}

    config.coefs = {
        'crs_ent': config.coefs_crs_ent,
        'clst': config.coefs_clst,
        'sep': config.coefs_sep,
        'l1': config.coefs_l1,
    }

    # ---------------------------------
    #       Prepare data
    # ---------------------------------

    # train_loader, test_loader, val_loader, train_push_loader = get_CUB_dataloaders(config)
    train_loader, test_loader, val_loader, train_push_loader = get_AwA_dataloaders(
        classes_file=os.path.join(config.data_path, 'Animals_with_Attributes2/classes.txt'),
        data_path=config.data_path, 
        train_batch_size=config.train_batch_size, 
        test_batch_size=config.test_batch_size, 
        train_push_batch_size=config.train_push_batch_size,
        preload=True
    )

    logging.info('training set size: {0}'.format(len(train_loader.dataset)))
    logging.info('push set size: {0}'.format(len(train_push_loader.dataset)))
    logging.info('validation set size: {0}'.format(len(val_loader.dataset)))
    logging.info('test set size: {0}'.format(len(test_loader.dataset)))
    logging.info('batch size: {0}'.format(config.train_batch_size))

    # ---------------------------------
    #       Construct the model
    # ---------------------------------

    ppnet = model.construct_PPNet(base_architecture=config.base_architecture,
                              pretrained=True, img_size=config.img_size,
                              prototype_shape=config.prototype_shape,
                              num_classes=config.num_classes,
                              prototype_activation_function=config.prototype_activation_function,
                              add_on_layers_type=config.add_on_layers_type)

    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # ---------------------------------
    #       Define optimizer
    # ---------------------------------

    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': config.joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet.add_on_layers.parameters(), 'lr': config.joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': config.joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=config.joint_lr_step_size, gamma=0.1)

    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': config.warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': config.warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': config.last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # ---------------------------------
    #       Train model
    # ---------------------------------
    logging.info('Start training')
    best_accu = 0

    for epoch in progress_bar(range(config.num_train_epochs), total=config.num_train_epochs, leave=True):
        logging.info('epoch: \t{0}'.format(epoch))

        if epoch < config.num_warm_epochs:
            tnt.warm_only(model=ppnet_multi)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=config.coefs)
        else:
            tnt.joint(model=ppnet_multi)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=config.coefs)

        accu = tnt.test(model=ppnet_multi, dataloader=val_loader, class_specific=class_specific)
        if accu > best_accu:
            best_accu = accu
        save_model_w_condition(model=ppnet_multi.module, model_dir=model_dir, model_name='best_model', epoch=epoch, accu=accu, target_accu=best_accu)

        if epoch >= config.push_start and epoch in config.push_epochs:
            push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=None, #img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True) #, log=log
            accu = tnt.test(model=ppnet_multi, dataloader=val_loader, class_specific=class_specific) 
            if accu > best_accu:
                best_accu = accu
            save_model_w_condition(model=ppnet_multi.module, model_dir=model_dir, model_name='best_model', epoch=epoch, accu=accu, target_accu=best_accu)

            if config.prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi) 
                for i in progress_bar(range(20), total=20, leave=True):
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=config.coefs) 
                    accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
                                    class_specific=class_specific) 
                    if accu > best_accu:
                        best_accu = accu
                    save_model_w_condition(model=ppnet_multi.module, model_dir=model_dir, model_name='best_model', epoch=epoch, accu=accu, target_accu=best_accu)
    
    _ = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, testing=True)
    logging.info(f"Test accuracy: {accu}")

    logging.info("Finished")

if __name__ == "__main__":
    parse_args(config)

    wandb.login() # log in to wandb

    train(config)