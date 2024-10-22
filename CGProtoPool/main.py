import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import PrototypeChooser
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
    data_type = 'birds',
    data_path = '.',
    train_path ='.',
    test_path='.',
    val_path='.',
    img_size=224,
    attr_index=None,
    img_aug=True,

    batch_size=80,
    lr=0.001, 
    epochs=100,
    push_start=20,
    when_push=2,
    seed=None, 
    checkpoint=None,

    num_descriptive=10, 
    num_prototypes=200,
    num_classes=200,

    arch='resnet34',
    add_on_layers_type='log',
    prototype_activation_function='log',

    clst_weight=0.8,
    sep_weight=-0.08,
    l1_weight=1e-4,
    orth_p_weight=1,
    orth_c_weight=1,
    run_name='test',

    earlyStopping=None,
    results='.',
    ppnet_path=None,
    warmup_time=100,
    gumbel_time=10,
    proto_depth=128,
    gpuid='0', 
    proto_img_dir='img'
)

# Util functions (from original code)
def save_model(model, path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }, path)

def load_model(model, path, device):
    if device.type == 'cuda':
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'\033[0;32mLoad model form: {path}\033[0m')
    return model, checkpoint['epoch']

def adjust_learning_rate(optimizer, rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= rate

def parse_args(config):
    parser = argparse.ArgumentParser(description='PrototypeGraph')
    parser.add_argument('--evaluate', '-e', action='store_true', help='The run evaluation training model')

    parser.add_argument('--data_type', default=config.data_type, choices=['birds', 'awa'])
    parser.add_argument('--data_path', default=config.data_path, help='Path to root folder with data')
    parser.add_argument('--train_path', default=config.train_path, help='Path to train data')
    parser.add_argument('--test_path', default=config.test_path, help='Path to tets data')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--attr_index', type=int, default=config.attr_index, help='concept to use as label')

    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='input batch size for training (default: 80)')
    parser.add_argument('--lr', type=float, default=config.lr, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs to train (default: 100)')
    parser.add_argument('--push_start', type=int, default=config.push_start)
    parser.add_argument('--when_push', type=int, default=config.when_push)
    parser.add_argument('--no_cuda', action='store_true', help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--checkpoint', type=str, default=config.checkpoint)

    parser.add_argument('--num_descriptive', type=int, default=config.num_descriptive, help='prototypes at most assigned to each class')
    parser.add_argument('--num_prototypes', type=int, default=config.num_prototypes)
    parser.add_argument('--num_classes', type=int, default=config.num_classes)

    parser.add_argument('--arch', type=str, default=config.arch)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--add_on_layers_type', type=str, default=config.add_on_layers_type)
    parser.add_argument('--prototype_activation_function', type=str, default=config.prototype_activation_function)

    parser.add_argument('--clst_weight', type=float, default=config.clst_weight)
    parser.add_argument('--sep_weight', type=float, default=config.sep_weight)
    parser.add_argument('--l1_weight', type=float, default=config.l1_weight)
    parser.add_argument('--orth_p_weight', type=float, default=config.orth_p_weight)
    parser.add_argument('--orth_c_weight', type=float, default=config.orth_c_weight)
    parser.add_argument('--run_name', type=str, default=config.run_name)

    parser.add_argument('--use_thresh', action='store_true')
    parser.add_argument('--earlyStopping', type=int, default=config.earlyStopping, help='Number of epochs to early stopping')
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--results', default=config.results, help='Path to dictionary where will be save results.')
    parser.add_argument('--ppnet_path', default=config.ppnet_path)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--warmup_time', default=config.warmup_time, type=int)
    parser.add_argument('--gumbel_time', default=config.gumbel_time, type=int)
    parser.add_argument('--proto_depth', default=config.proto_depth, type=int)
    parser.add_argument('--last_layer', action='store_true') 
    parser.add_argument('--inat', action='store_true')
    parser.add_argument('--mixup_data', action='store_true')
    parser.add_argument('--push_only', action='store_true')
    parser.add_argument('--gpuid', nargs=1, type=str, default=config.gpuid) # python3 main.py -gpuid=0,1,2,3
    parser.add_argument('--proto_img_dir', type=str, default=config.proto_img_dir)
    parser.add_argument('--pp_ortho', action='store_true')
    parser.add_argument('--pp_gumbel', action='store_true')

    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

def learn_model(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')

    # Set parameters -------------------------------------------------------------------------------------

    start_val = 1.3
    end_val = 10 ** 3
    epoch_interval = args.gumbel_time
    alpha = (end_val / start_val) ** 2 / epoch_interval

    # tau for Gumbel-Softmax distribution
    def lambda1(epoch): return start_val * np.sqrt(alpha *
                                                   (epoch)) if epoch < epoch_interval else end_val

    clst_weight = args.clst_weight
    sep_weight = args.sep_weight
    l1_weight = args.l1_weight
    orth_p_weight = args.orth_p_weight
    orth_c_weight = args.orth_c_weight

    if args.seed is None:  # 1234
        args.seed = np.random.randint(10, 10000, size=1)[0]
    torch.manual_seed(args.seed)
    kwargs = {}
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        kwargs.update({'num_workers': 9, 'pin_memory': True})
    
    if args.data_type == 'birds':
        train_loader, val_loader, test_loader, train_push_loader = get_CUB_dataloaders(args)
    elif args.data_type == 'awa':
        train_loader, val_loader, test_loader, train_push_loader = get_AwA_dataloaders(
            classes_file=os.path.join(args.data_path, 'Animals_with_Attributes2/classes.txt'),
            data_path=args.data_path, 
            batch_size=args.batch_size, 
            preload=True
        )
    else:
        raise ValueError

    # Instantiate model -------------------------------------------------------------------------------------
    model = PrototypeChooser(
        num_prototypes=args.num_prototypes,
        num_descriptive=args.num_descriptive,
        num_classes=args.num_classes,
        use_thresh=args.use_thresh,
        arch=args.arch,
        pretrained=args.pretrained,
        add_on_layers_type=args.add_on_layers_type,
        prototype_activation_function=args.prototype_activation_function,
        proto_depth=args.proto_depth,
        use_last_layer=args.last_layer,
        inat=args.inat,
    )
    if args.ppnet_path:
        model.load_state_dict(torch.load(args.ppnet_path, map_location='cpu')[
                              'model_state_dict'], strict=True)
        print('Successfully loaded ' + args.ppnet_path)

    model.to(device)
    if args.warmup:
        model.features.requires_grad_(False)
        model.last_layer.requires_grad_(True)
        if args.ppnet_path:
            model.add_on_layers.requires_grad_(False)
            model.prototype_vectors.requires_grad_(False)
    if args.checkpoint:
        model, start_epoch = load_model(model, args.checkpoint, device)
    else:
        start_epoch = 0

    # Optimizer and loss function -------------------------------------------------------------------------------------
    warm_optimizer = torch.optim.Adam(
        [{'params': model.add_on_layers.parameters(), 'lr': 3 * args.lr, 'weight_decay': 1e-3},
         {'params': model.proto_presence, 'lr': 3 * args.lr},
         {'params': model.prototype_vectors, 'lr': 3 * args.lr}]
    )
    joint_optimizer = torch.optim.Adam(
        [{'params': model.features.parameters(), 'lr': args.lr / 10, 'weight_decay': 1e-3},
         {'params': model.add_on_layers.parameters(), 'lr': 3 * args.lr, 'weight_decay': 1e-3},
         {'params': model.proto_presence, 'lr': 3 * args.lr},
         {'params': model.prototype_vectors, 'lr': 3 * args.lr}]
    )
    push_optimizer = torch.optim.Adam(
        [{'params': model.last_layer.parameters(), 'lr': args.lr / 10,
          'weight_decay': 1e-3}, ]
    )
    optimizer = warm_optimizer
    criterion = torch.nn.BCEWithLogitsLoss()

    # Run information -------------------------------------------------------------------------------------
    info = f'{args.data_type}_descriptive-{args.num_descriptive}_prototypes-{args.num_prototypes}' \
           f'_lr-{args.lr}' \
           f'_{args.arch}_{"True" if args.pretrained else f"No"}' \
           f'_{args.add_on_layers_type}_{args.prototype_activation_function}' \
           f'{"_warmup" if args.warmup else ""}' \
           f'{"_ll" if args.last_layer else ""}' \
           f'{"_mixup" if args.mixup_data else ""}' \
           f'{"_iNaturalist" if args.inat else ""}' \
           f'_seed-{args.seed}' \
           f'_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}'
    # Directory to save results
    dir_checkpoint = f'{args.results}/{args.run_name}/checkpoint'
    if args.proto_img_dir:
        proto_img_dir = f'{args.results}/{args.run_name}/img_proto'
        Path(proto_img_dir).mkdir(parents=True, exist_ok=True)
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    

    ####################################
    #          learning model          #
    ####################################
    min_val_loss = np.Inf
    max_val_tst = 0
    epochs_no_improve = 0
    steps = False

    model_multi = torch.nn.DataParallel(model)

    if not args.push_only:
        print('Model learning')
        # for epoch in epoch_tqdm:
        for epoch in progress_bar(range(start_epoch, args.epochs), total=int(args.epochs)-start_epoch, leave=True):
            gumbel_scalar = lambda1(epoch) if args.pp_gumbel else 0

            ####################################
            #            Warm-up               #
            ####################################
            if args.warmup and args.warmup_time == epoch:
                print("Warm up")
                model.features.requires_grad_(True)
                optimizer = joint_optimizer
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.1)
                steps = True
                print("Warm up ends")

            model.train()

            ####################################
            #            train step            #
            ####################################
            trn_loss = 0
            n_examples, n_samples_batch, n_batches = 0, 0, 0
            n_correct, n_incorrect = 0, 0
            n_tp_batch, n_tp_fn_batch, n_tp_fp_batch = 0, 0, 0
            n_instance_batch = 0
            n_tp_total, n_tp_fn_total, n_tp_fp_total = 0, 0, 0
            if epoch > 0:
                for i, (data, concepts) in enumerate(progress_bar(train_loader, leave=False)):

                    label_p = concepts.numpy().tolist()
                    data = data.to(device)
                    label = concepts.to(torch.float).to(device)
                    
                    if args.mixup_data:
                        data, targets_a, targets_b, lam = mixup_data(
                            data, label, 0.5)

                    # ===================forward=====================
                    prob, min_distances, proto_presence = model_multi(
                        data, gumbel_scale=gumbel_scalar)
                    np.savez_compressed(f'{dir_checkpoint}/pp_{epoch * 80 + i}.pth', proto_presence.detach().cpu().numpy())

                    if args.mixup_data:
                        entropy_loss = lam * \
                            criterion(prob, targets_a) + (1 - lam) * \
                            criterion(prob, targets_b)
                    else:
                        entropy_loss = criterion(prob, label)
                    orthogonal_loss_p = torch.Tensor([0]).cuda()
                    orthogonal_loss_c = torch.Tensor([0]).cuda()
                    if args.pp_ortho:
                        for c in range(0, model_multi.module.proto_presence.shape[0], 1000):
                            # Orthogonal loss per class
                            orthogonal_loss_p += \
                                torch.nn.functional.cosine_similarity(model_multi.module.proto_presence.unsqueeze(2)[c:c+1000],
                                                                      model_multi.module.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                            
                            # Orthogonal loss positive-negative concepts
                            positive_concepts = model_multi.module.proto_presence[:model_multi.module.num_classes].unsqueeze(2)
                            negative_concepts = model_multi.module.proto_presence[model_multi.module.num_classes:].unsqueeze(-1)
                            orthogonal_loss_c += torch.nn.functional.cosine_similarity(positive_concepts[c:c+1000], negative_concepts[c:c+1000], dim=1).sum()

                        orthogonal_loss_p = (orthogonal_loss_p / (args.num_descriptive * args.num_classes * 2) - 1) 
                        orthogonal_loss_c = (orthogonal_loss_c / (args.num_descriptive * args.num_classes))

                    indices = label.cpu() * torch.arange(model_multi.module.num_classes) + (1 - label.cpu()) * (torch.arange(model_multi.module.num_classes) + model_multi.module.num_classes)
                    inverted_indices = (1 - label.cpu()) * torch.arange(model_multi.module.num_classes) + label.cpu() * (torch.arange(model_multi.module.num_classes) + model_multi.module.num_classes)
                    
                    inverted_proto_presence = 1 - proto_presence

                    proto_presence = proto_presence[indices.long()] #label_p
                    inverted_proto_presence = inverted_proto_presence[inverted_indices.long()] # label_p

                    clst_loss_val = dist_loss(model, min_distances, proto_presence, args.num_descriptive, indices)  
                    sep_loss_val = dist_loss(model, min_distances, inverted_proto_presence, args.num_prototypes - args.num_descriptive, inverted_indices)  
                    prototypes_of_correct_class = proto_presence.sum(dim=-1).detach()
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    avg_separation_cost = torch.sum(min_distances.unsqueeze(1).repeat(1,model_multi.module.num_classes,1) * prototypes_of_wrong_class, dim=-1) / torch.sum(prototypes_of_wrong_class, dim=-1)
                    avg_separation_cost = torch.mean(avg_separation_cost)

                    l1 = (model.last_layer.weight * model_multi.module.l1_mask).norm(p=1)

                    loss = entropy_loss + clst_loss_val * clst_weight + \
                        sep_loss_val * sep_weight + l1_weight * l1 + orth_p_weight * orthogonal_loss_p + orth_c_weight * orthogonal_loss_c

                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # ===================logging====================
                    wandb.log({
                        'train/loss': loss,
                        'train/entropy': entropy_loss.item(),
                        'train/clst': clst_loss_val.item(),
                        'train/sep': sep_loss_val.item(),
                        'train/l1': l1.item(),
                        'train/avg_sep': avg_separation_cost.item(),
                        'train/orthogonal_loss_p':orthogonal_loss_p.item(),
                        'train/orthogonal_loss_c':orthogonal_loss_c.item(),
                        } #, step=epoch * len(train_loader) + i
                    )

                    trn_loss += loss.item()

                    # ===================metrics====================
                    predicted = (prob > 0).float()

                    n_examples += label.size(0)*label.size(1) # batch_size*num_classes (concepts)
                    n_samples_batch += label.size(0) # batch_size

                    # accuracy and hamming loss
                    n_correct += (predicted == label).sum().item()
                    n_incorrect += (predicted != label).sum().item()

                    # macro-averaged F1
                    n_tp_batch += torch.sum(predicted*label, axis = 0)            
                    n_tp_fn_batch += torch.sum(label, axis = 0)
                    n_tp_fp_batch += torch.sum(predicted, axis = 0)

                    # instance-averaged F1
                    n_instance_batch += torch.sum(2*torch.sum(predicted*label, axis = 1)/(torch.sum(label, axis = 1) + torch.sum(predicted, axis = 1))).item()

                    # micro-averaged F1
                    n_tp_total += torch.sum(predicted*label).item()
                    n_tp_fn_total += torch.sum(label).item()
                    n_tp_fp_total += torch.sum(predicted).item()

                    n_batches += 1

                trn_loss /= len(train_loader)

                # evaluation statistics
                trn_acc = n_correct / n_examples
                hamming_loss = n_incorrect / n_examples
                
                macro_f1 = (1/model_multi.module.num_classes)*torch.sum((2*n_tp_batch/(n_tp_fn_batch+ n_tp_fp_batch))).item()
                instance_f1 = n_instance_batch / n_samples_batch
                micro_f1 = 2*n_tp_total / (n_tp_fn_total + n_tp_fp_total)
                
                # logging
                wandb.log({
                'train/accu': trn_acc * 100,
                'train/hamming': hamming_loss * 100,
                'train/macro_f1': macro_f1,
                'train/instance_f1': instance_f1,
                'train/micro_f1': micro_f1,
                })

            if steps:
                lr_scheduler.step()

            ####################################
            #          validation step         #
            ####################################
            model_multi.eval()
            tst_loss = np.zeros((args.num_classes, 1))
            prob_leaves = np.zeros((args.num_classes, 1))
            n_examples, n_samples_batch, n_batches = 0, 0, 0
            n_correct, n_incorrect = 0, 0
            n_tp_batch, n_tp_fn_batch, n_tp_fp_batch = 0, 0, 0
            n_instance_batch = 0
            n_tp_total, n_tp_fn_total, n_tp_fp_total = 0, 0, 0
            with torch.no_grad():
                for i, (data, concepts) in enumerate(progress_bar(val_loader, leave=False)):
                    data = data.to(device)
                    label_p = concepts.detach().numpy().tolist()
                    label = concepts.to(torch.float).to(device)

                    # ===================forward=====================

                    prob, min_distances, proto_presence = model_multi(data, gumbel_scale=gumbel_scalar)
                    loss = criterion(prob, label)
                    entropy_loss = loss

                    orthogonal_loss_p = torch.Tensor([0]).cuda() 
                    orthogonal_loss_c = torch.Tensor([0]).cuda()                                                                                                                                            
                    if args.pp_ortho: 
                        for c in range(0, model_multi.module.proto_presence.shape[0], 1000):
                            # Orthogonal loss per class
                            orthogonal_loss_p += \
                                torch.nn.functional.cosine_similarity(model_multi.module.proto_presence.unsqueeze(2)[c:c+1000],
                                                                      model_multi.module.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                            
                            # Orthogonal loss positive-negative concepts
                            positive_concepts = model_multi.module.proto_presence[:model_multi.module.num_classes].unsqueeze(2)
                            negative_concepts = model_multi.module.proto_presence[model_multi.module.num_classes:].unsqueeze(-1)
                            orthogonal_loss_c += torch.nn.functional.cosine_similarity(positive_concepts[c:c+1000], negative_concepts[c:c+1000], dim=1).sum()

                        orthogonal_loss_p = (orthogonal_loss_p / (args.num_descriptive * args.num_classes * 2) - 1)
                        orthogonal_loss_c = (orthogonal_loss_c / (args.num_descriptive * args.num_classes))

                    indices = label.cpu() * torch.arange(model_multi.module.num_classes) + (1 - label.cpu()) * (torch.arange(model_multi.module.num_classes) + model_multi.module.num_classes)
                    inverted_indices = (1 - label.cpu()) * torch.arange(model_multi.module.num_classes) + label.cpu() * (torch.arange(model_multi.module.num_classes) + model_multi.module.num_classes)
                    inverted_proto_presence = 1 - proto_presence

                    l1 = (model_multi.module.last_layer.weight * model_multi.module.l1_mask).norm(p=1)

                    proto_presence = proto_presence[indices.long()] #label_p
                    inverted_proto_presence = inverted_proto_presence[inverted_indices.long()] # label_p

                    clst_loss_val = dist_loss(model_multi.module, min_distances, proto_presence, args.num_descriptive, indices) * clst_weight
                    sep_loss_val = dist_loss(model_multi.module, min_distances, inverted_proto_presence, args.num_prototypes - args.num_descriptive, inverted_indices) * sep_weight
                    loss = entropy_loss + clst_loss_val + sep_loss_val + orth_p_weight * orthogonal_loss_p + orth_c_weight * orthogonal_loss_c + l1_weight * l1
                    tst_loss += loss.item()
                    
                    predicted = (prob > 0).float()

                    n_examples += label.size(0)*label.size(1) # batch_size*num_classes (concepts)
                    n_samples_batch += label.size(0) # batch_size

                    # accuracy and hamming loss
                    n_correct += (predicted == label).sum().item()
                    n_incorrect += (predicted != label).sum().item()

                    # macro-averaged F1
                    n_tp_batch += torch.sum(predicted*label, axis = 0)            
                    n_tp_fn_batch += torch.sum(label, axis = 0)
                    n_tp_fp_batch += torch.sum(predicted, axis = 0)

                    # instance-averaged F1
                    n_instance_batch += torch.sum(2*torch.sum(predicted*label, axis = 1)/(torch.sum(label, axis = 1) + torch.sum(predicted, axis = 1))).item()

                    # micro-averaged F1
                    n_tp_total += torch.sum(predicted*label).item()
                    n_tp_fn_total += torch.sum(label).item()
                    n_tp_fp_total += torch.sum(predicted).item()

                    n_batches += 1

            tst_loss /= len(val_loader)

            # evaluation statistics
            tst_acc = n_correct / n_examples
            hamming_loss = n_incorrect / n_examples
            
            macro_f1 = (1/model_multi.module.num_classes)*torch.sum((2*n_tp_batch/(n_tp_fn_batch+ n_tp_fp_batch))).item()
            instance_f1 = n_instance_batch / n_samples_batch
            micro_f1 = 2*n_tp_total / (n_tp_fn_total + n_tp_fp_total)

            ####################################
            #             logger               #
            ####################################

            tst_loss = tst_loss.mean()

            wandb.log({
                'test/loss': tst_loss.mean(),
                'test/entropy': entropy_loss.item(),
                'test/clst': clst_loss_val.item(),
                'test/sep': sep_loss_val.item(),
                'test/l1': l1.item(),
                'test/orthogonal_loss_p':orthogonal_loss_p.item(),
                'test/orthogonal_loss_c':orthogonal_loss_c.item(),
                'test/accu': tst_acc * 100,
                'test/hamming': hamming_loss * 100,
                'test/macro_f1': macro_f1,
                'test/instance_f1': instance_f1,
                'test/micro_f1': micro_f1,
                }
            )

            if trn_loss is None:
                trn_loss = loss.mean().detach()
                trn_loss = trn_loss.cpu().numpy() / len(train_loader)
            print(f'Epoch {epoch}|{args.epochs}, train loss: {trn_loss:.5f}, test loss: {tst_loss.mean():.5f} '
                  f'| acc: {tst_acc:.5f},  orthogonal c: {orthogonal_loss_c.item():.5f}, orthogonal p: {orthogonal_loss_p.item():.5f} '
                  f'(minimal test-loss: {min_val_loss:.5f}, early stop: {epochs_no_improve}|{args.earlyStopping}) - ')

            ####################################
            #  scheduler and early stop step   #
            ####################################
            if (tst_loss.mean() < min_val_loss) or (tst_acc > max_val_tst):
                # save the best model
                if tst_acc > max_val_tst:
                    save_model(model_multi.module, f'{dir_checkpoint}/best_model.pth', epoch)

                epochs_no_improve = 0
                if tst_loss.mean() < min_val_loss:
                    min_val_loss = tst_loss.mean()
                if tst_acc > max_val_tst:
                    max_val_tst = tst_acc
            else:
                epochs_no_improve += 1

            if args.use_scheduler:
                # scheduler.step()
                if epochs_no_improve > 5:
                    adjust_learning_rate(optimizer, 0.95)

            if args.earlyStopping is not None and epochs_no_improve > args.earlyStopping:
                print('\033[1;31mEarly stopping!\033[0m')
                break

    ####################################
    #            push step             #
    ####################################
    print('Model push')
    model_multi.eval()

    ####################################
    #          validation step         #
    ####################################
    tst_loss = np.zeros((args.num_classes, 1))
    n_examples, n_correct = 0, 0
    with torch.no_grad():
        for i, (data, concepts) in enumerate(progress_bar(val_loader, leave=False)):
        
            data = data.to(device)
            label = concepts.to(torch.float).to(device)

            # ===================forward=====================
            prob, min_distances, proto_presence = model_multi(data, gumbel_scale=10e3)

            loss = criterion(prob, label)
            entropy_loss = loss

            l1 = 1e-4 * (model_multi.module.last_layer.weight * model_multi.module.l1_mask).norm(p=1)

            loss = entropy_loss + l1
            tst_loss += loss.item()

            predicted = (prob > 0).float()

            n_examples += label.size(0)*label.size(1) # batch_size*num_classes (concepts)
            n_correct += (predicted == label).sum().item()

        tst_loss /= len(val_loader)
        tst_acc = n_correct / n_examples
    print(f'Before tuning, test loss: {tst_loss.mean():.5f} | acc: {tst_acc:.5f}')

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
                                   dir_for_saving_prototypes=proto_img_dir,
                                   prototype_img_filename_prefix='prototype-img',
                                   prototype_self_act_filename_prefix='prototype-self-act',
                                   prototype_activation_function_in_numpy=None)

    np.save(os.path.join(proto_img_dir, 'bb-receptive_field.npy'), proto_rf_boxes)
    np.save(os.path.join(proto_img_dir, 'bb.npy'), proto_bound_boxes)

    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(model_multi.module.prototype_shape))
    model_multi.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    # ===================fine tune=====================

    print('Fine-tuning')
    max_val_tst = 0
    min_val_loss = 10e5
    for tune_epoch in range(25):
        trn_loss = 0
        model_multi.train()
        for i, (data, concepts) in enumerate(progress_bar(train_loader, leave=False)):
            data = data.to(device)
            label = concepts.to(torch.float).to(device)

            # ===================forward=====================
            if args.mixup_data:
                data, targets_a, targets_b, lam = mixup_data(data, label, 0.5)

            # ===================forward=====================
            prob, min_distances, proto_presence = model_multi(data, gumbel_scale=10e3)

            if args.mixup_data:
                entropy_loss = lam * \
                    criterion(prob, targets_a) + (1 - lam) * \
                    criterion(prob, targets_b)
            else:
                entropy_loss = criterion(prob, label)

            l1 = 1e-4 * (model_multi.module.last_layer.weight * model_multi.module.l1_mask).norm(p=1)

            loss = entropy_loss + l1

            # ===================backward====================
            push_optimizer.zero_grad()
            loss.backward()
            push_optimizer.step()
            trn_loss += loss.item()
            
            wandb.log({
                'train_push/loss': loss,
                'train_push/l1': l1.item(),
                } 
            )
            
        ####################################
        #          validation step         #
        ####################################
        model_multi.eval()
        tst_loss = np.zeros((args.num_classes, 1))
        n_examples, n_correct = 0, 0
        with torch.no_grad():
            for i, (data, concepts) in enumerate(progress_bar(test_loader, leave=False)):

                data = data.to(device)
                label = concepts.to(torch.float).to(device)

                # ===================forward=====================
                prob, min_distances, proto_presence = model_multi(data, gumbel_scale=10e3)

                loss = criterion(prob, label)
                entropy_loss = loss

                l1 = 1e-4 * (model_multi.module.last_layer.weight * model_multi.module.l1_mask).norm(p=1)

                loss = entropy_loss + l1
                tst_loss += loss.item()

                predicted = (prob > 0).float()

                n_examples += label.size(0)*label.size(1) # batch_size*num_classes (concepts)
                n_correct += (predicted == label).sum().item()

            tst_loss /= len(test_loader)
            tst_acc = n_correct / n_examples

        ####################################
        #             logger               #
        ####################################

        tst_loss = tst_loss.mean()

        wandb.log({
            'test_push/acc': tst_acc,
            'test_push/loss': tst_loss.mean(),
            'test_push/entropy': entropy_loss.item(),
            'test_push/l1': l1.item(),
            }
        )

        if trn_loss is None:
            trn_loss = loss.mean().detach()
            trn_loss = trn_loss.cpu().numpy() / len(train_loader)
        print(f'Epoch {tune_epoch}|{5}, train loss: {trn_loss:.5f}, test loss: {tst_loss.mean():.5f} '
              f'| acc: {tst_acc:.5f}, (minimal test-loss: {min_val_loss:.5f}- ')

        ####################################
        #  scheduler and early stop step   #
        ####################################
        if (tst_loss.mean() < min_val_loss) or (tst_acc > max_val_tst):
            # save the best model
            if tst_acc > max_val_tst:
                save_model(model_multi.module, f'{dir_checkpoint}/best_model_push.pth', tune_epoch)
            if tst_loss.mean() < min_val_loss:
                min_val_loss = tst_loss.mean()
            if tst_acc > max_val_tst:
                max_val_tst = tst_acc

        if (tune_epoch + 1) % 5 == 0:
            adjust_learning_rate(push_optimizer, 0.95)

    # writer.close()
    print('Finished training model. Have nice day :)')

# Other helper functions from the original code
def dist_loss(model, min_distances, proto_presence, top_k, indices):
    #         model, [b, p],        [b, c, p, n],  [scalar]
    max_dist = (model.prototype_shape[1]
                * model.prototype_shape[2]
                * model.prototype_shape[3])

    # In this implementation, we add all probabilities for a different prototype across slots
    # and then pick the top k prototypes (all will be different)
    basic_proto = proto_presence.sum(dim=-1).detach()  # [b, c, p]
    _, idx = torch.topk(basic_proto, top_k, dim=2)  # [b, c, n]

    # In this implementation, for each distribution we pick the prototype with the highest
    # probability and that way we get our top k prototypes (some might be the same)
    # idx = torch.argmax(proto_presence, dim=2)

    # In this implementation, we use the method get_map_class_to_prototypes
    # which is already built in the model
    # idx = torch.from_numpy(model.get_map_class_to_prototypes())[indices.long()].cuda()

    binarized_top_k = torch.zeros_like(basic_proto)
    binarized_top_k.scatter_(dim=2, src=torch.ones_like(basic_proto), index=idx)  # [b, c, p]

    inverted_distances, _ = torch.max((max_dist - min_distances).unsqueeze(1) * binarized_top_k, dim=2)  # [b, c]
    inverted_distances = inverted_distances.mean(dim = 1) # [b], mean of all concept
    cost = torch.mean(max_dist - inverted_distances)
    return cost

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
        # this computation currently is not parallelized
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
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
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
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)



if __name__ == '__main__':

    parse_args(config)

    wandb.login() # wandb login

    learn_model(config)

    