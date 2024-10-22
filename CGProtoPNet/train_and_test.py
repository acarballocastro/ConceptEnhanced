import time
import torch
from fastprogress import progress_bar 
import logging
import wandb

import itertools

from CGProtoPNet.utils.helpers import list_of_distances, make_one_hot

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%d-%m %I:%M:%S")

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, testing=False, use_l1_mask=True,
                   coefs=None): 
    '''
    Train or test the model on the given dataloader
    @param model: the model to be trained or tested
    @param dataloader: the dataloader to be used
    @param optimizer: the optimizer to be used for training
    @param class_specific: always set to True
    @param testing: if testing mode
    @param use_l1_mask: whether to use l1 mask
    @param coefs: coefficients for the loss function
    '''
    is_train = optimizer is not None
    start = time.time()

    n_examples = 0
    n_samples_batch = 0
    n_batches = 0
    n_correct, n_incorrect = 0, 0
    n_tp_batch, n_tp_fn_batch, n_tp_fp_batch = 0, 0, 0 
    n_instance_batch = 0
    n_tp_total, n_tp_fn_total, n_tp_fp_total = 0, 0, 0

    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0

    for i, (image, concepts) in enumerate(progress_bar(dataloader, leave=True)):
        input = image.cuda()
        target = concepts.to(torch.float).cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # indices: according to whether concept is positive or negative
                # indices_flipped: opposite of indices for the other prototypes
                indices = concepts * torch.arange(model.module.num_classes) + (1 - concepts) * (torch.arange(model.module.num_classes) + model.module.num_classes)
                indices_flipped = (1 - concepts) * torch.arange(model.module.num_classes) + concepts * (torch.arange(model.module.num_classes) + model.module.num_classes)

                batch_size = output.shape[0]
                min_distances_clust = torch.zeros((batch_size*model.module.num_classes, model.module.num_prototypes_per_class))
                min_distances_sep = torch.zeros((batch_size*model.module.num_classes, model.module.num_prototypes_per_class))

                for n, k in itertools.product(range(batch_size), range(model.module.num_classes)):
                    start_index_clust = indices[n, k]*model.module.num_prototypes_per_class
                    start_index_sep = indices_flipped[n, k]*model.module.num_prototypes_per_class    

                    min_distances_clust[n*model.module.num_classes+k] = min_distances[n, start_index_clust:start_index_clust+model.module.num_prototypes_per_class]
                    min_distances_sep[n*model.module.num_classes+k] = min_distances[n, start_index_sep:start_index_sep+model.module.num_prototypes_per_class]

                cluster_cost = torch.mean(torch.min(min_distances_clust, axis = 1)[0])
                separation_cost = torch.mean(torch.min(min_distances_sep, axis = 1)[0])
                
                if use_l1_mask:
                    l1 = (model.module.last_layer.weight * model.module.l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                raise ValueError
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # for evaluation statistics
            predicted = (output > 0).float()
            
            n_examples += target.size(0)*target.size(1) # batch_size*num_classes (concepts)
            n_samples_batch += target.size(0) # batch_size
            
            # accuracy and hamming loss
            n_correct += (predicted == target).sum().item()
            n_incorrect += (predicted != target).sum().item()

            # macro-averaged F1
            n_tp_batch += torch.sum(predicted*target, axis = 0)            
            n_tp_fn_batch += torch.sum(target, axis = 0)
            n_tp_fp_batch += torch.sum(predicted, axis = 0)
            
            # instance-averaged F1
            n_instance_batch += torch.sum(2*torch.sum(predicted*target, axis = 1)/(torch.sum(target, axis = 1) + torch.sum(predicted, axis = 1))).item()

            # micro-averaged F1
            n_tp_total += torch.sum(predicted*target).item()
            n_tp_fn_total += torch.sum(target).item()
            n_tp_fp_total += torch.sum(predicted).item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end = time.time()

    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    
    # evaluation statistics
    accuracy = n_correct / n_examples
    hamming_loss = n_incorrect / n_examples
    
    macro_f1 = (1/model.module.num_classes)*torch.sum((2*n_tp_batch/(n_tp_fn_batch+ n_tp_fp_batch))).item()
    instance_f1 = n_instance_batch / n_samples_batch
    micro_f1 = 2*n_tp_total / (n_tp_fn_total + n_tp_fp_total)

    if is_train:
        wandb.log({
            'train/accu': accuracy * 100,
            'train/hamming': hamming_loss * 100,
            'train/macro_f1': macro_f1,
            'train/instance_f1': instance_f1,
            'train/micro_f1': micro_f1,
            'train/l1': l1,
            })
        wandb.log({
            'train/time': end - start,
            'train/cross_ent': total_cross_entropy / n_batches,
            'train/cluster': total_cluster_cost / n_batches
            })
        wandb.log({
                'train/separation': total_separation_cost / n_batches
                })
        wandb.log({
            'train/p dist pair': p_avg_pair_dist.item(),
        })

    elif testing:
        wandb.log({
            'test/accu': accuracy * 100,
            'test/hamming': hamming_loss * 100,
            'test/macro_f1': macro_f1,
            'test/instance_f1': instance_f1,
            'test/micro_f1': micro_f1,
            'test/l1': l1,
            })
        wandb.log({
            'test/time': end - start,
            'test/cross_ent': total_cross_entropy / n_batches,
            'test/cluster': total_cluster_cost / n_batches
            })
        wandb.log({
                'test/separation': total_separation_cost / n_batches
                })
        wandb.log({
            'test/p dist pair': p_avg_pair_dist.item(),
        })

    else:
        wandb.log({
            'val/accu': accuracy * 100,
            'val/hamming': hamming_loss * 100,
            'val/macro_f1': macro_f1,
            'val/instance_f1': instance_f1,
            'val/micro_f1': micro_f1,
            'val/l1': l1,
            })
        wandb.log({
            'val/time': end - start,
            'val/cross_ent': total_cross_entropy / n_batches,
            'val/cluster': total_cluster_cost / n_batches
            })
        wandb.log({
                'val/separation': total_separation_cost / n_batches
                })
        wandb.log({
            'val/p dist pair': p_avg_pair_dist.item(),
        })

    return accuracy

def train(model, dataloader, optimizer, class_specific=False, coefs=None): 
    # Train the model on the given dataloader
    assert(optimizer is not None)
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs) 


def test(model, dataloader, class_specific=False, testing=False):
    # Validate or test the model on the given dataloader
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, testing=testing) 

def last_only(model): 
    # Freeze all layers except the last layer
    logging.info('last layer')
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

def warm_only(model): 
    # Freeze all layers except the last layer and the prototype vectors
    logging.info('warm')
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

def joint(model): 
    # Unfreeze all layers
    logging.info('joint')
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
