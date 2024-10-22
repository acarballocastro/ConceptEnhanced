import os
import torch
import numpy as np

def save_model_w_condition(model, model_dir, model_name, epoch, accu, target_accu):
    '''
    model: this is not the multigpu model
    model_dir: directory to save the model
    model_name: name of the model
    epoch: number of epochs
    accu: accuracy
    target_accu: target accuracy
    '''
    if accu >= target_accu:
        print('\tabove {0:.2f}%'.format(target_accu * 100))
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, os.path.join(model_dir, (model_name + '.pth')))

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1
