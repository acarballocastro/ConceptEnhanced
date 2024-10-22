import os, random
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate_concept_matrix(m):
    """
    Generate a matrix with all the potential combinations based on the input mask tensor m
    Args:
        m: Mask tensor
    Returns:
        matrix: Generated matrix
    """
    active_indices = torch.nonzero(m).view(-1)
    num_active = active_indices.size(0)
    num_combinations = 2 ** num_active

    matrix = torch.zeros(num_combinations, len(m), dtype=torch.int32)
    
    binary_representations = [(i >> bit) & 1 for i in range(num_combinations) for bit in range(num_active)]
    
    for i in range(num_combinations):
        for j, index in enumerate(active_indices):
            matrix[i, index] = binary_representations[i * num_active + j]

    return matrix

def one_batch(dl):
    return next(iter(dl))
        
def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    train_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size + int(.25*args.img_size)),  # args.img_size + 1/4 *args.img_size
        T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, args.train_folder), transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, args.val_folder), transform=val_transforms)
    
    if args.slice_size>1:
        train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), args.slice_size))
        val_dataset = torch.utils.data.Subset(val_dataset, indices=range(0, len(val_dataset), args.slice_size))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataloader, val_dataset

def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)