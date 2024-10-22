
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import random
import numpy as np

# For the CUB dataset
class RandomRotationWithFlip:
    def __init__(self, degrees, flip_prob=0.5):
        self.rotation = transforms.RandomRotation(degrees)
        self.flip_prob = flip_prob

    def __call__(self, img):
        img = self.rotation(img)
        if random.random() < self.flip_prob:
            img = transforms.functional.hflip(img)
        return img

class RandomSkewWithFlip:
    def __init__(self, distortion_scale, flip_prob=0.5):
        self.skew = transforms.RandomPerspective(distortion_scale, p=1.0)
        self.flip_prob = flip_prob

    def __call__(self, img):
        img = self.skew(img)
        if random.random() < self.flip_prob:
            img = transforms.functional.hflip(img)
        return img

class RandomShearWithFlip:
    def __init__(self, shear, flip_prob=0.5):
        self.shear = transforms.RandomAffine(degrees=0, shear=shear)
        self.flip_prob = flip_prob

    def __call__(self, img):
        img = self.shear(img)
        if random.random() < self.flip_prob:
            img = transforms.functional.hflip(img)
        return img

# For numpy arrays (AwA2 dataset)
class RandomRotationWithFlipNP:
    def __init__(self, degrees, flip_prob=0.5):
        self.rotation = transforms.RandomRotation(degrees)
        self.flip_prob = flip_prob

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = self.rotation(img)
        if random.random() < self.flip_prob:
            img = transforms.functional.hflip(img)
        return np.array(img)

class RandomSkewWithFlipNP:
    def __init__(self, distortion_scale, flip_prob=0.5):
        self.skew = transforms.RandomPerspective(distortion_scale, p=1.0)
        self.flip_prob = flip_prob

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = self.skew(img)
        if random.random() < self.flip_prob:
            img = transforms.functional.hflip(img)
        return np.array(img)

class RandomShearWithFlipNP:
    def __init__(self, shear, flip_prob=0.5):
        self.shear = transforms.RandomAffine(degrees=0, shear=shear)
        self.flip_prob = flip_prob

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = self.shear(img)
        if random.random() < self.flip_prob:
            img = transforms.functional.hflip(img)
        return np.array(img)

# import Augmentor
# def makedir(path):
#     '''
#     if path does not exist in the file system, create it
#     '''
#     if not os.path.exists(path):
#         os.makedirs(path)

# datasets_root_dir = './datasets/cub200_cropped/'
# dir = datasets_root_dir + 'train_cropped/'
# target_dir = datasets_root_dir + 'train_cropped_augmented/'

# makedir(target_dir)
# folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
# target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

# for i in range(len(folders)):
#     fd = folders[i]
#     tfd = target_folders[i]
#     # rotation
#     p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
#     p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
#     p.flip_left_right(probability=0.5)
#     for i in range(10):
#         p.process()
#     del p
#     # skew
#     p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
#     p.skew(probability=1, magnitude=0.2)  # max 45 degrees
#     p.flip_left_right(probability=0.5)
#     for i in range(10):
#         p.process()
#     del p
#     # shear
#     p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
#     p.shear(probability=1, max_shear_left=10, max_shear_right=10)
#     p.flip_left_right(probability=0.5)
#     for i in range(10):
#         p.process()
#     del p
#     # random_distortion
#     #p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
#     #p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
#     #p.flip_left_right(probability=0.5)
#     #for i in range(10):
#     #    p.process()
#     #del p


