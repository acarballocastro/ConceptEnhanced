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