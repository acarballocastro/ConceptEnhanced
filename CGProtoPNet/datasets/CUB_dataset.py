import pickle
import numpy as np
import pandas as pd
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

from datasets.img_aug import RandomRotationWithFlip, RandomShearWithFlip, RandomSkewWithFlip

class CUB_DatasetGenerator(Dataset):
    """CUB dataset object"""

    def __init__(self, split_path, root_path, attr_index=0, transform=None):
        """
        Args:
            split_path (string): Path to the pkl file with annotations.
            root_path (string): Root directory with all the images.
            attr_index (int): Index of the attribute to be used. Default = 0
            transform (callable, optional): Optional transform to be applied. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.root_path = root_path
        with open(split_path, 'rb') as f:
            self.data = pickle.load(f)

        self.dataset_paths()
        self.transform = transform
        self.attr_index = attr_index

        self.bounding_boxes = pd.read_csv('bounding_boxes.txt', sep = ' ', header = None, index_col = 0)

    def dataset_paths(self):
        for i in range(len(self.data)):
            parts = self.data[i]['img_path'].split('/')
            index = parts.index('images')
            end_path = '/'.join(parts[index:])
            self.data[i]['img_path'] = os.path.join(self.root_path, 'CUB_200_2011/', end_path)

    def __getitem__(self, idx):
        # Gets an element of the dataset
        img_data = self.data[idx]
        img_path = img_data['img_path']
        img = Image.open(img_path).convert('RGB')
        img_label = img_data['class_label']
        img_attr = np.array(img_data['attribute_label'])

        bounding_box = tuple(self.bounding_boxes.loc[img_data['id']])
        img = img.crop((bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]))

        if self.transform != None:
            img = self.transform(img)

        # Return a list of the image and the attribute vector
        if self.attr_index is None:
            return [img, img_attr]
        else:
            return [img, img_attr[self.attr_index]]
    
    def __len__(self):
        # Returns the length of the dataset
        return len(self.data)



def get_CUB_dataloaders(
    config, shuffle=True
):
    """Returns a dictionary of data loaders for the CUB dataset, for the training, validation, and test sets."""

    push_transform = transforms.Compose(
        [

            transforms.Resize(size=(config.img_size, config.img_size)),
            transforms.ToTensor(),  # implicitly divides by 255
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    

    aug_transform =  transforms.Compose(
        [
            RandomRotationWithFlip(degrees=15, flip_prob=0.5),
            RandomSkewWithFlip(distortion_scale=0.2, flip_prob=0.5),
            RandomShearWithFlip(shear=10, flip_prob=0.5),
            transforms.Resize(size=(config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if config.img_aug:
        train_dataset = CUB_DatasetGenerator(config.train_path, config.data_path, config.attr_index, aug_transform)
        test_dataset = CUB_DatasetGenerator(config.test_path, config.data_path, config.attr_index, aug_transform)
        val_dataset = CUB_DatasetGenerator(config.val_path, config.data_path, config.attr_index, aug_transform)
        train_push_dataset = CUB_DatasetGenerator(config.train_path, config.data_path, config.attr_index, push_transform)
    else:
        train_dataset = CUB_DatasetGenerator(config.train_path, config.data_path, config.attr_index, push_transform)
        test_dataset = CUB_DatasetGenerator(config.test_path, config.data_path, config.attr_index, push_transform)
        val_dataset = CUB_DatasetGenerator(config.val_dataset, config.data_path, config.attr_index, push_transform)
        train_push_dataset = CUB_DatasetGenerator(config.train_path, config.data_path, config.attr_index, push_transform)

    train_loader =  DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=False, # As per the original code
        num_workers=4,
        pin_memory=False, # As per the original code
        generator=torch.Generator().manual_seed(config.seed),
        drop_last=False,)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        generator=torch.Generator().manual_seed(config.seed),
        drop_last=False,)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        generator=torch.Generator().manual_seed(config.seed),
        drop_last=False,)

    train_push_loader = DataLoader(
        train_push_dataset,
        batch_size=config.train_push_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        generator=torch.Generator().manual_seed(config.seed),
        drop_last=False,)

    return train_loader, test_loader, val_loader, train_push_loader