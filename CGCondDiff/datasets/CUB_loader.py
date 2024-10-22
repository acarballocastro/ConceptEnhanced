import pickle
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CUB_DatasetGenerator(Dataset):
    """CUB dataset object"""

    def __init__(self, train_path, root_path, transform=None):
        """
        Args:
            train_path (string): Path to the pkl file with annotations.
            root_path (string): Root directory with all the images.
            attr_index (int): Index of the attribute to be used. Default = 0
            transform (callable, optional): Optional transform to be applied. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.root_path = root_path
        with open(train_path, 'rb') as f:
            self.data = pickle.load(f)

        self.dataset_paths()
        self.transform = transform
        
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

        if self.transform != None:
            img = self.transform(img)

        # Return a list of the image and the attribute vector
        return [img, img_attr]
    
    def __len__(self):
        # Returns the length of the dataset
        return len(self.data)

def get_CUB_dataloaders(args, dataloader=True, shuffle=True):
    """
    Args:
        train_path (string): Path to the pkl file with annotations.
        root_path (string): Root directory with all the images.
        attr_index (int): Index of the attribute to be used. Default = 0
        dataloader (bool): Set to True to return a DataLoader object. Default = True
        batch_size (int): Number of images in each batch
        num_workers (int): Number of subprocesses to use for data loading. Default = 0
        shuffle (bool): Set to True to have the data reshuffled at every epoch. Default = True
    """

    # Define the transformations for the dataset
    train_transform = transforms.Compose([
        transforms.Resize(args.img_size + int(.25*args.img_size)),  # args.img_size + 1/4 *args.img_size
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create the dataset
    dataset = CUB_DatasetGenerator(args.train_path, args.root_path, train_transform)

    if dataloader:
        # Create the dataloaders
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
        return dataloader

    else:
        return dataset